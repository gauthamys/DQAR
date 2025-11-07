#!/usr/bin/env python3
"""
Benchmark baseline DiT sampling vs. DQAR-enabled sampling.

The script expects that your DiTPipeline (or custom sampler) accepts an optional
`controller` keyword argument and will invoke the DQAR controller hooks
(`begin_step`, `should_reuse`, `commit`, …) internally. The stock diffusers
pipeline does not yet do this; see README for wiring details.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import urllib.request
from diffusers import DiTPipeline
from urllib.parse import urlparse

from dqar import DQARConfig, DQARController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DiT with and without DQAR.")
    parser.add_argument("--model", default="facebook/DiT-XL-2-256", help="Hugging Face repo id or path.")
    parser.add_argument("--variant", default="fp16", help="Checkpoint variant to load (fp16/bf16/full).")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'.")
    parser.add_argument("--conditioning", choices=("text", "class"), default="text")
    parser.add_argument("--prompts", type=Path, help="Optional text file (one prompt per line).")
    parser.add_argument("--prompt-dataset-url", help="HTTP(S) URL pointing to a text/JSON/JSONL prompt dataset.")
    parser.add_argument("--prompt-dataset-format", choices=("auto", "txt", "json", "jsonl"), default="auto")
    parser.add_argument("--prompt-dataset-field", default="prompt", help="Key to pull from JSON/JSONL rows.")
    parser.add_argument(
        "--prompt-dataset-cache-dir", type=Path, default=Path("prompt_datasets"), help="Where downloaded prompts are cached."
    )
    parser.add_argument(
        "--prompt-dataset-refresh", action="store_true", help="Force re-download of the prompt dataset even if cached."
    )
    parser.add_argument("--class-ids", help="Comma-separated class ids or path to file (one id per line).")
    parser.add_argument("--num-prompts", type=int, default=4, help="Fallback prompt count / class count.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=7.5, help="CFG guidance scale.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_outputs"))
    parser.add_argument("--plot-path", type=Path, default=Path("benchmark_plot.png"))
    parser.add_argument("--save-json", type=Path, default=Path("benchmark_results.json"))
    parser.add_argument("--images-per-prompt", type=int, default=1)
    parser.add_argument("--skip-images", action="store_true", help="Skip saving generated images.")
    parser.add_argument("--dqar-min-prob", type=float, default=0.5)
    parser.add_argument("--dqar-entropy", type=float, default=1.3)
    parser.add_argument("--dqar-snr-min", type=float, default=0.25)
    parser.add_argument("--dqar-snr-max", type=float, default=64.0)
    parser.add_argument("--dqar-cooldown", type=int, default=2)
    parser.add_argument("--dqar-max-gap", type=int, default=4)
    parser.add_argument("--dqar-max-reuse", type=int, default=3)
    return parser.parse_args()


def load_prompts(path: Path | None, count: int) -> List[str]:
    if path and path.exists():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    defaults = [
        "A watercolor skyline at dusk reflecting on water",
        "Close up portrait of a cyberpunk cat wearing neon goggles",
        "Isometric cutaway of a futuristic data center",
        "Macro photograph of a glass flower made of circuits",
    ]
    if count <= len(defaults):
        return defaults[:count]
    extra = [f"Abstract geometry #{i}" for i in range(count - len(defaults))]
    return defaults + extra


def download_prompt_dataset(url: str, cache_dir: Path, refresh: bool) -> Path:
    ensure_dir(cache_dir)
    parsed = urlparse(url)
    filename = Path(parsed.path).name or "prompt_dataset.txt"
    cache_path = cache_dir / filename
    if cache_path.exists() and not refresh:
        print(f"[+] Reusing cached prompt dataset at {cache_path}")
        return cache_path

    print(f"[+] Downloading prompt dataset from {url}")
    with urllib.request.urlopen(url) as response:
        data = response.read()
    cache_path.write_bytes(data)
    return cache_path


def detect_prompt_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if suffix == ".json":
        return "json"
    return "txt"


def _coerce_prompt(entry, field: str | None) -> str | None:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict) and field:
        value = entry.get(field)
        if isinstance(value, str):
            return value.strip()
    return None


def parse_prompt_dataset(path: Path, fmt: str, field: str, limit: int) -> List[str]:
    if fmt == "txt":
        prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    elif fmt == "json":
        raw = json.loads(path.read_text())
        if isinstance(raw, dict):
            field_value = raw.get(field)
            if isinstance(field_value, list):
                raw = field_value
            else:
                raw = raw.get("prompts", [])
        prompts = []
        for entry in raw if isinstance(raw, list) else []:
            prompt = _coerce_prompt(entry, field)
            if prompt:
                prompts.append(prompt)
    elif fmt == "jsonl":
        prompts = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            prompt = _coerce_prompt(entry, field)
            if prompt:
                prompts.append(prompt)
    else:
        raise ValueError(f"Unsupported prompt dataset format: {fmt}")

    if not prompts:
        raise ValueError(f"No prompts could be extracted from {path} using format '{fmt}'.")
    if limit > 0:
        return prompts[:limit]
    return prompts


def resolve_text_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompt_dataset_url:
        dataset_path = download_prompt_dataset(args.prompt_dataset_url, args.prompt_dataset_cache_dir, args.prompt_dataset_refresh)
        dataset_format = detect_prompt_format(dataset_path, args.prompt_dataset_format)
        prompts = parse_prompt_dataset(dataset_path, dataset_format, args.prompt_dataset_field, args.num_prompts)
        if len(prompts) < args.num_prompts:
            print(f"[!] Only {len(prompts)} prompts available from dataset; requested {args.num_prompts}.")
        return prompts
    return load_prompts(args.prompts, args.num_prompts)


def parse_class_ids(
    raw_ids: str | None,
    count: int,
    id2label: Dict[str, str] | None,
) -> List[Tuple[int, str]]:
    def _parse_line(line: str) -> Tuple[int, str]:
        parts = [p.strip() for p in line.split(",", 1)]
        idx = int(parts[0])
        label = parts[1] if len(parts) > 1 else (id2label.get(str(idx), f"class_{idx}") if id2label else f"class_{idx}")
        return idx, label

    if raw_ids:
        path = Path(raw_ids)
        if path.exists():
            entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        else:
            entries = [token.strip() for token in raw_ids.split(",") if token.strip()]
        parsed = [_parse_line(entry) for entry in entries]
        return parsed

    if id2label:
        numeric_keys = []
        for key in id2label.keys():
            try:
                numeric_keys.append(int(key))
            except ValueError:
                continue
        sorted_ids = sorted(numeric_keys)
        selected = sorted_ids[:count]
        return [(idx, id2label.get(str(idx), f"class_{idx}")) for idx in selected]

    # fallback synthetic classes
    return [(i, f"class_{i}") for i in range(count)]


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:80] or "prompt"


@dataclass
class RunStats:
    name: str
    runtimes: List[float]
    peak_memory_gb: float | None

    @property
    def avg_time(self) -> float:
        return sum(self.runtimes) / max(len(self.runtimes), 1)

    @property
    def latencies(self) -> Dict[str, float]:
        if not self.runtimes:
            return {"avg": 0.0, "std": 0.0}
        avg = self.avg_time
        var = sum((t - avg) ** 2 for t in self.runtimes) / len(self.runtimes)
        return {"avg": avg, "std": math.sqrt(var)}


def build_dqar_config(args: argparse.Namespace) -> DQARConfig:
    cfg = DQARConfig()
    cfg.gate.min_probability = args.dqar_min_prob
    cfg.gate.entropy_threshold = args.dqar_entropy
    cfg.gate.snr_range = (args.dqar_snr_min, args.dqar_snr_max)
    cfg.gate.cooldown_steps = args.dqar_cooldown
    cfg.scheduler.max_gap = args.dqar_max_gap
    cfg.scheduler.max_reuse_per_block = args.dqar_max_reuse
    return cfg


def summarize(name: str, stats: RunStats) -> Dict[str, float | str]:
    summary = {
        "name": name,
        "avg_time_s": stats.latencies["avg"],
        "std_time_s": stats.latencies["std"],
        "num_samples": len(stats.runtimes),
    }
    if stats.peak_memory_gb is not None:
        summary["peak_memory_gb"] = stats.peak_memory_gb
    return summary


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_pass(
    pipe: DiTPipeline,
    condition_items: Sequence,
    args: argparse.Namespace,
    *,
    use_dqar: bool,
) -> RunStats:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtimes: List[float] = []
    peak_memory = None
    dqar_controller = None

    if use_dqar:
        if not hasattr(pipe, "transformer"):
            raise ValueError("Pipeline does not expose a `.transformer` module for DQAR.")
        num_layers = len(getattr(pipe.transformer, "blocks", []))
        if num_layers == 0:
            raise ValueError("Unable to infer transformer block count; patch integration first.")
        dqar_controller = DQARController(num_layers=num_layers, config=build_dqar_config(args))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    out_dir = args.output_dir / ("dqar" if use_dqar else "baseline")
    if not args.skip_images:
        ensure_dir(out_dir)

    gen = torch.Generator(device=device).manual_seed(args.seed)

    for idx, item in enumerate(condition_items):
        kwargs = dict(
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            num_images_per_prompt=args.images_per_prompt,
            generator=gen.manual_seed(args.seed + idx),
            output_type="pil",
        )
        label_for_name = ""
        if args.conditioning == "text":
            kwargs["prompt"] = item
            label_for_name = item
        else:
            class_id, class_name = item
            kwargs["class_labels"] = torch.tensor([class_id], device=device)
            label_for_name = class_name
        if use_dqar:
            kwargs["controller"] = dqar_controller

        torch.cuda.synchronize(device) if device.type == "cuda" else None
        start = time.perf_counter()
        result = pipe(**kwargs)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        elapsed = time.perf_counter() - start
        runtimes.append(elapsed)

        if not args.skip_images:
            for img_idx, image in enumerate(result.images):
                filename = f"{idx:02d}-{img_idx:02d}-{slugify(label_for_name)}.png"
                image.save(out_dir / filename)

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)

    return RunStats(
        name="DQAR" if use_dqar else "Baseline",
        runtimes=runtimes,
        peak_memory_gb=peak_memory,
    )


def plot_results(results: Sequence[RunStats], path: Path) -> None:
    labels = [r.name for r in results]
    avg_times = [r.latencies["avg"] for r in results]
    mems = [r.peak_memory_gb or 0.0 for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].bar(labels, avg_times, color=["#999999", "#1f77b4"])
    axes[0].set_title("Average Runtime (s)")
    axes[0].set_ylabel("Seconds")

    axes[1].bar(labels, mems, color=["#999999", "#1f77b4"])
    axes[1].set_title("Peak VRAM (GB)")
    axes[1].set_ylabel("Gigabytes")

    fig.tight_layout()
    fig.savefig(path)
    print(f"[+] Saved plot to {path}")


def main() -> None:
    args = parse_args()
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    pipe = DiTPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        variant=args.variant if args.variant != "none" else None,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if args.conditioning == "text":
        condition_items = resolve_text_prompts(args)
    else:
        id2label = getattr(pipe.config, "id2label", None)
        condition_items = parse_class_ids(args.class_ids, args.num_prompts, id2label)

    ensure_dir(args.output_dir)

    baseline = run_pass(pipe, condition_items, args, use_dqar=False)
    dqar = run_pass(pipe, condition_items, args, use_dqar=True)

    summaries = [summarize("baseline", baseline), summarize("dqar", dqar)]
    args.save_json.write_text(json.dumps(summaries, indent=2))
    print(f"[+] Wrote summary to {args.save_json}")

    plot_results([baseline, dqar], args.plot_path)


if __name__ == "__main__":
    main()
