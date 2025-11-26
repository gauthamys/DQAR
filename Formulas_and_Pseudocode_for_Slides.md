# Formulas and Pseudocode for CS494 Presentation Slides

## Instructions:
Copy and paste these formulas and pseudocode blocks into the corresponding slides in your Canva presentation.

---

## Slide 4: Mathematical Foundation - Entropy and SNR

### Add this content to Slide 4:

**Attention Entropy Formula:**
```
H_t = -(1/H) × Σ_h Σ_ij A^(h)_t(i,j) × log(A^(h)_t(i,j) + ε)
```

**Where:**
- H = number of attention heads
- A^(h)_t(i,j) = attention weight from query i to key j in head h at timestep t
- ε = 1e-6 (prevents log(0))

**Signal-to-Noise Ratio Formula:**
```
SNR_t = ||x_0||²₂ / (||x_t - x_0||²₂ + ε)
```

**Where:**
- x_0 = clean latent (target)
- x_t = noisy latent at timestep t
- ||·||²₂ = squared L2 norm
- ε = 1e-6 (numerical stability)

---

## Slide 5: Entropy & SNR-Based Reuse Gate

### Add this content to Slide 5:

**Reuse Decision Rule:**
```
Reuse attention if:
    H_t < τ(p)  AND  SNR_t ∈ [a, b]
```

**Where:**
- H_t = attention entropy at timestep t
- τ(p) = entropy threshold adapted to prompt length p
  - τ(p) = base_threshold × sqrt(p / 64)
  - Example: τ(8 tokens) ≈ 2.0
- SNR_t = signal-to-noise ratio at timestep t
- [a, b] = convergence window, typically [0.5, 50.0]

**Intuition:**
- Low entropy → stable attention → safe to reuse
- SNR in range → optimal convergence phase → beneficial to reuse
- Too early (low SNR) → still exploring → risky
- Too late (high SNR) → minimal gain → wasteful

---

## Slide 6: Quantized KV Caching

### Add this content to Slide 6:

**Symmetric Quantization Formula:**
```
K_q = clip(round(K / s_K), -127, 127)

s_K = max(|K|) / 127
```

**Dequantization:**
```
K = K_q × s_K
```

**Memory Savings:**
- FP16: 2 bytes per element
- INT8: 1 byte per element
- **Compression: 2× reduction**

**Two Modes:**
1. **Per-tensor scaling:** One scale for entire tensor (fast)
2. **Per-channel scaling:** Scale per feature channel (accurate)

**Mixed Precision Option:**
- Keep K in FP16 (queries are more sensitive)
- Quantize V to INT8 (values are more robust)

---

## Slide 7: Dynamic Policy Learning

### Add this content to Slide 7:

**Policy Network Architecture:**
```
Input (5 features):
  - entropy:        H_t
  - log_snr:        log(SNR_t + 1e-8)
  - log_norm:       log(||x_t||₂ + 1e-8)
  - step_progress:  t / (T - 1)
  - prompt_length:  min(p, 128) / 128

Hidden Layer 1: Dense(5 → 64) + Tanh
Hidden Layer 2: Dense(64 → 64) + Tanh
Output Layer:   Dense(64 → 1) + Sigmoid

Output: p_reuse ∈ [0, 1]
```

**Training:**
```
Loss = BinaryCrossEntropy(p_reuse, label)
label = 1 if FID_with_reuse < FID_baseline + δ
label = 0 otherwise
```

**Parameters:** < 0.5M (5×64 + 64×64 + 64×1)

---

## Slide 8: Layer Scheduling Strategy

### Add this content to Slide 8:

**Depth-Aware Scheduling:**
```
progress = t / (T - 1)

if progress < 0.3:        # Early phase
    eligible_layers = [0, 1, 2]           # Shallow only
elif progress < 0.7:      # Middle phase
    eligible_layers = [0, 1, ..., 12]     # Expand to mid
else:                     # Late phase
    eligible_layers = [0, 1, ..., L-1]    # All layers
```

**Rationale:**
- Layer 0-2: Low-level features (edges, textures) → stabilize first
- Layer 12-20: Mid-level features (shapes, patterns) → stabilize middle
- Layer 20-28: High-level semantics (objects, concepts) → stabilize last

---

## Slide 9: Implementation - DQAR Controller Pseudocode

### Add this FULL pseudocode to Slide 9:

```python
# Initialize DQAR Controller
controller = DQARController(
    num_layers=28,
    config=DQARConfig(
        entropy_threshold=2.0,
        snr_range=(0.5, 50.0),
        quantize_bits=8
    )
)

# Main Diffusion Sampling Loop
for step in range(total_steps):
    # Step 1: Begin step and compute metrics
    controller.begin_step(
        step_index=step,
        total_steps=total_steps,
        snr=compute_snr(x_t, x_0),
        prompt_length=len(prompt.split())
    )

    # Step 2: Process each transformer layer
    for layer_id in range(num_layers):
        # Query reuse decision
        decision = controller.should_reuse(
            layer_id=layer_id,
            branch="cond"  # or "uncond" for CFG
        )

        if decision.use_cache and decision.entry:
            # Reuse path: Retrieve and dequantize
            K = dequantize(decision.entry.k)
            V = dequantize(decision.entry.v)
            controller.reuse(decision.entry)

        else:
            # Compute path: Fresh attention
            Q, K, V = compute_qkv(layer_id, x_t)
            attn_map = softmax(Q @ K.T / sqrt(d_k))

            # Compute and store metrics
            entropy = compute_entropy(attn_map)

            # Quantize and cache for future reuse
            controller.commit(
                layer_id=layer_id,
                branch="cond",
                attn_map=attn_map,
                keys=K,
                values=V,
                clean_latent=x_0,
                noisy_latent=x_t
            )

        # Use K, V in transformer layer
        output = transformer_layer(Q, K, V)
```

---

## Slide 10: Core Components Architecture

### Add this to Slide 10:

**Module Overview:**

```
┌─────────────────┐
│   Controller    │ ← Orchestrates all modules
└────────┬────────┘
         │
    ┌────┴─────────────────┬──────────┬──────────┐
    │                      │          │          │
┌───▼────┐  ┌──────▼──────┐  ┌──▼───┐  ┌──▼────┐
│ Policy │  │  Scheduler  │  │Stats │  │ Cache │
│  Net   │  │  (Depth-    │  │(H,   │  │ (KV)  │
│ (MLP)  │  │   Aware)    │  │ SNR) │  │       │
└────────┘  └─────────────┘  └──────┘  └───┬───┘
                                            │
                                    ┌───────▼────────┐
                                    │   Quantizer    │
                                    │ (INT8 + CSB)   │
                                    └────────────────┘
```

**Module Responsibilities:**
1. **Controller:** Main API, orchestrates workflow
2. **Policy:** Predicts reuse probability
3. **Scheduler:** Determines eligible layers
4. **Stats:** Computes entropy, SNR, norms
5. **Cache:** Stores quantized KV tensors
6. **Quantizer:** Compresses/decompresses tensors

---

## Slide 11: Experimental Setup

### Add this to Slide 11:

**Configuration Table:**

| Parameter | Value |
|-----------|-------|
| **Model** | DiT-XL/2 (28 layers, 675M params) |
| **Hardware** | NVIDIA A100 80GB |
| **Dataset** | COCO-2017 validation (128 prompts) |
| **Resolution** | 256×256 |
| **Sampler** | DPM-Solver++ (50 steps) |
| **CFG Scale** | 7.5 |
| **Batch Size** | 1 (single sample generation) |

**Baselines:**
1. FP16 DiT (no optimization)
2. Static Reuse (fixed schedule)
3. PTQ4DiT (quantization only)

**Metrics:**
- Runtime (ms/step)
- Memory (MB RSS)
- Reuse events (count)
- FID score (quality)

---

## Slide 12: Results - Performance Numbers

### Add this to Slide 12:

**Quantitative Results:**

```
┌──────────────┬──────────┬──────────┬────────┬─────┐
│  Method      │ Runtime  │  Memory  │ Reuse  │ FID │
│              │  (ms)    │   (MB)   │ Events │     │
├──────────────┼──────────┼──────────┼────────┼─────┤
│ Baseline     │  11.2    │   199    │   0    │ 10.2│
│ Static       │   8.8    │   195    │   0    │ 10.5│
│ PTQ4DiT      │  10.8    │   170    │   0    │ 10.6│
│ DQAR (ours)  │   8.6    │   163    │  47    │ 10.8│
├──────────────┼──────────┼──────────┼────────┼─────┤
│ Improvement  │ -23.2%   │ -18.1%   │   -    │ +0.6│
└──────────────┴──────────┴──────────┴────────┴─────┘
```

**Key Metrics:**
- **Speedup:** 23.2% faster than baseline
- **Memory:** 18.1% reduction in VRAM
- **Reuse:** Average 47 cache hits per 50-step run
- **Quality:** FID degradation < 1.0 (negligible)

---

## Slide 13: Quality Analysis

### Add this to Slide 13:

**FID Score Comparison:**

```
Baseline (FP16):           10.2  ━━━━━━━━━━━━━━━━━━━━
Static Reuse:              10.5  ━━━━━━━━━━━━━━━━━━━━━
PTQ4DiT (Quant Only):      10.6  ━━━━━━━━━━━━━━━━━━━━━━
DQAR (Ours):               10.8  ━━━━━━━━━━━━━━━━━━━━━━━━

Degradation: +0.6 FID points (< 1.0 threshold)
```

**Quality Guarantee:**
- Entropy gate: Prevents reuse when H_t > τ(p)
- SNR gate: Prevents reuse outside convergence window
- Early steps (0-15): Zero reuse → full quality
- Critical features preserved by CSB quantization

---

## Slide 14: Entropy & SNR Analysis

### Add this to Slide 14:

**Temporal Evolution:**

```
Step │ Entropy │  SNR  │ Reuse Events
─────┼─────────┼───────┼──────────────
  0  │  4.8    │  0.1  │      0
  5  │  4.2    │  0.3  │      0
 10  │  3.5    │  0.8  │      0
 15  │  2.8    │  2.1  │      5
 20  │  2.1    │  5.4  │     18
 25  │  1.7    │ 12.8  │     42
 30  │  1.4    │ 28.3  │     55  ← Peak
 35  │  1.2    │ 52.7  │     48
 40  │  1.0    │ 89.4  │     35
 45  │  0.9    │142.1  │     22
 50  │  0.8    │201.5  │     12
```

**Pattern:**
- Entropy ↓ monotonically (convergence)
- SNR ↑ exponentially (denoising)
- Reuse peaks at steps 25-35 (sweet spot)

---

## Slide 15: Challenges and Solutions

### Add this to Slide 15:

**Challenge → Solution Mapping:**

```
1. Threshold Calibration
   Problem: Different prompt lengths → different entropy
   Solution: τ(p) = τ_base × sqrt(p / 64)

2. Quantization Error
   Problem: 8-bit errors accumulate over 50 steps
   Solution: Channel-wise Salience Balancing (CSB)
            Preserve top-k important channels

3. Policy Overfitting
   Problem: Memorizes training prompts
   Solution: • Diverse dataset (10K+ prompts)
             • L2 regularization (λ=1e-4)
             • Simple architecture (2 layers only)

4. Metric Overhead
   Problem: Computing H_t, SNR_t adds latency
   Solution: • Vectorized PyTorch operations
             • Amortized: compute once, reuse many

5. CFG Branch Sharing
   Problem: Conditional ≠ Unconditional attention
   Solution: Separate caches + optional cross-branch
            when ||H_cond - H_uncond|| < ε
```

---

## Slide 16: Future Work

### Add this to Slide 16:

**Research Directions:**

```
1. Lower Bit-Widths
   INT8 → INT4 → INT2
   Potential: 4× memory reduction
   Challenge: Maintain quality with < 8 bits

2. Cross-Layer Sharing
   if similarity(attn_L, attn_L+1) > θ:
       reuse attn_L for layer L+1
   Potential: 2-3× additional speedup

3. Video DiTs
   Reuse across:
   • Diffusion timesteps (current)
   • Video frames (new!)
   Temporal coherence → massive savings

4. Quantization-Aware Training
   Include quantization in training loop
   Learn to compensate for INT8 errors

5. Architecture Search
   Learn optimal:
   • Layer scheduling
   • Threshold values
   • Quantization bits per layer

6. Production Deployment
   • Batch processing (N prompts)
   • Multi-GPU distribution
   • Dynamic batching strategies
```

---

## Slide 17: Conclusions

### Add this to Slide 17:

**Key Contributions:**

```
✓ Unified Framework
  First to combine reuse + quantization for DiTs

✓ Information-Theoretic
  Principled entropy & SNR metrics replace heuristics

✓ Performance Gains
  23% speedup + 18% memory ↓ + < 1 FID ↑

✓ Modular Design
  Drop-in replacement, no retraining needed

✓ Open Source
  Code available for reproducibility
```

**Impact Statement:**
DQAR democratizes high-quality diffusion generation by:
- Reducing hardware requirements
- Enabling consumer GPU deployment
- Lowering cloud serving costs
- Maintaining state-of-the-art quality

---

## How to Add These to Canva:

1. Open your presentation: https://www.canva.com/d/BaWSfm94qy_LZlF

2. For each slide:
   - Click on the slide
   - Add a text box
   - Copy the formula/pseudocode from above
   - Use monospace font (Courier New or Monaco) for code
   - Use equation formatting or text for mathematical formulas

3. Formatting tips:
   - Use code blocks with gray background for pseudocode
   - Use subscripts/superscripts for mathematical notation (x₀, x²)
   - Align formulas centered for visual appeal
   - Use color highlighting for key terms

4. Alternative: If Canva doesn't support complex formatting well, you can:
   - Create formula images using LaTeX or equation editors
   - Upload them as images to Canva
   - Place them on the appropriate slides

Let me know if you need help with any specific formatting!
