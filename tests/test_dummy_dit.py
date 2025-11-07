import unittest

from dqar import DQARController, DQARConfig
from dqar.dummy_dit import DummyDiffusionTransformer


class DummyDiTTest(unittest.TestCase):
    def test_generate_runs(self):
        config = DQARConfig()
        config.gate.min_step = 0
        controller = DQARController(num_layers=4, config=config)
        model = DummyDiffusionTransformer(num_layers=4)
        output = model.generate(controller=controller, steps=4, prompt_length=8, seed=1)
        self.assertEqual(len(output.latents), model.latent_dim)
        self.assertGreaterEqual(output.reuse_events, 0)
        self.assertGreater(output.recompute_events, 0)


if __name__ == "__main__":
    unittest.main()
