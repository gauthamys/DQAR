import unittest

from dqar import compute_attention_entropy, compute_snr


class StatsTest(unittest.TestCase):
    def test_entropy(self):
        attn = [[[0.5, 0.5], [0.25, 0.75]]]
        value = compute_attention_entropy(attn)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)

    def test_snr(self):
        clean = [0.0, 1.0, -1.0]
        noisy = [0.1, 0.8, -1.1]
        value = compute_snr(clean, noisy)
        self.assertGreater(value, 0.0)


if __name__ == "__main__":
    unittest.main()
