import unittest

from dqar import QuantizationConfig
from dqar.quantization import Quantizer, SalienceCalibrator


class QuantizationTest(unittest.TestCase):
    def test_roundtrip(self):
        cfg = QuantizationConfig(num_bits=8, per_channel=False)
        quantizer = Quantizer(cfg)
        tensor = [[-0.5, 0.5], [0.25, -0.25]]
        qtensor = quantizer.quantize(tensor)
        restored = quantizer.dequantize(qtensor)
        self.assertEqual(len(restored), len(tensor))

    def test_calibrator(self):
        cfg = QuantizationConfig()
        calibrator = SalienceCalibrator(cfg)
        calibrator.observe("layer0", [[0.1, 0.5]], [[0.2, 0.4]])
        profile = calibrator.build()
        self.assertIn("layer0", profile)
        self.assertEqual(len(profile["layer0"].act_scale), 2)


if __name__ == "__main__":
    unittest.main()
