import importlib.util
import pathlib
import unittest

import numpy as np


MODULE_PATH = pathlib.Path(__file__).with_name("insightface_noise_sensitivity_cpu.py")


def load_module():
    spec = importlib.util.spec_from_file_location("gaussian_noise_tool", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GaussianNoiseTests(unittest.TestCase):
    def test_add_gaussian_noise_is_reproducible_and_clipped(self):
        module = load_module()
        image = np.full((4, 4, 3), 250, dtype=np.uint8)

        first = module.add_gaussian_noise(image, sigma=20, seed=123)
        second = module.add_gaussian_noise(image, sigma=20, seed=123)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.uint8)
        self.assertGreaterEqual(first.min(), 0)
        self.assertLessEqual(first.max(), 255)
        self.assertFalse(np.array_equal(first, image))

    def test_noise_percent_converts_to_sigma(self):
        module = load_module()

        self.assertEqual(module.percent_to_sigma(30), 76.5)


if __name__ == "__main__":
    unittest.main()
