import importlib.util
import pathlib
import unittest

import numpy as np


MODULE_PATH = pathlib.Path(__file__).with_name("insightface_blur_sensitivity_cpu.py")


def load_module():
    spec = importlib.util.spec_from_file_location("gaussian_blur_tool", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GaussianBlurTests(unittest.TestCase):
    def test_apply_gaussian_blur_changes_pixels_and_keeps_dtype(self):
        module = load_module()
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[8, 8] = [255, 255, 255]

        blurred = module.apply_gaussian_blur(image, sigma=2.0)

        self.assertEqual(blurred.shape, image.shape)
        self.assertEqual(blurred.dtype, np.uint8)
        self.assertGreaterEqual(blurred.min(), 0)
        self.assertLessEqual(blurred.max(), 255)
        self.assertFalse(np.array_equal(blurred, image))
        # 中心點亮度因為被平均擴散，會比原本 255 還低。
        self.assertLess(int(blurred[8, 8, 0]), 255)
        # 周圍原本是 0 的像素應該被擴散成正值。
        self.assertGreater(int(blurred[7, 8, 0]), 0)

    def test_zero_sigma_returns_copy_not_same_object(self):
        module = load_module()
        image = np.full((4, 4, 3), 100, dtype=np.uint8)

        result = module.apply_gaussian_blur(image, sigma=0.0)

        np.testing.assert_array_equal(result, image)
        self.assertIsNot(result, image)

    def test_sigma_to_kernel_size_is_odd_and_scales_with_sigma(self):
        module = load_module()

        self.assertEqual(module.sigma_to_kernel_size(1), 7)
        self.assertEqual(module.sigma_to_kernel_size(3), 19)
        self.assertEqual(module.sigma_to_kernel_size(8), 49)
        for sigma in [0.5, 1, 2, 5, 10]:
            self.assertEqual(module.sigma_to_kernel_size(sigma) % 2, 1)

    def test_explicit_even_kernel_is_rejected(self):
        module = load_module()
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            module.apply_gaussian_blur(image, sigma=2.0, kernel_size=4)


if __name__ == "__main__":
    unittest.main()
