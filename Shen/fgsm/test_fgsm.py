import importlib.util
import pathlib
import unittest

import numpy as np


MODULE_PATH = pathlib.Path(__file__).with_name("insightface_fgsm_sensitivity_cpu.py")


def load_module():
    spec = importlib.util.spec_from_file_location("fgsm_tool", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FgsmTests(unittest.TestCase):
    def test_zero_eps_returns_copy(self):
        module = load_module()
        image = np.full((8, 8, 3), 100, dtype=np.uint8)

        result = module.fgsm_attack(
            image,
            eps=0.0,
            surrogate=None,
            weights=None,
        )

        np.testing.assert_array_equal(result, image)
        self.assertIsNot(result, image)

    def test_negative_eps_is_rejected(self):
        module = load_module()
        image = np.zeros((4, 4, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            module.fgsm_attack(image, eps=-0.01, surrogate=None, weights=None)

    def test_fgsm_changes_pixels_and_keeps_dtype(self):
        module = load_module()
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        surrogate = module.build_simple_cnn(num_classes=10)
        adv = module.fgsm_attack(
            image,
            eps=0.05,
            surrogate=surrogate,
            weights=None,
            target_size=(32, 32),
        )

        self.assertEqual(adv.shape, image.shape)
        self.assertEqual(adv.dtype, np.uint8)
        self.assertGreaterEqual(adv.min(), 0)
        self.assertLessEqual(adv.max(), 255)
        self.assertFalse(np.array_equal(adv, image))

        # FGSM 是 sign 攻擊，每個 channel 的像素差最多 ceil(eps*255)。
        max_diff = int(np.abs(adv.astype(np.int32) - image.astype(np.int32)).max())
        self.assertLessEqual(max_diff, int(np.ceil(0.05 * 255)) + 1)

    def test_resolve_eps_levels_uses_g_format(self):
        module = load_module()

        class DummyArgs:
            eps = [0.01, 0.1]

        levels = module.resolve_eps_levels(DummyArgs())
        self.assertEqual(levels, [("eps_0.01", 0.01), ("eps_0.1", 0.1)])

    def test_output_path_replaces_dot_in_label(self):
        module = load_module()

        path = module.output_path_for(
            "photos/musk2.jpg", "results/fgsm", "eps_0.05"
        )
        self.assertTrue(path.endswith("musk2_fgsm_eps_0p05.jpg"))


if __name__ == "__main__":
    unittest.main()
