"""tests for L∞ projection helpers."""
from __future__ import annotations

import numpy as np

from advface.attacks.common import project_delta_linf, project_linf_around_x0


def test_perturbation_within_epsilon():
    rng = np.random.default_rng(0)
    x0 = rng.uniform(0, 255, size=(4, 4, 3)).astype(np.float32)
    noise = rng.uniform(-50, 50, size=x0.shape).astype(np.float32)
    eps_px = 10.0
    x_adv = project_linf_around_x0(x0 + noise, x0, eps_px)
    assert np.max(np.abs(x_adv - x0)) <= eps_px + 1e-5


def test_pixel_range_clipped():
    x0 = np.full((2, 2, 3), 10.0, dtype=np.float32)
    x_adv = np.full_like(x0, 500.0)
    out = project_linf_around_x0(x_adv, x0, eps_px=100.0)
    assert out.min() >= 0.0
    assert out.max() <= 255.0


def test_projection_centered_on_original():
    x0 = np.array([[[100.0, 100.0, 100.0]]], dtype=np.float32)
    x_adv = np.array([[[200.0, 0.0, 150.0]]], dtype=np.float32)
    eps_px = 20.0
    out = project_linf_around_x0(x_adv, x0, eps_px)
    np.testing.assert_allclose(out, np.array([[[120.0, 80.0, 120.0]]], dtype=np.float32))


def test_delta_linf_projection():
    delta = np.array([30.0, -40.0, 5.0], dtype=np.float32)
    out = project_delta_linf(delta, eps_px=10.0)
    np.testing.assert_allclose(out, np.array([10.0, -10.0, 5.0], dtype=np.float32))
