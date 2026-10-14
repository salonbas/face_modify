"""像素域擾動度量（與模型無關）。"""
from __future__ import annotations

import numpy as np

from advface.config import LINF_TOLERANCE, PIXEL_MAX


def perturbation_metrics(original_bgr: np.ndarray, adversarial_bgr: np.ndarray) -> dict[str, float]:
    a = original_bgr.astype(np.float64)
    b = adversarial_bgr.astype(np.float64)
    diff = b - a
    linf = float(np.max(np.abs(diff)) / PIXEL_MAX)
    l2_pixel = float(np.sqrt(np.sum(diff * diff)))
    return {"linf": linf, "l2_pixel": l2_pixel}


def validate_linf(linf: float, eps: float, *, tolerance: float = LINF_TOLERANCE) -> bool:
    if linf != linf:  # NaN
        return False
    return float(linf) <= float(eps) + float(tolerance)
