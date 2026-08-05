"""攻擊共用工具：解析 eps、貼回對齊臉、噪聲圖、L∞ 投影。不改動數學語意。"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from advface.config import ALIGNED_FACE_SIZE, PIXEL_MAX, PIXEL_MIN


def parse_eps_list(eps_list_str: str) -> List[float]:
    return [float(x.strip()) for x in eps_list_str.split(",") if x.strip()]


def _bgr_to_rgb_float(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _save_noise_map(diff_rgb: np.ndarray, out_path: Path, base_bgr: np.ndarray) -> None:
    heat = np.abs(diff_rgb).max(axis=2)
    heat_u8 = (heat / (heat.max() + 1e-12) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_bgr, 0.6, heat_color, 0.4, 0)
    cv2.imwrite(str(out_path), overlay)


def _paste_aligned_patch(img_bgr: np.ndarray, adv_crop_bgr: np.ndarray, M: np.ndarray) -> np.ndarray:
    """把 112×112 對齊臉貼回原圖座標（仿射逆變換 + mask 混合）。"""
    h, w = img_bgr.shape[:2]
    size = ALIGNED_FACE_SIZE
    Minv = cv2.invertAffineTransform(M.astype(np.float32))
    warped = cv2.warpAffine(adv_crop_bgr, Minv, (w, h))
    mask = cv2.warpAffine(np.ones((size, size), np.float32), Minv, (w, h))
    m3 = np.stack([np.clip(mask, 0, 1)] * 3, axis=2)
    out = img_bgr.astype(np.float32) * (1 - m3) + warped.astype(np.float32) * m3
    return np.clip(out, PIXEL_MIN, PIXEL_MAX).astype(np.uint8)


def project_linf_around_x0(x_adv, x0, eps_px: float, pixel_min: float = PIXEL_MIN, pixel_max: float = PIXEL_MAX):
    """
    裁切模式 L∞ 投影：x_adv ∈ [x0 ± eps_px] ∩ [pixel_min, pixel_max]。
    支援 torch.Tensor 與 numpy.ndarray。
    """
    try:
        import torch

        if isinstance(x_adv, torch.Tensor):
            x_adv = torch.max(x0 - eps_px, torch.min(x0 + eps_px, x_adv))
            return x_adv.clamp(pixel_min, pixel_max)
    except Exception:
        pass

    x_adv = np.minimum(np.maximum(x_adv, x0 - eps_px), x0 + eps_px)
    return np.clip(x_adv, pixel_min, pixel_max)


def project_delta_linf(delta, eps_px: float):
    """全圖模式：將 delta 投影到 [-eps_px, eps_px]。"""
    try:
        import torch

        if isinstance(delta, torch.Tensor):
            return delta.clamp(-eps_px, eps_px)
    except Exception:
        pass
    return np.clip(delta, -eps_px, eps_px)


def clip_pixel_range(x, pixel_min: float = PIXEL_MIN, pixel_max: float = PIXEL_MAX):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.clamp(pixel_min, pixel_max)
    except Exception:
        pass
    return np.clip(x, pixel_min, pixel_max)
