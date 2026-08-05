"""Optional numerical check: OpenCV warpAffine vs PyTorch grid_sample via _cv2M_to_torch_theta."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from advface.attacks.pgd import _cv2M_to_torch_theta


def test_cv2M_to_torch_theta_aligns_with_opencv_warp():
    """
    Synthetic image + simple translation/scale affine.
    Compare OpenCV warpAffine output with torch grid_sample.
    If this fails, do NOT silently change the formula — investigate first.
    """
    rng = np.random.default_rng(42)
    src_h, src_w = 64, 80
    dst = 32
    img = rng.integers(0, 255, size=(src_h, src_w, 3), dtype=np.uint8).astype(np.float32)

    # Simple center-crop-ish affine (scale + translate) in OpenCV src→dst form
    scale = dst / min(src_h, src_w)
    M = np.array(
        [
            [scale, 0.0, (dst - scale * src_w) / 2.0],
            [0.0, scale, (dst - scale * src_h) / 2.0],
        ],
        dtype=np.float64,
    )

    cv_crop = cv2.warpAffine(
        img, M.astype(np.float32), (dst, dst), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    theta = _cv2M_to_torch_theta(M, src_h, src_w, dst_H=dst, dst_W=dst)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    grid = F.affine_grid(torch.from_numpy(theta), (1, 3, dst, dst), align_corners=True)
    torch_crop = F.grid_sample(x, grid, mode="bilinear", align_corners=True, padding_mode="border")
    torch_crop_np = torch_crop.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    mae = float(np.mean(np.abs(cv_crop - torch_crop_np)))
    max_err = float(np.max(np.abs(cv_crop - torch_crop_np)))
    # bilinear + border handling can differ slightly; keep a generous but meaningful tolerance
    assert mae < 2.0, f"MAE too large: mae={mae}, max_err={max_err}"
    assert max_err < 25.0, f"max error too large: mae={mae}, max_err={max_err}"
