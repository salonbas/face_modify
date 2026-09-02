"""Full-image MI-FGSM following Dong et al., CVPR 2018.

The paper's L1-normalized gradient and momentum accumulation are used. Since
this platform minimizes surrogate self-match cosine, the update is negative.
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from advface.attacks.common import clip_pixel_range, project_delta_linf
from advface.attacks.pgd import _cv2M_to_torch_theta
from advface.config import ALIGNED_FACE_SIZE, ARCFACE_INPUT_MEAN, ARCFACE_INPUT_SCALE, PIXEL_MAX
from advface.models.arcface_torch import load_arcface_torch
from advface.models.insightface_app import pick_best_face


def normalize_gradient_l1(gradient, *, min_norm: float = 1e-12):
    """Normalize each image gradient by its L1 norm (paper algorithm)."""
    norm = gradient.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(min_norm)
    return gradient / norm


def momentum_update(momentum, gradient, decay: float):
    """MI-FGSM recurrence, kept small and deterministic for unit testing."""
    return decay * momentum + normalize_gradient_l1(gradient)


def run_mi_fgsm(
    img_bgr: np.ndarray,
    eps: float,
    app: Any,
    *,
    steps: int,
    alpha: Optional[float] = None,
    momentum: float = 1.0,
    device: Optional[str] = None,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | str]]:
    """Generate an untargeted full-image L∞-bounded MI-FGSM example.

    ``eps`` and ``alpha`` use normalized image units ([0, 1]), matching the
    existing attack API; projection itself is performed in pixel units.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise RuntimeError("需要 PyTorch。請安裝 torch。") from exc
    if steps <= 0:
        raise ValueError("MI-FGSM steps 必須大於 0")
    if eps < 0 or momentum < 0:
        raise ValueError("MI-FGSM eps / momentum 不可為負數")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    face = pick_best_face(app.get(img_bgr), "input")
    from insightface.utils import face_align
    _, matrix = face_align.norm_crop2(img_bgr, landmark=face.kps, image_size=ALIGNED_FACE_SIZE)
    height, width = img_bgr.shape[:2]
    x0 = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).float()
    x0 = x0.permute(2, 0, 1).unsqueeze(0).to(device)
    theta = torch.from_numpy(_cv2M_to_torch_theta(matrix, height, width)).to(device)
    grid = F.affine_grid(theta, (1, 3, ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE), align_corners=True)
    model = load_arcface_torch(device)

    def embedding(x):
        crop = F.grid_sample(x, grid, mode="bilinear", align_corners=True, padding_mode="border")
        return F.normalize(model((crop - ARCFACE_INPUT_MEAN) / ARCFACE_INPUT_SCALE), dim=1)

    with torch.no_grad():
        reference = embedding(x0).detach()
    eps_px = float(eps) * PIXEL_MAX
    alpha_px = float(alpha if alpha is not None else eps / steps) * PIXEL_MAX
    delta = torch.zeros_like(x0)
    accumulated = torch.zeros_like(x0)
    for _ in range(steps):
        delta.requires_grad_(True)
        cosine = (embedding(clip_pixel_range(x0 + delta)) * reference).sum()
        gradient = torch.autograd.grad(cosine, delta)[0]
        with torch.no_grad():
            accumulated = momentum_update(accumulated, gradient, float(momentum))
            delta = project_delta_linf(delta.detach() - alpha_px * accumulated.sign(), eps_px)
    with torch.no_grad():
        adversarial = clip_pixel_range(x0 + delta)
        linf_tensor = float((adversarial - x0).abs().max().item() / PIXEL_MAX)
    adversarial_u8_rgb = adversarial.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
    adversarial_bgr = cv2.cvtColor(
        adversarial_u8_rgb,
        cv2.COLOR_RGB2BGR,
    )
    linf_serialized = float(np.max(np.abs(adversarial_bgr.astype(np.int16) - img_bgr.astype(np.int16))) / PIXEL_MAX)
    if not return_metadata:
        return adversarial_bgr
    return adversarial_bgr, {
        "linf_tensor": linf_tensor,
        "linf_serialized": linf_serialized,
        "linf_serialization_note": "serialized uint8 value may differ due to 8-bit quantization",
    }
