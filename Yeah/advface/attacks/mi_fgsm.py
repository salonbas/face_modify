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
    constraint: Any | None = None,
    tv_weight: float = 0.0,
    diagnostics: bool = False,
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
    if eps < 0 or momentum < 0 or tv_weight < 0:
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
    mask_t = None
    final_mask = None
    if constraint is not None:
        mask, _, _ = constraint.build(img_bgr)
        final_mask = mask.copy()
        mask_t = torch.from_numpy(mask).to(device=device, dtype=x0.dtype)[None, None]

    def total_variation(value):
        return (value[:, :, 1:, :] - value[:, :, :-1, :]).abs().mean() + (value[:, :, :, 1:] - value[:, :, :, :-1]).abs().mean()

    trace: list[dict[str, float | int | bool | None]] = []
    previous_raw = previous_momentum = previous_update = None

    def _cos(a, b):
        denom = a.norm() * b.norm()
        return None if float(denom.item()) == 0.0 else float((a * b).sum().div(denom).item())

    for step in range(steps):
        delta.requires_grad_(True)
        cosine = (embedding(clip_pixel_range(x0 + delta)) * reference).sum()
        objective = cosine + float(tv_weight) * total_variation(delta)
        gradient = torch.autograd.grad(objective, delta)[0]
        with torch.no_grad():
            raw_gradient = gradient
            record = None
            if diagnostics:
                raw_abs = raw_gradient.abs()
                record = {"step": step + 1, "raw_grad_l1": float(raw_abs.sum().item()), "raw_grad_l2": float(raw_gradient.norm().item()), "raw_grad_max_abs": float(raw_abs.max().item()), "raw_grad_mean_abs": float(raw_abs.mean().item()), "raw_grad_prev_cosine": _cos(raw_gradient, previous_raw) if previous_raw is not None else None}
                if mask_t is not None:
                    inside, outside = raw_gradient * mask_t, raw_gradient * (1.0 - mask_t)
                    inside_abs, outside_abs = inside.abs(), outside.abs()
                    elements_inside, elements_outside = mask_t.sum() * raw_gradient.shape[1], (1.0 - mask_t).sum() * raw_gradient.shape[1]
                    record.update({"inside_l1_ratio": float((inside_abs.sum() / raw_abs.sum().clamp_min(1e-12)).item()), "inside_l2_ratio": float((inside.norm() / raw_gradient.norm().clamp_min(1e-12)).item()), "inside_grad_abs_sum": float(inside_abs.sum().item()), "outside_grad_abs_sum": float(outside_abs.sum().item()), "mask_coverage": float(mask_t.mean().item()), "inside_grad_mean_abs": float((inside_abs.sum() / elements_inside.clamp_min(1)).item()), "outside_grad_mean_abs": float((outside_abs.sum() / elements_outside.clamp_min(1)).item())})
            if mask_t is not None:
                gradient = gradient * mask_t
            prior_momentum = accumulated
            accumulated = momentum_update(accumulated, gradient, float(momentum))
            update = -accumulated.sign()
            delta = project_delta_linf(delta.detach() + alpha_px * update, eps_px)
            if diagnostics:
                assert record is not None
                cosine_after = (embedding(clip_pixel_range(x0 + delta)) * reference).sum()
                delta_abs = delta.abs()
                record.update({"arcface_cosine": float(cosine_after.item()), "objective": float(objective.item()), "arcface_threshold_crossing": None, "current_linf": float(delta_abs.max().item() / PIXEL_MAX), "current_l1": float(delta_abs.sum().item()), "current_l2": float(delta.norm().item()), "changed_pixel_ratio": float(torch.any(delta != 0, dim=1).float().mean().item()), "epsilon_saturation_ratio": float((delta_abs >= eps_px - 1e-6).float().mean().item()), "momentum_l1": float(accumulated.abs().sum().item()), "momentum_l2": float(accumulated.norm().item()), "raw_prev_momentum_cosine": _cos(raw_gradient, prior_momentum), "gradient_updated_momentum_cosine": _cos(gradient, accumulated), "momentum_prev_cosine": _cos(accumulated, previous_momentum) if previous_momentum is not None else None, "update_raw_grad_cosine": _cos(update, raw_gradient), "update_prev_cosine": _cos(update, previous_update) if previous_update is not None else None})
                trace.append(record)
                previous_raw, previous_momentum, previous_update = raw_gradient.detach().clone(), accumulated.detach().clone(), update.detach().clone()
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
        **({"diagnostics": {"trajectory": trace, "operation_order": "raw gradient → optional mask → L1 normalize masked gradient → momentum accumulation → sign → update → L∞ projection → pixel clipping"}, "final_mask": final_mask} if diagnostics else {}),
    }
