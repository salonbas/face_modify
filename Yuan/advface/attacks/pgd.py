"""
PGD（Projected Gradient Descent）— 多步迭代 L∞ 白盒攻擊

提供兩種模式：

  pgd       — 裁切 112×112 對齊臉 → 迭代擾動 → 貼回原圖
               適合快速實驗，但對抗圖邊界有矩形接縫（肉眼可見）

  pgd_full  — 擾動 delta 施加在整張圖，透過 PyTorch 可微分仿射裁切
               （affine_grid + grid_sample）讓梯度直接流回整張圖
               無接縫、無邊界，推薦用於最終攻擊

共同公式（每步）：
  delta = delta - step_size × sign( ∂cosine / ∂delta )
  delta = clamp(delta, -eps×255, +eps×255)    # 投影回 L∞ ball

新輸出命名見 advface.experiments.output。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np

from advface.attacks.common import (
    _paste_aligned_patch,
    clip_pixel_range,
    parse_eps_list,
    project_delta_linf,
    project_linf_around_x0,
)
from advface.config import (
    ALIGNED_FACE_SIZE,
    ARCFACE_INPUT_MEAN,
    ARCFACE_INPUT_SCALE,
    DEFAULT_ATTACK_SEED,
    PIXEL_MAX,
    PIXEL_MIN,
)
from advface.experiments.output import build_run_config, evaluate_and_save_attack_run
from advface.image_io import image_stem, load_bgr
from advface.models.arcface_torch import load_arcface_torch
from advface.models.insightface_app import create_face_app, pick_best_face


@dataclass
class PgdResult:
    eps: float
    eps_255: float
    steps: int
    cosine: float
    attacked_bgr: np.ndarray
    linf_tensor: float | None = None
    diagnostics: dict[str, Any] | None = None
    final_mask: np.ndarray | None = None


def run_pgd(
    img_bgr: np.ndarray,
    eps_values: List[float],
    app: Any,
    steps: int = 20,
    device: Optional[str] = None,
) -> List[PgdResult]:
    """
    對 112×112 對齊臉做多步迭代擾動，最後貼回原圖。
    注意：對抗圖邊界有矩形接縫（建議改用 run_pgd_full）。
    """
    try:
        import torch
        import torch.nn.functional as F
    except ModuleNotFoundError as e:
        raise RuntimeError("需要 PyTorch。請安裝 torch。") from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    face = pick_best_face(app.get(img_bgr), "input")

    from insightface.utils import face_align

    crop_bgr, M = face_align.norm_crop2(
        img_bgr, landmark=face.kps, image_size=ALIGNED_FACE_SIZE
    )
    x0 = torch.from_numpy(
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    ).float().permute(2, 0, 1).unsqueeze(0).to(device)

    model = load_arcface_torch(device)

    def _emb(x: "torch.Tensor") -> "torch.Tensor":
        return F.normalize(
            model((x - ARCFACE_INPUT_MEAN) / ARCFACE_INPUT_SCALE), dim=1
        )

    with torch.no_grad():
        emb_ref = _emb(x0).detach()

    results: List[PgdResult] = []
    for eps in eps_values:
        eps_px = eps * PIXEL_MAX
        step_size = eps_px / steps
        x_adv = x0.clone().detach()

        for _ in range(steps):
            x_adv.requires_grad_(True)
            (_emb(x_adv) * emb_ref).sum().backward()
            with torch.no_grad():
                x_adv = x_adv.detach() - step_size * x_adv.grad.detach().sign()
                x_adv = project_linf_around_x0(x_adv, x0, eps_px)

        adv_crop_bgr = cv2.cvtColor(
            x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        results.append(PgdResult(
            eps=eps, eps_255=eps_px, steps=steps, cosine=float("nan"),
            attacked_bgr=_paste_aligned_patch(img_bgr, adv_crop_bgr, M),
        ))
    return results


def _cv2M_to_torch_theta(
    M: np.ndarray, src_H: int, src_W: int, dst_H: int = 112, dst_W: int = 112
) -> np.ndarray:
    """
    把 OpenCV 2×3 仿射矩陣（src→dst）轉成 PyTorch affine_grid 的 theta（dst_norm→src_norm）。
    推導（align_corners=True）：
      theta[i,j]  = M_inv[i,j] × (dst_dim_j−1) / (src_dim_i−1)
      theta[i,2]  = Σ_j theta[i,j] + 2×M_inv[i,2] / (src_dim_i−1) − 1
    """
    M_inv = cv2.invertAffineTransform(M.astype(np.float64))
    t = np.zeros((1, 2, 3), dtype=np.float32)
    t[0, 0, 0] = M_inv[0, 0] * (dst_W - 1) / (src_W - 1)
    t[0, 0, 1] = M_inv[0, 1] * (dst_H - 1) / (src_W - 1)
    t[0, 0, 2] = t[0, 0, 0] + t[0, 0, 1] + 2 * M_inv[0, 2] / (src_W - 1) - 1
    t[0, 1, 0] = M_inv[1, 0] * (dst_W - 1) / (src_H - 1)
    t[0, 1, 1] = M_inv[1, 1] * (dst_H - 1) / (src_H - 1)
    t[0, 1, 2] = t[0, 1, 0] + t[0, 1, 1] + 2 * M_inv[1, 2] / (src_H - 1) - 1
    return t


def run_pgd_full(
    img_bgr: np.ndarray,
    eps_values: List[float],
    app: Any,
    steps: int = 100,
    device: Optional[str] = None,
    constraint: Any | None = None,
    tv_weight: float = 0.0,
    diagnostics: bool = False,
    precomputed_mask: np.ndarray | None = None,
) -> List[PgdResult]:
    """
    全圖 PGD：delta 維度 = 原圖大小，透過可微分仿射裁切讓梯度流回整張圖。
    最終輸出 = 原圖 + delta，無接縫無邊界，推薦使用。
    """
    try:
        import torch
        import torch.nn.functional as F
    except ModuleNotFoundError as e:
        raise RuntimeError("需要 PyTorch。請安裝 torch。") from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    face = pick_best_face(app.get(img_bgr), "input")

    from insightface.utils import face_align

    _, M = face_align.norm_crop2(
        img_bgr, landmark=face.kps, image_size=ALIGNED_FACE_SIZE
    )
    H, W = img_bgr.shape[:2]

    x_full = torch.from_numpy(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ).float().permute(2, 0, 1).unsqueeze(0).to(device)

    theta_t = torch.from_numpy(_cv2M_to_torch_theta(M, H, W)).to(device)
    grid = F.affine_grid(
        theta_t, (1, 3, ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE), align_corners=True
    )

    model = load_arcface_torch(device)

    def _emb_full(x: "torch.Tensor") -> "torch.Tensor":
        crop = F.grid_sample(
            x, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )
        return F.normalize(
            model((crop - ARCFACE_INPUT_MEAN) / ARCFACE_INPUT_SCALE), dim=1
        )

    if tv_weight < 0:
        raise ValueError("tv_weight must be non-negative")
    # The constraint is built once on clean B and is held fixed through the
    # iterative optimization.  It lives in original image coordinates, like
    # full-image delta; only the gradient, not the objective, is spatially
    # constrained.
    mask_t = None
    final_mask = None
    if constraint is not None or precomputed_mask is not None:
        mask = precomputed_mask if precomputed_mask is not None else constraint.build(img_bgr)[0]
        if mask.shape != img_bgr.shape[:2]:
            raise ValueError("precomputed mask shape must match the input image")
        final_mask = mask.copy()
        mask_t = torch.from_numpy(mask).to(device=device, dtype=x_full.dtype)[None, None]

    def _total_variation(delta: "torch.Tensor") -> "torch.Tensor":
        return (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean() + (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()

    with torch.no_grad():
        emb_ref = _emb_full(x_full).detach()

    results: List[PgdResult] = []
    for eps in eps_values:
        eps_px = eps * PIXEL_MAX
        step_size = eps_px / steps
        delta = torch.zeros_like(x_full)
        trace: list[dict[str, float | int | bool | None]] = []
        # These are detached uint8 snapshots only.  Keeping them in the
        # diagnostic result lets an external evaluator score FaceNet after the
        # attack, without ever placing the victim in this optimization loop.
        checkpoints: list[tuple[int, np.ndarray]] = []
        if diagnostics:
            checkpoints.append((0, img_bgr.copy()))
        previous_raw = None
        previous_update = None
        pending_record = None

        def _cos(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
            denom = a.norm() * b.norm()
            return torch.where(denom == 0, torch.full_like(denom, torch.nan), (a * b).sum().div(denom))

        for step in range(steps):
            delta.requires_grad_(True)
            cosine = (_emb_full(clip_pixel_range(x_full + delta)) * emb_ref).sum()
            if diagnostics and pending_record is not None:
                pending_record["arcface_cosine"] = cosine.detach()
            # TV is optional and intentionally independent of the mask.  The
            # negative update minimizes the cosine objective; positive TV is
            # added to the minimized loss only when explicitly requested.
            objective = cosine + float(tv_weight) * _total_variation(delta)
            objective.backward()
            with torch.no_grad():
                gradient = delta.grad.detach()
                # Diagnostic observations are made before the optional spatial
                # constraint.  They do not participate in autograd or updates.
                raw_gradient = gradient
                record: dict[str, float | int | bool | None] | None = None
                if diagnostics:
                    raw_abs = raw_gradient.abs()
                    record = {
                        "step": step + 1,
                        "raw_grad_l1": raw_abs.sum(),
                        "raw_grad_l2": raw_gradient.norm(),
                        "raw_grad_max_abs": raw_abs.max(),
                        "raw_grad_mean_abs": raw_abs.mean(),
                        "raw_grad_prev_cosine": _cos(raw_gradient, previous_raw) if previous_raw is not None else None,
                    }
                    if mask_t is not None:
                        # Derive outside L1 from total minus inside instead of
                        # materializing multiple full-image masked tensors.
                        # This is observation-only and leaves the subsequent
                        # masked update exactly as before.
                        inside_abs = raw_abs * mask_t
                        raw_l1, raw_l2 = raw_abs.sum(), raw_gradient.norm()
                        inside_l1 = inside_abs.sum()
                        elements_inside = mask_t.sum() * raw_gradient.shape[1]
                        elements_outside = (1.0 - mask_t).sum() * raw_gradient.shape[1]
                        record.update({
                            "inside_l1_ratio": inside_l1 / raw_l1.clamp_min(1e-12),
                            "inside_l2_ratio": torch.sqrt((raw_gradient.square() * mask_t).sum()) / raw_l2.clamp_min(1e-12),
                            "inside_grad_abs_sum": inside_l1,
                            "outside_grad_abs_sum": raw_l1 - inside_l1,
                            "mask_coverage": mask_t.mean(),
                            "inside_grad_mean_abs": inside_l1 / elements_inside.clamp_min(1),
                            "outside_grad_mean_abs": (raw_l1 - inside_l1) / elements_outside.clamp_min(1),
                        })
                if mask_t is not None:
                    gradient = gradient * mask_t
                update = -gradient.sign()
                delta = delta.detach() + step_size * update
                delta = project_delta_linf(delta, eps_px)
                if diagnostics:
                    assert record is not None
                    delta_abs = delta.abs()
                    changed = torch.any(delta != 0, dim=1)
                    record.update({
                        "objective": objective.detach(),
                        "arcface_threshold_crossing": None,  # populated by diagnostic runner using calibrated threshold
                        "current_linf": delta_abs.max() / PIXEL_MAX,
                        "current_l1": delta_abs.sum(),
                        "current_l2": delta.norm(),
                        "changed_pixel_ratio": changed.float().mean(),
                        "epsilon_saturation_ratio": (delta_abs >= eps_px - 1e-6).float().mean(),
                        "update_raw_grad_cosine": _cos(update, raw_gradient),
                        "update_prev_cosine": _cos(update, previous_update) if previous_update is not None else None,
                    })
                    trace.append(record)
                    pending_record = record
                    if (step + 1) % 10 == 0:
                        checkpoint_bgr = cv2.cvtColor(
                            clip_pixel_range(x_full + delta).squeeze(0).permute(1, 2, 0)
                            .cpu().numpy().clip(0, 255).astype(np.uint8),
                            cv2.COLOR_RGB2BGR,
                        )
                        checkpoints.append((step + 1, checkpoint_bgr))
                    previous_raw = raw_gradient.detach().clone()
                    previous_update = update.detach().clone()

        if diagnostics:
            trace[-1]["arcface_cosine"] = (_emb_full(clip_pixel_range(x_full + delta)) * emb_ref).sum().detach()
            tensor_cells = [(record, key, value) for record in trace for key, value in record.items() if torch.is_tensor(value)]
            if tensor_cells:
                values = torch.stack([value.reshape(()) for _, _, value in tensor_cells]).detach().cpu().numpy()
                for (record, key, _), value in zip(tensor_cells, values):
                    record[key] = None if np.isnan(value) else float(value)

        with torch.no_grad():
            adv = clip_pixel_range(x_full + delta)
        adv_bgr = cv2.cvtColor(
            adv.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        results.append(PgdResult(
            eps=eps, eps_255=eps_px, steps=steps, cosine=float("nan"),
            attacked_bgr=adv_bgr,
            # Measured before uint8 serialization; this is observational only
            # and does not alter the established PGD update or projection.
            linf_tensor=float(torch.max(torch.abs(adv - x_full)).item() / PIXEL_MAX),
            diagnostics={"trajectory": trace, "checkpoints": checkpoints, "operation_order": "raw gradient → optional mask → sign → update → L∞ projection → pixel clipping"} if diagnostics else None,
            final_mask=final_mask,
        ))
    return results


def run_pgd_demo(
    img_path: str | Path,
    out_dir: Path,
    det_size: tuple[int, int],
    eps_list_str: str,
    steps: int,
    save_noise_maps: bool,
    run_name: str | None = None,
    seed: int = DEFAULT_ATTACK_SEED,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = Path(img_path)
    stem = image_stem(img_path)
    img_bgr = load_bgr(img_path)
    eps_values = parse_eps_list(eps_list_str)

    try:
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)
    except Exception:
        pass

    app = create_face_app(det_size=det_size)
    print(f"步驟 A：PGD（{steps} 步）— 裁切臉後迭代，注意有接縫")
    results = run_pgd(img_bgr=img_bgr, eps_values=eps_values, app=app, steps=steps)

    config = build_run_config(
        attack_mode="pgd",
        source_image=img_path,
        eps_list=eps_values,
        steps=steps,
        seed=seed,
        det_size=det_size,
        run_name=run_name or out_dir.name,
    )
    evaluate_and_save_attack_run(
        results=results,
        img_bgr=img_bgr,
        stem=stem,
        out_dir=out_dir,
        app=app,
        mode="pgd",
        steps=steps,
        save_noise_maps=save_noise_maps,
        config=config,
        chart_title=f"InsightFace Self-Match Cosine vs Perturbation (PGD ({steps} steps))",
        chart_marker="s",
        chart_color="#d62728",
        chart_ylim=(-0.1, 1.05),
    )


def run_pgd_full_demo(
    img_path: str | Path,
    out_dir: Path,
    det_size: tuple[int, int],
    eps_list_str: str,
    steps: int,
    save_noise_maps: bool,
    run_name: str | None = None,
    seed: int = DEFAULT_ATTACK_SEED,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = Path(img_path)
    stem = image_stem(img_path)
    img_bgr = load_bgr(img_path)
    eps_values = parse_eps_list(eps_list_str)

    try:
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)
    except Exception:
        pass

    app = create_face_app(det_size=det_size)
    print(f"步驟 A：PGD Full（{steps} 步）— 全圖可微分裁切，無接縫")
    results = run_pgd_full(img_bgr=img_bgr, eps_values=eps_values, app=app, steps=steps)

    config = build_run_config(
        attack_mode="pgd_full",
        source_image=img_path,
        eps_list=eps_values,
        steps=steps,
        seed=seed,
        det_size=det_size,
        run_name=run_name or out_dir.name,
    )
    evaluate_and_save_attack_run(
        results=results,
        img_bgr=img_bgr,
        stem=stem,
        out_dir=out_dir,
        app=app,
        mode="pgd_full",
        steps=steps,
        save_noise_maps=save_noise_maps,
        config=config,
        chart_title=f"InsightFace Self-Match Cosine vs Perturbation (PGD Full ({steps} steps))",
        chart_marker="^",
        chart_color="#2ca02c",
        chart_ylim=(-0.1, 1.05),
    )
