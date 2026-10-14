"""
FGSM（Fast Gradient Sign Method）— 單步 L∞ 白盒攻擊

原理：
  對 InsightFace 的 ArcFace 模型（w600k_r50.onnx）計算一次梯度，
  沿著讓「自匹配 cosine」下降的方向走一步：

    x_adv = x - (eps × 255) × sign( ∂cosine / ∂x )

  eps 越大攻擊越強，但擾動越明顯。建議從 eps=0.01 開始掃描。

新輸出命名見 advface.experiments.output。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np

from advface.attacks.common import (  # noqa: F401
    _bgr_to_rgb_float,
    _paste_aligned_patch,
    _save_noise_map,
    parse_eps_list,
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
class FgsmResult:
    eps: float
    eps_255: float
    cosine: float
    attacked_bgr: np.ndarray


def run_fgsm(
    img_bgr: np.ndarray,
    eps_values: List[float],
    app: Any,
    device: Optional[str] = None,
) -> List[FgsmResult]:
    """
    單步 L∞ 攻擊：對每個 eps 各計算一次梯度並更新。

    app：已 prepare 的 FaceAnalysis（與後續驗收共用，避免重複載入模型）。
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

    results: List[FgsmResult] = []
    for eps in eps_values:
        x = x0.clone().detach().requires_grad_(True)
        (_emb(x) * emb_ref).sum().backward()
        x_adv = torch.clamp(
            x.detach() - eps * PIXEL_MAX * x.grad.detach().sign(),
            PIXEL_MIN,
            PIXEL_MAX,
        )

        adv_crop_bgr = cv2.cvtColor(
            x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        results.append(FgsmResult(
            eps=eps, eps_255=eps * PIXEL_MAX, cosine=float("nan"),
            attacked_bgr=_paste_aligned_patch(img_bgr, adv_crop_bgr, M),
        ))
    return results


def run_fgsm_demo(
    img_path: str | Path,
    out_dir: Path,
    det_size: tuple[int, int],
    eps_list_str: str,
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
    print("步驟 A：FGSM — ArcFace 梯度單步降低自匹配 cosine")
    results = run_fgsm(img_bgr=img_bgr, eps_values=eps_values, app=app)

    config = build_run_config(
        attack_mode="fgsm",
        source_image=img_path,
        eps_list=eps_values,
        steps=1,
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
        mode="fgsm",
        steps=1,
        save_noise_maps=save_noise_maps,
        config=config,
        chart_title="InsightFace Self-Match Cosine vs Perturbation (FGSM)",
        chart_marker="o",
        chart_color="#1f77b4",
        chart_ylim=(0.0, 1.05),
        save_summary=True,
    )
