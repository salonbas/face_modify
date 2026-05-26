"""
FGSM（Fast Gradient Sign Method）— 單步 L∞ 白盒攻擊

原理：
  對 InsightFace 的 ArcFace 模型（w600k_r50.onnx）計算一次梯度，
  沿著讓「自匹配 cosine」下降的方向走一步：

    x_adv = x - (eps × 255) × sign( ∂cosine / ∂x )

  eps 越大攻擊越強，但擾動越明顯。建議從 eps=0.01 開始掃描。

輸出檔案（存入 out_dir）：
  fgsm_metrics.csv          每個 eps 的 cosine 數值
  fgsm_chart.png            cosine vs eps 折線圖
  fgsm_summary.png          原圖 + 各 eps 對抗圖總覽
  fgsm_{stem}_{eps:.3f}.png 各 eps 的對抗圖
  fgsm_noise_{eps:.3f}.png  擾動熱圖（加 --save-noise-maps 才產生）
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from advface.attacks._arcface import load_arcface_torch
from advface.image_io import image_stem, load_bgr
from advface.insightface_backend import create_face_app, get_embedding_from_bgr, pick_best_face
from advface.metrics import cosine_similarity


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------


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
    Minv = cv2.invertAffineTransform(M.astype(np.float32))
    warped = cv2.warpAffine(adv_crop_bgr, Minv, (w, h))
    mask = cv2.warpAffine(np.ones((112, 112), np.float32), Minv, (w, h))
    m3 = np.stack([np.clip(mask, 0, 1)] * 3, axis=2)
    out = img_bgr.astype(np.float32) * (1 - m3) + warped.astype(np.float32) * m3
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 核心攻擊
# ---------------------------------------------------------------------------


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

    crop_bgr, M = face_align.norm_crop2(img_bgr, landmark=face.kps, image_size=112)
    x0 = torch.from_numpy(
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    ).float().permute(2, 0, 1).unsqueeze(0).to(device)

    model = load_arcface_torch(device)

    def _emb(x: "torch.Tensor") -> "torch.Tensor":
        return F.normalize(model((x - 127.5) / 127.5), dim=1)

    with torch.no_grad():
        emb_ref = _emb(x0).detach()

    results: List[FgsmResult] = []
    for eps in eps_values:
        x = x0.clone().detach().requires_grad_(True)
        (_emb(x) * emb_ref).sum().backward()
        x_adv = torch.clamp(x.detach() - eps * 255.0 * x.grad.detach().sign(), 0, 255)

        adv_crop_bgr = cv2.cvtColor(
            x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        results.append(FgsmResult(
            eps=eps, eps_255=eps * 255.0, cosine=float("nan"),
            attacked_bgr=_paste_aligned_patch(img_bgr, adv_crop_bgr, M),
        ))
    return results


# ---------------------------------------------------------------------------
# Demo：執行 + InsightFace 驗收 + 存檔
# ---------------------------------------------------------------------------


def run_fgsm_demo(
    img_path: str | Path,
    out_dir: Path,
    det_size: tuple[int, int],
    eps_list_str: str,
    save_noise_maps: bool,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = Path(img_path)
    stem = image_stem(img_path)
    img_bgr = load_bgr(img_path)
    eps_values = parse_eps_list(eps_list_str)

    try:
        import torch
        torch.manual_seed(0)
        np.random.seed(0)
    except Exception:
        pass

    app = create_face_app(det_size=det_size)
    print("步驟 A：FGSM — ArcFace 梯度單步降低自匹配 cosine")
    results = run_fgsm(img_bgr=img_bgr, eps_values=eps_values, app=app)

    print("步驟 B：InsightFace 驗收，寫入 CSV／圖表")
    emb_base = get_embedding_from_bgr(app, img_bgr, label="base")
    cv2.imwrite(str(out_dir / f"fgsm_base_{stem}.png"), img_bgr)

    csv_path = out_dir / "fgsm_metrics.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("eps,eps_255,cosine\n")
        for r in results:
            try:
                r.cosine = float(cosine_similarity(
                    emb_base, get_embedding_from_bgr(app, r.attacked_bgr, label=f"eps={r.eps}")
                ))
                cos_str, status = f"{r.cosine:.8f}", f"cosine={r.cosine:.6f}"
            except ValueError:
                r.cosine = float("nan")
                cos_str, status = "nan", "cosine=NaN（擾動過大，人臉偵測失敗）"
            f.write(f"{r.eps:.6f},{r.eps_255:.3f},{cos_str}\n")
            cv2.imwrite(str(out_dir / f"fgsm_{stem}_{r.eps:.3f}.png"), r.attacked_bgr)
            if save_noise_maps:
                _save_noise_map(
                    _bgr_to_rgb_float(r.attacked_bgr) - _bgr_to_rgb_float(img_bgr),
                    out_dir / f"fgsm_noise_{r.eps:.3f}.png", img_bgr,
                )
            print(f"  eps={r.eps:.4f} (≈{r.eps_255:.1f}/255)  {status}")

    eps_sorted = sorted(results, key=lambda r: r.eps)

    # 折線圖
    plt.figure(figsize=(8, 4.8))
    plt.plot([r.eps_255 for r in eps_sorted], [r.cosine for r in eps_sorted], marker="o", color="#1f77b4")
    plt.axhline(0.4, color="gray", linestyle="--", linewidth=1.2, label="threshold 0.4")
    plt.title("InsightFace Self-Match Cosine vs Perturbation (FGSM)")
    plt.xlabel("Perturbation strength (eps x 255)")
    plt.ylabel("Cosine (original vs adversarial)")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    chart_path = out_dir / "fgsm_chart.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=160)
    plt.close()

    # 總覽圖
    try:
        rows = len(eps_sorted) + 1
        fig, axes = plt.subplots(rows, 1, figsize=(5, rows * 3))
        axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original ({stem})")
        axes[0].axis("off")
        for i, r in enumerate(eps_sorted, 1):
            axes[i].imshow(cv2.cvtColor(r.attacked_bgr, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"eps={r.eps:.3f}  cos={r.cosine:.3f}")
            axes[i].axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / "fgsm_summary.png", dpi=160)
        plt.close(fig)
    except Exception:
        pass

    print(f"\n已輸出 CSV: {csv_path}")
    print(f"已輸出圖表: {chart_path}")
    print(f"已輸出總覽: {out_dir / 'fgsm_summary.png'}")
