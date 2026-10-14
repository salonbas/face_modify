"""實驗輸出契約：檔名、CSV、圖表、config.json。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from advface.attacks.common import _bgr_to_rgb_float, _save_noise_map
from advface.config import (
    FACE_MODEL_NAME,
    SIMILARITY_THRESHOLD,
    insightface_providers,
)
from advface.evaluation.attack_result import AttackResult
from advface.evaluation.transfer import evaluate_on_model
from advface.models.insightface_app import InsightFaceEmbedder


# ---------------------------------------------------------------------------
# 命名規格（新 run；舊 results/ 不更名）
# ---------------------------------------------------------------------------
#   base_<stem>.png
#   adv_<mode>_<stem>_eps_<eps>.png
#   <mode>_metrics.csv
#   <mode>_cosine_chart.png
#   config.json
#   <mode>_noise_eps_<eps>.png   （可選）
#   <mode>_summary.png           （可選，FGSM 總覽）
# ---------------------------------------------------------------------------


def format_eps(eps: float) -> str:
    return f"{eps:.3f}"


def base_image_name(stem: str) -> str:
    return f"base_{stem}.png"


def adv_image_name(mode: str, stem: str, eps: float) -> str:
    return f"adv_{mode}_{stem}_eps_{format_eps(eps)}.png"


def metrics_csv_name(mode: str) -> str:
    return f"{mode}_metrics.csv"


def cosine_chart_name(mode: str) -> str:
    return f"{mode}_cosine_chart.png"


def noise_map_name(mode: str, eps: float) -> str:
    return f"{mode}_noise_eps_{format_eps(eps)}.png"


def summary_image_name(mode: str) -> str:
    return f"{mode}_summary.png"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_config_json(out_dir: Path, config: dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.json"
    payload = dict(config)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def build_run_config(
    *,
    attack_mode: str,
    source_image: str | Path,
    eps_list: Sequence[float] | str,
    steps: Optional[int],
    seed: int,
    det_size: tuple[int, int],
    run_name: Optional[str] = None,
    provider: Optional[Sequence[str]] = None,
    model_name: str = FACE_MODEL_NAME,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    src = Path(source_image)
    cfg: dict[str, Any] = {
        "attack_mode": attack_mode,
        "source_image": str(src),
        "source_image_hash": file_sha256(src) if src.is_file() else None,
        "eps_list": list(eps_list) if not isinstance(eps_list, str) else eps_list,
        "steps": steps,
        "seed": seed,
        "model_name": model_name,
        "detector_size": list(det_size),
        "provider": list(provider) if provider is not None else insightface_providers(),
        "run_name": run_name,
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }
    if extra:
        cfg.update(extra)
    return cfg


def write_metrics_csv(out_dir: Path, mode: str, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(out_dir) / metrics_csv_name(mode)
    rows = list(rows)
    with path.open("w", encoding="utf-8") as f:
        f.write("eps,eps_255,steps,cosine,success\n")
        for r in rows:
            steps = r.get("steps", "")
            steps_str = "" if steps is None or steps == "" else str(int(steps))
            cos = r["cosine"]
            cos_str = "nan" if cos != cos else f"{float(cos):.8f}"
            success = int(r.get("success", 0))
            f.write(
                f"{float(r['eps']):.6f},{float(r['eps_255']):.3f},{steps_str},{cos_str},{success}\n"
            )
    return path


def save_cosine_chart(
    out_dir: Path,
    mode: str,
    eps_255: Sequence[float],
    cosines: Sequence[float],
    *,
    title: str,
    marker: str = "o",
    color: str = "#1f77b4",
    ylim: tuple[float, float] = (-0.1, 1.05),
) -> Path:
    path = Path(out_dir) / cosine_chart_name(mode)
    plt.figure(figsize=(8, 4.8))
    plt.plot(list(eps_255), list(cosines), marker=marker, color=color)
    plt.axhline(
        SIMILARITY_THRESHOLD,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"threshold {SIMILARITY_THRESHOLD}",
    )
    plt.title(title)
    plt.xlabel("Perturbation strength (eps x 255)")
    plt.ylabel("Cosine (original vs adversarial)")
    plt.ylim(*ylim)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def evaluate_and_save_attack_run(
    *,
    results: List[Any],
    img_bgr: np.ndarray,
    stem: str,
    out_dir: Path,
    app: Any,
    mode: str,
    steps: Optional[int],
    save_noise_maps: bool,
    config: dict[str, Any],
    chart_title: str,
    chart_marker: str = "o",
    chart_color: str = "#1f77b4",
    chart_ylim: tuple[float, float] = (-0.1, 1.05),
    save_summary: bool = False,
) -> List[AttackResult]:
    """
    InsightFace 驗收 + 依新命名規格寫檔。
    `results` 元素需有：eps, eps_255, attacked_bgr；可選 steps；cosine 會被寫回。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / base_image_name(stem)), img_bgr)
    write_config_json(out_dir, config)

    embedder = InsightFaceEmbedder(app=app)
    print("步驟 B：InsightFace 驗收，寫入 CSV／圖表")
    evaluated: List[AttackResult] = []
    metric_rows: List[dict[str, Any]] = []

    for r in results:
        r_steps = getattr(r, "steps", steps)
        ev = evaluate_on_model(img_bgr, r.attacked_bgr, embedder)
        cos = float(ev.cosine_after)
        if ev.error or cos != cos:
            status = "cosine=NaN（擾動過大，人臉偵測失敗）" if cos != cos else f"error={ev.error}"
        else:
            status = f"cosine={cos:.6f}"

        r.cosine = cos
        success = bool(ev.success)
        adv_name = adv_image_name(mode, stem, r.eps)
        adv_path = out_dir / adv_name
        cv2.imwrite(str(adv_path), r.attacked_bgr)

        if save_noise_maps:
            _save_noise_map(
                _bgr_to_rgb_float(r.attacked_bgr) - _bgr_to_rgb_float(img_bgr),
                out_dir / noise_map_name(mode, r.eps),
                img_bgr,
            )

        ar = AttackResult(
            eps=float(r.eps),
            cosine=cos,
            success=success,
            attack_mode=mode,
            output_image_path=str(adv_path),
            steps=int(r_steps) if r_steps is not None else None,
            euclidean=ev.euclidean_distance,
            eps_255=float(r.eps_255),
            attacked_bgr=r.attacked_bgr,
        )
        evaluated.append(ar)
        metric_rows.append(ar.to_metrics_row())

        step_info = f" steps={r_steps}" if r_steps is not None else ""
        print(f"  eps={r.eps:.4f} (≈{r.eps_255:.1f}/255){step_info}  {status}  success={success}")

    csv_path = write_metrics_csv(out_dir, mode, metric_rows)
    eps_sorted = sorted(evaluated, key=lambda x: x.eps)
    chart_path = save_cosine_chart(
        out_dir,
        mode,
        [a.eps_255 if a.eps_255 is not None else a.eps * 255.0 for a in eps_sorted],
        [a.cosine for a in eps_sorted],
        title=chart_title,
        marker=chart_marker,
        color=chart_color,
        ylim=chart_ylim,
    )

    if save_summary:
        try:
            rows = len(eps_sorted) + 1
            fig, axes = plt.subplots(rows, 1, figsize=(5, rows * 3))
            if rows == 1:
                axes = [axes]
            axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"Original ({stem})")
            axes[0].axis("off")
            for i, a in enumerate(eps_sorted, 1):
                axes[i].imshow(cv2.cvtColor(a.attacked_bgr, cv2.COLOR_BGR2RGB))
                axes[i].set_title(f"eps={a.eps:.3f}  cos={a.cosine:.3f}")
                axes[i].axis("off")
            plt.tight_layout()
            fig.savefig(out_dir / summary_image_name(mode), dpi=160)
            plt.close(fig)
        except Exception:
            pass

    print(f"\n已輸出 CSV: {csv_path}")
    print(f"已輸出圖表: {chart_path}")
    print(f"已輸出 config: {out_dir / 'config.json'}")
    return evaluated
