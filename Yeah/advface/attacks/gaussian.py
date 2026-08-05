from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from advface.config import GAUSSIAN_FAIL_THRESHOLD
from advface.evaluation.similarity import cosine_similarity_or_nan
from advface.image_io import image_stem, load_bgr
from advface.models.insightface_app import create_face_app, get_embedding_from_bgr


def run_gaussian_noise_experiment(
    img_path: str | Path,
    out_dir: Path,
    det_size: tuple[int, int],
    seed: int,
    eps_start: int,
    eps_end: int,
    eps_step: int,
    save_noisy: bool,
) -> None:
    np.random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_dir = out_dir / "noise" if save_noisy else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    eps_values = list(range(eps_start, eps_end + 1, eps_step))
    if not eps_values:
        raise ValueError("eps 範圍無有效值，請確認 eps-start / eps-end / eps-step。")

    app = create_face_app(det_size=det_size)
    base_bgr = load_bgr(img_path)
    base_f = base_bgr.astype(np.float32)
    stem = image_stem(img_path)

    emb_base = get_embedding_from_bgr(app, base_bgr, str(img_path))
    similarities: list[float] = []

    for epsilon in eps_values:
        noise = np.random.normal(0, epsilon, base_f.shape).astype(np.float32)
        attacked_bgr = np.clip(base_f + noise, 0, 255).astype(np.uint8)

        if save_dir:
            out_path = save_dir / f"test_eps_{epsilon}_{stem}.jpg"
            cv2.imwrite(str(out_path), attacked_bgr)

        try:
            emb_attacked = get_embedding_from_bgr(app, attacked_bgr, f"attacked eps={epsilon}")
            sim = cosine_similarity_or_nan(emb_base, emb_attacked)
        except Exception as exc:
            print(f"eps={epsilon:>3} | [偵測/embedding失敗] {exc}")
            sim = float("nan")

        similarities.append(sim)
        status = "FAILED" if np.isnan(sim) or sim < GAUSSIAN_FAIL_THRESHOLD else "OK"
        cos_text = "nan" if np.isnan(sim) else f"{sim:.6f}"
        print(f"eps={epsilon:>3} | cosine={cos_text} | {status}")

    csv_path = out_dir / "gaussian_metrics.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("eps,cosine,success\n")
        for eps, s in zip(eps_values, similarities):
            success = int(np.isnan(s) or s < GAUSSIAN_FAIL_THRESHOLD)
            cos_str = "nan" if np.isnan(s) else f"{s:.8f}"
            f.write(f"{eps},{cos_str},{success}\n")

    fig = plt.figure(figsize=(10, 5))
    plt.plot(eps_values, similarities, marker="o", color="#1f77b4")
    plt.axhline(
        y=GAUSSIAN_FAIL_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"threshold {GAUSSIAN_FAIL_THRESHOLD}",
    )
    plt.title("InsightFace: Cosine Similarity vs Gaussian Epsilon")
    plt.xlabel("Epsilon (Gaussian std)")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    chart_path = out_dir / "gaussian_cosine_chart.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)

    print(f"\n已輸出 CSV: {csv_path}")
    print(f"已輸出圖表: {chart_path}")
