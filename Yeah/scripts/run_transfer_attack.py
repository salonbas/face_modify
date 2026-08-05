"""
Transfer Evaluation Experiment v1

一次完整實驗：
  Input Image → PGD Full（既有白盒）→ Adversarial
  → ArcFace Surrogate Evaluation → FaceNet Victim Evaluation → report.html

  python scripts/run_transfer_attack.py --image data/raw/sun.png
  python scripts/run_transfer_attack.py --image data/raw/musk1.jpg --run-name transfer_musk_v1

正式預設：eps=0.040、steps=200（與既有 PGD 研究設定一致）。
不重寫／不修改 PGD；Victim（FaceNet）僅做 inference。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from advface.attacks.pgd import run_pgd_full_demo
from advface.config import (
    DEFAULT_DET_SIZE,
    DEFAULT_IMAGE,
    DEFAULT_TRANSFER_EPS,
    DEFAULT_TRANSFER_STEPS,
    DEFAULT_VICTIM_MODEL,
    ensure_project_dirs,
    project_root,
)
from advface.evaluation.transfer import evaluate_transfer
from advface.experiments.output import adv_image_name, base_image_name, file_sha256
from advface.experiments.transfer_output import save_transfer_run
from advface.image_io import image_stem, load_bgr
from advface.models.facenet import FaceNetEmbedder
from advface.models.insightface_app import InsightFaceEmbedder
from advface.paths import make_run_dir

EXPERIMENT_NAME = "Transfer Evaluation Experiment v1"
PIPELINE_ID = "transfer_evaluation_experiment_v1"


def _resolve_img(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    candidate = project_root() / path
    if candidate.is_file():
        return candidate.resolve()
    return p


def _load_victim(name: str):
    key = name.strip().lower()
    if key in ("facenet", "facenet_vggface2"):
        return FaceNetEmbedder()
    if key in ("insightface", "buffalo_l", "insightface_buffalo_l"):
        return InsightFaceEmbedder()
    raise ValueError(f"未知 victim 模型：{name}（支援：facenet）")


def _pick_adv_path(whitebox_dir: Path, mode: str, stem: str, eps: float) -> Path:
    expected = whitebox_dir / adv_image_name(mode, stem, eps)
    if expected.is_file():
        return expected
    matches = sorted(whitebox_dir.glob(f"adv_{mode}_{stem}_eps_*.png"))
    if not matches:
        raise FileNotFoundError(f"找不到對抗圖於 {whitebox_dir}")
    return matches[-1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--image",
        "--img",
        dest="image",
        type=str,
        default=DEFAULT_IMAGE,
        help="輸入圖片路徑",
    )
    p.add_argument("--eps", type=str, default=DEFAULT_TRANSFER_EPS, help="單一 eps（正式預設 0.040）")
    p.add_argument("--steps", type=int, default=DEFAULT_TRANSFER_STEPS, help="PGD steps（正式預設 200）")
    p.add_argument("--det-size", type=int, nargs=2, default=list(DEFAULT_DET_SIZE), metavar=("W", "H"))
    p.add_argument("--victim", type=str, default=DEFAULT_VICTIM_MODEL, help="victim 模型：facenet（預設）")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    ensure_project_dirs()
    img_path = _resolve_img(args.image)
    if not img_path.is_file():
        print(f"找不到圖片：{args.image}", file=sys.stderr)
        return 1

    stem = image_stem(img_path)
    eps = float(str(args.eps).strip().split(",")[0])
    mode = "pgd_full"
    det_size = tuple(args.det_size)

    run_name = args.run_name or datetime.now().strftime("transfer_v1_%Y%m%d-%H%M%S")
    out = Path(args.out_dir) if args.out_dir else make_run_dir(attack_type="transfer", run_name=run_name)
    out.mkdir(parents=True, exist_ok=True)
    run_id = out.name

    # 白盒中間產物寫入子目錄，最終契約檔在 run 根目錄
    whitebox_dir = out / "whitebox"
    whitebox_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(EXPERIMENT_NAME)
    print(f"  image   : {img_path}")
    print(f"  attack  : {mode}  eps={eps:.3f}  steps={args.steps}")
    print(f"  victim  : {args.victim}")
    print(f"  out     : {out}")
    print("=" * 60)

    # --- Step 1: White-box PGD（完全重用既有 run_pgd_full_demo）---
    print("\n[1/3] Running white-box PGD Full (existing repository path)…")
    run_pgd_full_demo(
        img_path=img_path,
        out_dir=whitebox_dir,
        det_size=det_size,
        eps_list_str=f"{eps:.6f}",
        steps=args.steps,
        save_noise_maps=True,
        run_name=f"{run_id}_whitebox",
    )

    adv_path = _pick_adv_path(whitebox_dir, mode, stem, eps)
    base_path = whitebox_dir / base_image_name(stem)
    if base_path.is_file():
        original_bgr = load_bgr(base_path)
    else:
        original_bgr = load_bgr(img_path)
    adversarial_bgr = load_bgr(adv_path)
    print(f"  adversarial: {adv_path.name}")

    # --- Step 2: Surrogate + Victim evaluation（Victim 僅 inference）---
    print("\n[2/3] Evaluating surrogate (ArcFace) and victim (FaceNet)…")
    surrogate = InsightFaceEmbedder(det_size=det_size)
    victim = _load_victim(args.victim)
    results = evaluate_transfer(
        original_bgr,
        adversarial_bgr,
        surrogate=surrogate,
        victim=victim,
    )
    sur = results["surrogate"]
    vic = results["victim"]
    print(
        f"  surrogate [{sur.model_name}]  "
        f"cos_after={sur.cosine_after:.4f}  success={sur.success}"
    )
    if vic.error:
        print(f"  victim    [{vic.model_name}]  ERROR: {vic.error}")
    else:
        print(
            f"  victim    [{vic.model_name}]  "
            f"cos_after={vic.cosine_after:.4f}  "
            f"transfer_observed={vic.transfer_observed}"
        )

    # --- Step 3: Output contract + report.html ---
    print("\n[3/3] Writing experiment outputs + report.html…")
    try:
        rel_image = str(img_path.relative_to(project_root()))
    except ValueError:
        rel_image = str(img_path)

    config = {
        "pipeline": PIPELINE_ID,
        "experiment": EXPERIMENT_NAME,
        "attack_mode": mode,
        "source_image": rel_image,
        "source_image_abs": str(img_path),
        "source_image_hash": file_sha256(img_path),
        "eps": eps,
        "steps": int(args.steps),
        "seed": 0,
        "detector_size": list(det_size),
        "surrogate_model": sur.model_name,
        "victim_model": vic.model_name,
        "run_name": run_id,
        "whitebox_dir": str(whitebox_dir),
        "adversarial_source": str(adv_path),
    }
    attack_info = {
        "type": mode,
        "eps": eps,
        "steps": int(args.steps),
    }
    paths = save_transfer_run(
        out_dir=out,
        original_bgr=original_bgr,
        adversarial_bgr=adversarial_bgr,
        surrogate=sur,
        victim=vic,
        config=config,
        attack=attack_info,
        run_id=run_id,
        experiment=EXPERIMENT_NAME,
        image_path=rel_image,
        image_stem=stem,
        victim_model_name=vic.model_name,
    )

    print("\nDone.")
    print(f"  report : {paths['report']}")
    print(f"  metrics: {paths['metrics_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
