"""
Single-image transfer experiment.

  python scripts/run_transfer.py --image data/raw/sun.png
  python scripts/run_transfer.py --image data/raw/musk1.jpg --attack fgsm --eps 0.04
  python scripts/run_transfer.py --image data/raw/sun.png --attack pgd_full --eps 0.04 --steps 200

走 canonical run_experiment：Attack → Surrogate / Victim evaluation → report.html
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from advface.attacks.registry import AttackConfig, normalize_attack_name
from advface.config import (
    DEFAULT_DET_SIZE,
    DEFAULT_IMAGE,
    DEFAULT_TRANSFER_EPS,
    DEFAULT_TRANSFER_STEPS,
    DEFAULT_VICTIM_MODEL,
    ensure_project_dirs,
    project_root,
)
from advface.experiments.output import file_sha256
from advface.experiments.runner import run_experiment
from advface.experiments.transfer_output import save_transfer_run
from advface.image_io import image_stem, load_bgr
from advface.models import InsightFaceEmbedder, load_embedder
from advface.paths import make_run_dir

EXPERIMENT_NAME = "Transfer Evaluation Experiment"
PIPELINE_ID = "canonical_transfer_experiment"


def _resolve_img(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    candidate = project_root() / path
    if candidate.is_file():
        return candidate.resolve()
    return p


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", "--img", dest="image", type=str, default=DEFAULT_IMAGE)
    p.add_argument("--attack", type=str, default="pgd_full", help="已註冊攻擊名稱（預設 pgd_full）")
    p.add_argument("--eps", type=str, default=DEFAULT_TRANSFER_EPS)
    p.add_argument("--steps", type=int, default=DEFAULT_TRANSFER_STEPS)
    p.add_argument("--det-size", type=int, nargs=2, default=list(DEFAULT_DET_SIZE), metavar=("W", "H"))
    p.add_argument("--victim", type=str, default=DEFAULT_VICTIM_MODEL)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    ensure_project_dirs()
    img_path = _resolve_img(args.image)
    if not img_path.is_file():
        print(f"找不到圖片：{args.image}", file=sys.stderr)
        return 1

    stem = image_stem(img_path)
    eps = float(str(args.eps).strip().split(",")[0])
    attack = normalize_attack_name(args.attack)
    det_size = tuple(args.det_size)

    run_name = args.run_name or datetime.now().strftime("transfer_%Y%m%d-%H%M%S")
    out = Path(args.out_dir) if args.out_dir else make_run_dir(attack_type="transfer", run_name=run_name)
    out.mkdir(parents=True, exist_ok=True)
    run_id = out.name

    print("=" * 60)
    print(EXPERIMENT_NAME)
    print(f"  image   : {img_path}")
    print(f"  attack  : {attack}  eps={eps:.3f}  steps={args.steps}")
    print(f"  victim  : {args.victim}")
    print(f"  out     : {out}")
    print("=" * 60)

    original_bgr = load_bgr(img_path)
    surrogate = InsightFaceEmbedder(det_size=det_size)
    victim = load_embedder(args.victim)
    result = run_experiment(
        original_bgr,
        attack=attack,
        surrogate=surrogate,
        victims={"victim": victim},
        eps=eps,
        steps=int(args.steps),
        config=AttackConfig(name=attack, eps=eps, steps=int(args.steps)),
        device=args.device,
        app=surrogate.app,
    )
    if result.adversarial_bgr is None:
        print("攻擊未產生對抗圖", file=sys.stderr)
        return 1

    sur = result.surrogate
    vic = result.primary_victim
    print(
        f"  surrogate [{sur.model_name}]  "
        f"cos_after={sur.cosine_after:.4f}  success={sur.success}"
    )
    if vic is None:
        print("  victim    missing", file=sys.stderr)
        return 1
    if vic.error:
        print(f"  victim    [{vic.model_name}]  ERROR: {vic.error}")
    else:
        print(
            f"  victim    [{vic.model_name}]  "
            f"cos_after={vic.cosine_after:.4f}  "
            f"transfer_observed={vic.transfer_observed}"
        )

    try:
        rel_image = str(img_path.relative_to(project_root()))
    except ValueError:
        rel_image = str(img_path)

    config = {
        "pipeline": PIPELINE_ID,
        "experiment": EXPERIMENT_NAME,
        "attack_mode": attack,
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
        "perturbation": result.perturbation,
    }
    attack_info = {
        "type": attack,
        "name": attack,
        **result.attack_parameters,
    }
    paths = save_transfer_run(
        out_dir=out,
        original_bgr=original_bgr,
        adversarial_bgr=result.adversarial_bgr,
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
