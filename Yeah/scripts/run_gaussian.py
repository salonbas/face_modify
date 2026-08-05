"""
高斯噪聲對 InsightFace embedding 的影響（掃描 eps、輸出 CSV／折線圖）。

  python scripts/run_gaussian.py --img data/raw/sun.png --run-name demo --save-noisy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from advface.attacks.gaussian import run_gaussian_noise_experiment
from advface.config import (
    DEFAULT_DET_SIZE,
    DEFAULT_GAUSSIAN_EPS_END,
    DEFAULT_GAUSSIAN_EPS_START,
    DEFAULT_GAUSSIAN_EPS_STEP,
    DEFAULT_IMAGE,
    ensure_project_dirs,
    project_root,
)
from advface.paths import make_run_dir


def _resolve_img(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    candidate = project_root() / path
    if candidate.is_file():
        return candidate.resolve()
    return p


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--img", type=str, default=DEFAULT_IMAGE)
    p.add_argument("--det-size", type=int, nargs=2, default=list(DEFAULT_DET_SIZE), metavar=("W", "H"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps-start", type=int, default=DEFAULT_GAUSSIAN_EPS_START)
    p.add_argument("--eps-end", type=int, default=DEFAULT_GAUSSIAN_EPS_END)
    p.add_argument("--eps-step", type=int, default=DEFAULT_GAUSSIAN_EPS_STEP)
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="輸出目錄；未指定則 results/gaussian/<--run-name 或時間戳>",
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--save-noisy", action="store_true", help="另存各 eps 圖於 out/noise/")
    args = p.parse_args()

    ensure_project_dirs()
    out = Path(args.out_dir) if args.out_dir else make_run_dir(attack_type="gaussian", run_name=args.run_name)
    run_gaussian_noise_experiment(
        img_path=_resolve_img(args.img),
        out_dir=out,
        det_size=tuple(args.det_size),
        seed=args.seed,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_step=args.eps_step,
        save_noisy=args.save_noisy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
