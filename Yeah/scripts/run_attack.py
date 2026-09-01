"""
InsightFace ArcFace 白盒攻擊：FGSM / PGD / PGD Full

  # FGSM（單步，最快）
  python scripts/run_attack.py --mode fgsm --run-name fgsm_sun

  # PGD（多步，裁切後貼回，有接縫）
  python scripts/run_attack.py --mode pgd --steps 20 --run-name pgd_sun

  # PGD Full（多步全圖，無接縫，推薦）
  python scripts/run_attack.py --mode pgd_full --steps 100 --run-name pgd_full_sun

需：pip install onnx2torch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from advface.attacks.fgsm import run_fgsm_demo
from advface.attacks.pgd import run_pgd_demo, run_pgd_full_demo
from advface.attacks.registry import ATTACKS, normalize_attack_name
from advface.config import (
    DEFAULT_DET_SIZE,
    DEFAULT_FGSM_EPS_LIST,
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img", type=str, default=DEFAULT_IMAGE)
    p.add_argument("--eps-list", type=str, default=DEFAULT_FGSM_EPS_LIST)
    p.add_argument("--det-size", type=int, nargs=2, default=list(DEFAULT_DET_SIZE), metavar=("W", "H"))
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--mode",
        choices=sorted(ATTACKS),
        default="fgsm",
        help="攻擊模式：fgsm / pgd / pgd_full（推薦）",
    )
    p.add_argument("--steps", type=int, default=20, help="PGD 迭代步數（pgd/pgd_full 有效），預設 20")
    p.add_argument("--no-save-noise-maps", action="store_false", dest="save_noise_maps")
    p.set_defaults(save_noise_maps=True)
    args = p.parse_args()

    ensure_project_dirs()
    mode = normalize_attack_name(args.mode)
    attack_type = "fgsm" if mode == "fgsm" else "pgd"
    out = Path(args.out_dir) if args.out_dir else make_run_dir(attack_type=attack_type, run_name=args.run_name)
    img_path = _resolve_img(args.img)

    kwargs = dict(
        img_path=img_path,
        out_dir=out,
        det_size=tuple(args.det_size),
        eps_list_str=args.eps_list,
        save_noise_maps=args.save_noise_maps,
        run_name=args.run_name or out.name,
    )

    demos = {
        "fgsm": lambda: run_fgsm_demo(**kwargs),
        "pgd": lambda: run_pgd_demo(**kwargs, steps=args.steps),
        "pgd_full": lambda: run_pgd_full_demo(**kwargs, steps=args.steps),
    }
    demos[mode]()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
