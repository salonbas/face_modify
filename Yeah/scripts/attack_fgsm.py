"""
InsightFace ArcFace 白盒攻擊：FGSM / PGD / PGD Full

  # FGSM（單步，最快）
  python scripts/attack_fgsm.py --mode fgsm --run-name fgsm_sun

  # PGD（多步，裁切後貼回，有接縫）
  python scripts/attack_fgsm.py --mode pgd --steps 20 --run-name pgd_sun

  # PGD Full（多步全圖，無接縫，推薦）
  python scripts/attack_fgsm.py --mode pgd_full --steps 100 --run-name pgd_full_sun

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
from advface.config import DEFAULT_DET_SIZE, DEFAULT_FGSM_EPS_LIST, DEFAULT_IMAGE, ensure_project_dirs
from advface.paths import make_run_dir


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img", type=str, default=DEFAULT_IMAGE)
    p.add_argument("--eps-list", type=str, default=DEFAULT_FGSM_EPS_LIST)
    p.add_argument("--det-size", type=int, nargs=2, default=list(DEFAULT_DET_SIZE), metavar=("W", "H"))
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--mode",
        choices=["fgsm", "pgd", "pgd_full"],
        default="fgsm",
        help="攻擊模式：fgsm（單步）/ pgd（多步有接縫）/ pgd_full（多步無接縫，推薦）",
    )
    p.add_argument("--steps", type=int, default=20, help="PGD 迭代步數（pgd/pgd_full 有效），預設 20")
    p.add_argument("--no-save-noise-maps", action="store_false", dest="save_noise_maps")
    p.set_defaults(save_noise_maps=True)
    args = p.parse_args()

    ensure_project_dirs()
    attack_type = "fgsm" if args.mode == "fgsm" else "pgd"
    out = Path(args.out_dir) if args.out_dir else make_run_dir(attack_type=attack_type, run_name=args.run_name)

    kwargs = dict(
        img_path=args.img,
        out_dir=out,
        det_size=tuple(args.det_size),
        eps_list_str=args.eps_list,
        save_noise_maps=args.save_noise_maps,
    )

    if args.mode == "fgsm":
        run_fgsm_demo(**kwargs)
    elif args.mode == "pgd":
        run_pgd_demo(**kwargs, steps=args.steps)
    else:
        run_pgd_full_demo(**kwargs, steps=args.steps)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
