"""
Batch transfer benchmark：N images × M configs。

只負責 grid / checkpoint / resume / aggregation。
每個 unit 走 canonical run_experiment。

  python scripts/run_benchmark.py \\
      --manifest data/benchmark/manifest.csv \\
      --max-images 100 \\
      --attacks fgsm pgd_full \\
      --eps 0.005 0.01 0.02 0.03 0.04 \\
      --pgd-steps 20 50 100 200 \\
      --victim facenet_vggface2 \\
      --run-name transfer_benchmark_v0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from advface.benchmark.runner import run_benchmark
from advface.config import (
    DEFAULT_ATTACK_SEED,
    DEFAULT_BENCHMARK_ATTACKS,
    DEFAULT_BENCHMARK_EPS,
    DEFAULT_BENCHMARK_PGD_STEPS,
    DEFAULT_DET_SIZE,
    DEFAULT_VICTIM_MODEL,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=str, default="data/benchmark/manifest.csv")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--attacks", nargs="+", default=list(DEFAULT_BENCHMARK_ATTACKS))
    p.add_argument("--eps", nargs="+", type=float, default=list(DEFAULT_BENCHMARK_EPS))
    p.add_argument("--pgd-steps", nargs="+", type=int, default=list(DEFAULT_BENCHMARK_PGD_STEPS))
    p.add_argument("--victim", type=str, default=DEFAULT_VICTIM_MODEL)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--run-name", type=str, default="transfer_benchmark_v0")
    p.add_argument("--seed", type=int, default=DEFAULT_ATTACK_SEED)
    p.add_argument("--det-size", type=int, nargs=2, default=list(DEFAULT_DET_SIZE), metavar=("W", "H"))
    p.add_argument("--save-all-images", action="store_true")
    p.add_argument("--retry-failures", action="store_true")
    p.add_argument("--retry-invalid", action="store_true")
    p.add_argument("--skip-examples", action="store_true")
    args = p.parse_args()

    out = run_benchmark(
        manifest_path=args.manifest,
        max_images=args.max_images,
        attacks=list(args.attacks),
        eps_list=[float(x) for x in args.eps],
        pgd_steps=[int(x) for x in args.pgd_steps],
        victim=args.victim,
        device=args.device,
        run_name=args.run_name,
        seed=int(args.seed),
        det_size=tuple(args.det_size),
        save_all_images=bool(args.save_all_images),
        retry_failures=bool(args.retry_failures),
        retry_invalid=bool(args.retry_invalid),
        skip_examples=bool(args.skip_examples),
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
