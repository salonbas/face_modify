"""
用 InsightFace 驗證對抗攻擊效果：比較原圖與對抗圖的 embedding 相似度。

  # 基本用法：比較原圖與對抗圖
  python scripts/verify_attack.py \
    --orig data/raw/sun.png \
    --adv  results/pgd/pgdfull_sun_v1/pgdfull_sun_eps_0.012.png

  # 掃描整個資料夾，一次比較所有對抗圖
  python scripts/verify_attack.py \
    --orig data/raw/sun.png \
    --adv-dir results/pgd/pgdfull_sun_v1/ \
    --pattern "pgdfull_*.png"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2

from advface.config import DEFAULT_DET_SIZE, ensure_project_dirs
from advface.insightface_backend import create_face_app, get_embedding_from_bgr
from advface.metrics import cosine_similarity, euclidean_distance


THRESHOLD = 0.4  # cosine 低於此值視為「攻擊成功」


def verify_pair(app, orig_bgr, adv_bgr, label: str, threshold: float = 0.4) -> dict:
    emb_orig = get_embedding_from_bgr(app, orig_bgr, label="original")
    emb_adv  = get_embedding_from_bgr(app, adv_bgr,  label=label)
    cos  = float(cosine_similarity(emb_orig, emb_adv))
    dist = float(euclidean_distance(emb_orig, emb_adv))
    success = cos < threshold
    return {"label": label, "cosine": cos, "euclidean": dist, "success": success}


def print_result(r: dict) -> None:
    tag = "OK 攻擊成功" if r["success"] else "-- 攻擊失敗"
    print(f"  [{tag}]  cosine={r['cosine']:.4f}  euclidean={r['euclidean']:.4f}  ({r['label']})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orig", type=str, required=True, help="原始圖片路徑")
    p.add_argument("--adv",  type=str, default=None,  help="單張對抗圖路徑")
    p.add_argument("--adv-dir", type=str, default=None, help="對抗圖所在資料夾（批次模式）")
    p.add_argument("--pattern", type=str, default="*.png", help="批次模式的檔名 glob，預設 *.png")
    p.add_argument("--det-size", type=int, nargs=2, default=list(DEFAULT_DET_SIZE), metavar=("W", "H"))
    p.add_argument("--threshold", type=float, default=THRESHOLD, help=f"攻擊成功的 cosine 門檻，預設 {THRESHOLD}")
    args = p.parse_args()

    threshold = args.threshold
    ensure_project_dirs()
    app = create_face_app(det_size=tuple(args.det_size))

    orig_bgr = cv2.imread(args.orig)
    if orig_bgr is None:
        print(f"[ERROR] 找不到原圖：{args.orig}")
        return 1

    print(f"\n原圖：{args.orig}")
    print(f"攻擊成功門檻：cosine < {threshold}")
    print("-" * 60)

    results = []

    if args.adv:
        adv_bgr = cv2.imread(args.adv)
        if adv_bgr is None:
            print(f"[ERROR] 找不到對抗圖：{args.adv}")
            return 1
        r = verify_pair(app, orig_bgr, adv_bgr, label=Path(args.adv).name, threshold=threshold)
        print_result(r)
        results.append(r)

    elif args.adv_dir:
        adv_dir = Path(args.adv_dir)
        files = sorted(adv_dir.glob(args.pattern))
        files = [f for f in files if not f.name.startswith("base_")]
        if not files:
            print(f"[WARNING] 在 {adv_dir} 中找不到符合 {args.pattern} 的檔案")
            return 1
        for f in files:
            adv_bgr = cv2.imread(str(f))
            if adv_bgr is None:
                continue
            r = verify_pair(app, orig_bgr, adv_bgr, label=f.name, threshold=threshold)
            print_result(r)
            results.append(r)
    else:
        print("[ERROR] 請指定 --adv 或 --adv-dir")
        return 1

    # 摘要
    success_count = sum(1 for r in results if r["success"])
    print("-" * 60)
    print(f"總計：{len(results)} 張，攻擊成功 {success_count} 張（{success_count/len(results)*100:.0f}%）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
