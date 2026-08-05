"""
用 InsightFace 驗證對抗攻擊效果：比較原圖與對抗圖的 embedding 相似度。

  # 基本用法：比較原圖與對抗圖
  python scripts/verify_attack.py \
    --orig data/raw/sun.png \
    --adv  results/pgd/pgdfull_sun_v1/pgdfull_sun_eps_0.012.png

  # 掃描整個資料夾（新命名 adv_*.png 或舊命名皆可）
  python scripts/verify_attack.py \
    --orig data/raw/sun.png \
    --adv-dir results/pgd/pgdfull_sun_v1/ \
    --pattern "*.png"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from advface.config import DEFAULT_DET_SIZE, SIMILARITY_THRESHOLD, ensure_project_dirs, project_root
from advface.evaluation.attack_result import AttackResult, is_attack_success
from advface.evaluation.similarity import cosine_similarity, euclidean_distance
from advface.image_io import load_bgr
from advface.models.insightface_app import create_face_app, get_embedding_from_bgr


def _is_candidate_adv_image(path: Path) -> bool:
    """過濾 base／chart／metrics／noise／summary／config，保留對抗圖。"""
    name = path.name
    lower = name.lower()
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return False
    if lower == "config.json":
        return False
    if name.startswith("base_"):
        return False
    if name.startswith("noise_") or "_noise_" in name or "_noise_eps_" in name:
        return False
    if "chart" in lower or "metrics" in lower or "summary" in lower:
        return False
    return True


def verify_pair(app, orig_bgr, adv_bgr, label: str, threshold: float = SIMILARITY_THRESHOLD) -> AttackResult:
    emb_orig = get_embedding_from_bgr(app, orig_bgr, label="original")
    emb_adv = get_embedding_from_bgr(app, adv_bgr, label=label)
    cos = float(cosine_similarity(emb_orig, emb_adv))
    dist = float(euclidean_distance(emb_orig, emb_adv))
    return AttackResult(
        eps=float("nan"),
        cosine=cos,
        success=is_attack_success(cos, threshold),
        attack_mode="verify",
        output_image_path=label,
        euclidean=dist,
    )


def print_result(r: AttackResult) -> None:
    tag = "OK 攻擊成功" if r.success else "-- 攻擊失敗"
    euc = f"{r.euclidean:.4f}" if r.euclidean is not None else "n/a"
    print(f"  [{tag}]  cosine={r.cosine:.4f}  euclidean={euc}  ({r.output_image_path})")


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p.resolve()
    candidate = project_root() / path
    if candidate.exists():
        return candidate.resolve()
    return p


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orig", type=str, required=True, help="原始圖片路徑")
    p.add_argument("--adv", type=str, default=None, help="單張對抗圖路徑")
    p.add_argument("--adv-dir", type=str, default=None, help="對抗圖所在資料夾（批次模式）")
    p.add_argument("--pattern", type=str, default="*.png", help="批次模式的檔名 glob，預設 *.png")
    p.add_argument("--det-size", type=int, nargs=2, default=list(DEFAULT_DET_SIZE), metavar=("W", "H"))
    p.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=f"攻擊成功的 cosine 門檻，預設 {SIMILARITY_THRESHOLD}",
    )
    args = p.parse_args()

    threshold = args.threshold
    ensure_project_dirs()
    app = create_face_app(det_size=tuple(args.det_size))

    orig_path = _resolve_path(args.orig)
    try:
        orig_bgr = load_bgr(orig_path)
    except FileNotFoundError:
        print(f"[ERROR] 找不到原圖：{args.orig}")
        return 1

    print(f"\n原圖：{orig_path}")
    print(f"攻擊成功門檻：cosine < {threshold}")
    print("-" * 60)

    results: list[AttackResult] = []

    if args.adv:
        adv_path = _resolve_path(args.adv)
        try:
            adv_bgr = load_bgr(adv_path)
        except FileNotFoundError:
            print(f"[ERROR] 找不到對抗圖：{args.adv}")
            return 1
        r = verify_pair(app, orig_bgr, adv_bgr, label=adv_path.name, threshold=threshold)
        print_result(r)
        results.append(r)

    elif args.adv_dir:
        adv_dir = _resolve_path(args.adv_dir)
        files = sorted(f for f in adv_dir.glob(args.pattern) if _is_candidate_adv_image(f))
        if not files:
            print(f"[WARNING] 在 {adv_dir} 中找不到符合 {args.pattern} 的對抗圖")
            return 1
        for f in files:
            try:
                adv_bgr = load_bgr(f)
            except FileNotFoundError:
                continue
            r = verify_pair(app, orig_bgr, adv_bgr, label=f.name, threshold=threshold)
            print_result(r)
            results.append(r)
    else:
        print("[ERROR] 請指定 --adv 或 --adv-dir")
        return 1

    success_count = sum(1 for r in results if r.success)
    print("-" * 60)
    print(f"總計：{len(results)} 張，攻擊成功 {success_count} 張（{success_count/len(results)*100:.0f}%）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
