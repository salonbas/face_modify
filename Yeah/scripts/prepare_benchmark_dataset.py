"""
建立 Large-scale Transfer Benchmark 的 dataset contract。

來源優先順序：
  1. 倉庫既有 data/raw 人臉圖（local_raw）
  2. 公開研究資料集 LFW（Labeled Faces in the Wild, UMass）

不會從不明來源偷偷下載。LFW 下載失敗時停止該部分並保留已寫入的 local 列。

  python scripts/prepare_benchmark_dataset.py --max-images 200
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from advface.benchmark.dataset import ManifestRow, copy_image_into_benchmark, write_manifest
from advface.config import project_root
from advface.image_io import image_stem

LOCAL_NAMES = ("musk1.jpg", "face2.jpg", "face3.jpg", "sun.png")
LFW_HOME_PAGE = "http://vis-www.cs.umass.edu/lfw/"
LFW_NOTE = (
    "Labeled Faces in the Wild (LFW) is a public academic face dataset "
    "released by the University of Massachusetts for research. "
    "This project copies a subset of original JPEG files into data/benchmark/images/."
)


def _collect_local(root: Path, images_dir: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    raw = root / "data" / "raw"
    for name in LOCAL_NAMES:
        src = raw / name
        if not src.is_file():
            continue
        image_id = f"local_{image_stem(src)}"
        dest = copy_image_into_benchmark(src, images_dir, image_id)
        rel = dest.relative_to(root).as_posix()
        rows.append(
            ManifestRow(
                image_id=image_id,
                image_path=rel,
                identity_id=image_stem(src),
                source="local_raw",
            )
        )
    return rows


def _collect_lfw(root: Path, images_dir: Path, n_needed: int) -> tuple[list[ManifestRow], dict]:
    info: dict = {"attempted": True, "status": "ok"}
    if n_needed <= 0:
        info["status"] = "skipped"
        info["reason"] = "already have enough images"
        return [], info
    try:
        from sklearn.datasets import fetch_lfw_people, get_data_home
    except Exception as e:
        info["status"] = "blocked"
        info["reason"] = f"scikit-learn unavailable: {e}"
        return [], info

    try:
        fetch_lfw_people(color=True, download_if_missing=True, min_faces_per_person=1)
    except Exception as e:
        info["status"] = "blocked"
        info["reason"] = f"LFW download/load failed: {e}"
        return [], info

    funneled = Path(get_data_home()) / "lfw_home" / "lfw_funneled"
    if not funneled.is_dir():
        info["status"] = "blocked"
        info["reason"] = f"LFW funneled directory not found: {funneled}"
        return [], info

    people = sorted([p for p in funneled.iterdir() if p.is_dir()])
    picked: list[Path] = []
    for person_dir in people:
        jpgs = sorted(person_dir.glob("*.jpg"))
        if not jpgs:
            continue
        picked.append(jpgs[0])  # 一身份一張，提高多樣性
        if len(picked) >= n_needed:
            break

    rows: list[ManifestRow] = []
    for src in picked:
        identity = src.parent.name
        image_id = f"lfw_{identity}_{src.stem.split('_')[-1]}"
        dest = copy_image_into_benchmark(src, images_dir, image_id)
        rel = dest.relative_to(root).as_posix()
        rows.append(
            ManifestRow(
                image_id=image_id,
                image_path=rel,
                identity_id=identity,
                source="lfw",
            )
        )
    info["n_copied"] = len(rows)
    info["lfw_funneled_dir"] = str(funneled)
    return rows, info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-images", type=int, default=200)
    p.add_argument("--include-lfw", action="store_true", default=True)
    p.add_argument("--no-lfw", action="store_true")
    args = p.parse_args()

    root = project_root()
    bench = root / "data" / "benchmark"
    images_dir = bench / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_local(root, images_dir)
    # sun.png 解析度極大，不適合作為大量 PGD Full 的前段樣本；放到最後。
    rows_small = [r for r in rows if r.image_id != "local_sun"]
    rows_sun = [r for r in rows if r.image_id == "local_sun"]
    rows = rows_small
    lfw_info = {"attempted": False, "status": "skipped"}
    include_lfw = bool(args.include_lfw) and not bool(args.no_lfw)
    if include_lfw:
        extra, lfw_info = _collect_lfw(root, images_dir, max(0, int(args.max_images) - len(rows)))
        rows.extend(extra)
    rows.extend(rows_sun)

    rows = rows[: int(args.max_images)]
    write_manifest(bench / "manifest.csv", rows)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_images": len(rows),
        "sources": {
            "local_raw": {
                "n": sum(1 for r in rows if r.source == "local_raw"),
                "description": "Existing repository images under data/raw (project test assets).",
            },
            "lfw": {
                "n": sum(1 for r in rows if r.source == "lfw"),
                "homepage": LFW_HOME_PAGE,
                "note": LFW_NOTE,
                "fetch": lfw_info,
            },
        },
        "identity_pair_calibration": False,
        "contract": {
            "manifest": "data/benchmark/manifest.csv",
            "fields": ["image_id", "image_path", "identity_id", "source"],
        },
    }
    (bench / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (bench / "PROVENANCE.md").write_text(
        "# Benchmark dataset provenance\n\n"
        "This directory is the input contract for Large-scale Transfer Benchmark v0.\n\n"
        "## Layout\n\n"
        "- `images/` copied files used by the benchmark\n"
        "- `manifest.csv` columns: image_id, image_path, identity_id, source\n"
        "- `provenance.json` machine-readable source record\n\n"
        "## Sources\n\n"
        "1. **local_raw** — existing `data/raw` images already in this repository.\n"
        "2. **lfw** — Labeled Faces in the Wild (University of Massachusetts), "
        f"{LFW_HOME_PAGE} — public academic dataset for research. "
        "One image per identity is copied when fetch succeeds.\n\n"
        "No undocumented downloads are performed.\n"
        "This benchmark does not require identity-pair calibration.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows -> {bench / 'manifest.csv'}")
    print(f"LFW fetch: {lfw_info}")
    if include_lfw and lfw_info.get("status") == "blocked":
        print("LFW 取得被阻擋；local 部分已寫入。請人工確認 license/download 後再重跑。", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
