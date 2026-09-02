#!/usr/bin/env python3
"""Materialize lightweight metadata for an existing sklearn LFW cache.

This script intentionally never downloads data.  It validates the standard
sklearn cache, creates the repository entry/symlink, and writes manifests and
the official pair definitions.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

EXPECTED_IMAGES = 13_233
EXPECTED_IDENTITIES = 5_749


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-home",
        type=Path,
        default=Path(os.environ.get("SKLEARN_DATA_HOME", "~/scikit_learn_data")).expanduser(),
        help="existing sklearn data home; no download is performed",
    )
    return parser.parse_args()


def image_id(identity: str, number: int) -> str:
    return f"{identity}_{number:04d}"


def write_manifest(out: Path, image_root: Path) -> tuple[int, int]:
    files = sorted(image_root.glob("*/*.jpg"))
    identities = sorted({p.parent.name for p in files})
    if len(files) != EXPECTED_IMAGES or len(identities) != EXPECTED_IDENTITIES:
        raise RuntimeError(
            f"incomplete/unexpected LFW cache: {len(files)} images, "
            f"{len(identities)} identities"
        )
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "image_id", "identity_id", "image_path", "source",
                "split", "variant", "usable", "original_filename",
            ),
        )
        writer.writeheader()
        for path in files:
            identity = path.parent.name
            number = int(path.stem.rsplit("_", 1)[1])
            writer.writerow(
                {
                    "image_id": image_id(identity, number),
                    "identity_id": identity,
                    "image_path": f"data/datasets/lfw/images/{identity}/{path.name}",
                    "source": "scikit-learn_lfw-funneled_cache",
                    "split": "all",
                    "variant": "funneled",
                    "usable": "true",
                    "original_filename": path.name,
                }
            )
    return len(files), len(identities)


def parse_pairs(path: Path) -> list[dict[str, str]]:
    lines = [line.strip().split() for line in path.read_text(encoding="utf-8").splitlines()]
    if not lines or len(lines[0]) not in (1, 2):
        raise RuntimeError(f"invalid pair protocol header: {path}")
    if len(lines[0]) == 1:
        # Development files record the number of positive and negative pairs.
        expected = int(lines[0][0]) * 2
    else:
        # The evaluation header is: number_of_folds pairs_per_class_per_fold.
        expected = int(lines[0][0]) * int(lines[0][1]) * 2
    rows: list[dict[str, str]] = []
    for fields in lines[1:]:
        if len(fields) == 3:
            identity_a, number_a, number_b = fields
            identity_b = identity_a
            same = "true"
        elif len(fields) == 4:
            identity_a, number_a, identity_b, number_b = fields
            same = "false"
        else:
            raise RuntimeError(f"invalid pair row in {path}: {' '.join(fields)}")
        a = image_id(identity_a, int(number_a))
        b = image_id(identity_b, int(number_b))
        rows.append(
            {
                "image_a": a,
                "image_b": b,
                "identity_a": identity_a,
                "identity_b": identity_b,
                "same_identity": same,
            }
        )
    if len(rows) != expected:
        raise RuntimeError(f"pair count mismatch in {path}: header={expected}, rows={len(rows)}")
    return rows


def write_pairs(out: Path, protocol_root: Path) -> None:
    pair_dir = out / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    specs = [("dev_train", protocol_root / "pairsDevTrain.txt")]
    specs.append(("dev_test", protocol_root / "pairsDevTest.txt"))
    fieldnames = ("image_a", "image_b", "identity_a", "identity_b", "same_identity", "source_split")
    all_eval: list[dict[str, str]] = []
    for split, source in specs:
        rows = parse_pairs(source)
        for row in rows:
            row["source_split"] = split
        with (pair_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    evaluation_rows = parse_pairs(protocol_root / "pairs.txt")
    if len(evaluation_rows) != 6_000:
        raise RuntimeError(f"unexpected evaluation protocol size: {len(evaluation_rows)}")
    for i in range(10):
        split = f"evaluation_{i + 1:02d}"
        rows = evaluation_rows[i * 600 : (i + 1) * 600]
        for row in rows:
            row["source_split"] = split
        with (pair_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        all_eval.extend(rows)
    with (pair_dir / "evaluation_10fold.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_eval)


def main() -> None:
    args = parse_args()
    cache = args.data_home / "lfw_home"
    images = cache / "lfw_funneled"
    required = [images, cache / "pairsDevTrain.txt", cache / "pairsDevTest.txt", cache / "pairs.txt"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("LFW cache is incomplete; missing: " + ", ".join(missing))
    repo_root = Path(__file__).resolve().parents[1]
    out = repo_root / "data" / "datasets" / "lfw"
    out.mkdir(parents=True, exist_ok=True)
    link = out / "images"
    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.resolve() == images.resolve():
            pass
        else:
            raise SystemExit(f"refusing to replace existing path: {link}")
    else:
        link.symlink_to(os.path.relpath(images, out), target_is_directory=True)
    count, identities = write_manifest(out, images)
    write_pairs(out, cache)
    print(f"validated LFW funneled cache: {count} images, {identities} identities")
    print(f"wrote canonical metadata: {out}")


if __name__ == "__main__":
    main()
