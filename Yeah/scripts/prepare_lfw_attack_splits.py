#!/usr/bin/env python3
"""Create frozen, calibration-disjoint LFW verification evaluation splits."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from advface.config import project_root

SEED = 20260902
EXPECTED_ELIGIBLE_IDENTITIES = 328
EXPECTED_ELIGIBLE_IMAGES = 1517
SPLIT_SIZES = {"development": 20, "test": 100, "reserve": 208}
FIELDS = ("identity_id", "reference_image_id", "reference_image", "probe_image_id", "probe_image", "available_images", "split")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--calibration-source", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = project_root()
    dataset = root / "data/datasets/lfw"
    manifest = args.manifest or dataset / "manifest.csv"
    calibration = args.calibration_source or root / "results/calibration/arcface_lfw_v1/pair_scores.csv"
    out = args.output_dir or dataset / "evaluation_splits"

    # The frozen census is based on calibration *used* pairs.  Invalid pairs
    # were recorded as failures and must not silently exclude their identities.
    calibration_rows = [row for row in read_csv(calibration) if row.get("valid", "").lower() == "true"]
    calibrated = {
        row[key] for row in calibration_rows for key in ("identity_a", "identity_b")
        if row.get(key)
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(manifest):
        if row.get("usable", "").lower() == "true" and row["identity_id"] not in calibrated:
            grouped[row["identity_id"]].append(row)
    eligible = {identity: sorted(rows, key=lambda row: (row["original_filename"], row["image_id"]))
                for identity, rows in grouped.items() if len(rows) >= 2}
    image_count = sum(len(rows) for rows in eligible.values())
    if len(eligible) != EXPECTED_ELIGIBLE_IDENTITIES or image_count != EXPECTED_ELIGIBLE_IMAGES:
        raise RuntimeError(
            f"eligibility census mismatch: identities={len(eligible)} images={image_count}; "
            f"expected {EXPECTED_ELIGIBLE_IDENTITIES}/{EXPECTED_ELIGIBLE_IMAGES}"
        )
    identities = sorted(eligible)
    random.Random(SEED).shuffle(identities)
    boundaries = (SPLIT_SIZES["development"], SPLIT_SIZES["development"] + SPLIT_SIZES["test"])
    assignments = {
        "development": identities[:boundaries[0]],
        "test": identities[boundaries[0]:boundaries[1]],
        "reserve": identities[boundaries[1]:],
    }
    out.mkdir(parents=True, exist_ok=True)
    for split, split_ids in assignments.items():
        rows = []
        for identity in split_ids:
            images = eligible[identity]
            rows.append({"identity_id": identity, "reference_image_id": images[0]["image_id"],
                         "reference_image": images[0]["image_path"], "probe_image_id": images[1]["image_id"],
                         "probe_image": images[1]["image_path"], "available_images": len(images), "split": split})
        with (out / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    config = {
        "protocol": "LFW White-box Verification Attack Protocol v1",
        "seed": SEED,
        "selection": "eligible calibration-unused identities shuffled with Python random.Random(seed)",
        "pair_selection": "canonical original_filename ascending; first image is reference A, second is probe B; remaining images unused",
        "calibration_identity_source": str(calibration),
        "manifest_source": str(manifest),
        "eligible_identity_count": len(eligible), "eligible_image_count": image_count,
        "splits": {key: len(value) for key, value in assignments.items()},
    }
    (out / "split_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote frozen attack splits to {out}: " + ", ".join(f"{key}={len(value)}" for key, value in assignments.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
