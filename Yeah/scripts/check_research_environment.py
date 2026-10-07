#!/usr/bin/env python3
"""Read-only completeness check for the research reproducibility contract.

The checker distinguishes a downloaded Git-LFS object from an LFS pointer. It
never downloads assets and is safe to run in a fresh clone.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"
LFW_IMAGE_COUNT = 13_233


def asset_state(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing: {path}"
    try:
        with path.open("rb") as handle:
            if handle.read(len(LFS_POINTER_HEADER)) == LFS_POINTER_HEADER:
                return False, f"Git LFS pointer not downloaded: {path}"
    except OSError as exc:
        return False, f"unreadable: {path} ({exc})"
    return True, str(path)


def repository_file(relative_path: str) -> tuple[bool, str]:
    return asset_state(PROJECT_ROOT / relative_path)


def lfw_images_state() -> tuple[bool, str]:
    images = PROJECT_ROOT / "data/datasets/lfw/images"
    if not images.is_dir():
        return False, f"missing: {images} (expected {LFW_IMAGE_COUNT} JPEGs)"
    count = sum(1 for path in images.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg"})
    if count != LFW_IMAGE_COUNT:
        return False, f"incomplete: {images} ({count}/{LFW_IMAGE_COUNT} JPEGs)"
    return True, f"{images} ({count} JPEGs)"


def report(label: str, state: tuple[bool, str]) -> bool:
    present, detail = state
    print(f"{'✓' if present else '✗'} {label}: {detail}")
    return present


def main() -> int:
    insightface = PROJECT_ROOT / "models/insightface/buffalo_l"
    required = (
        ("frozen split", repository_file("data/datasets/lfw/evaluation_splits/attack_dev.csv")),
        ("manifest", repository_file("data/datasets/lfw/manifest.csv")),
        ("LFW raw images", lfw_images_state()),
        ("ArcFace / buffalo_l recognition model", asset_state(insightface / "w600k_r50.onnx")),
        ("detector", asset_state(insightface / "det_10g.onnx")),
        ("68-point landmark model", asset_state(insightface / "1k3d68.onnx")),
        ("FaceNet VGGFace2 weight", repository_file("models/facenet/20180402-114759-vggface2.pt")),
        ("ArcFace calibration", repository_file("results/calibration/arcface_lfw_v1/thresholds.json")),
        ("FaceNet calibration", repository_file("results/calibration/facenet_lfw_v1/thresholds.json")),
        ("formal four-method result", repository_file("results/transfer/lfw_attack_dev_20_four_method_v3/summary.json")),
        ("formal per-identity result", repository_file("results/transfer/lfw_attack_dev_20_four_method_v3/per_identity_results.csv")),
        ("mechanism analysis", repository_file("docs/reports/meeting_2026_09_30/assets/analysis/mechanism_analysis.json")),
    )

    # Do not short-circuit: a clean-clone diagnosis must display every missing
    # asset, not merely the first one.
    complete = all([report(label, state) for label, state in required])
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
