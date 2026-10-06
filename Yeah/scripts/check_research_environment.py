#!/usr/bin/env python3
"""Read-only completeness check for the research reproducibility contract."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def exists(relative_path: str) -> bool:
    return (PROJECT_ROOT / relative_path).is_file()


def directory_exists(relative_path: str) -> bool:
    return (PROJECT_ROOT / relative_path).is_dir()


def report(label: str, present: bool) -> bool:
    print(f"{'✓' if present else '✗'} {label}")
    return present


def main() -> int:
    required = (
        ("frozen attack_dev split", exists("data/datasets/lfw/evaluation_splits/attack_dev.csv")),
        ("LFW manifest", exists("data/datasets/lfw/manifest.csv")),
        ("benchmark provenance", exists("data/benchmark/provenance.json")),
        ("ArcFace calibration metadata", exists("results/calibration/arcface_lfw_v1/thresholds.json")),
        ("FaceNet calibration metadata", exists("results/calibration/facenet_lfw_v1/thresholds.json")),
        ("formal four-method result", exists("results/transfer/lfw_attack_dev_20_four_method_v3/summary.json")),
        ("formal per-identity result", exists("results/transfer/lfw_attack_dev_20_four_method_v3/per_identity_results.csv")),
        ("mechanism analysis", exists("docs/reports/meeting_2026_09_30/assets/analysis/mechanism_analysis.json")),
    )
    optional = (
        ("LFW image cache", directory_exists("data/datasets/lfw/images")),
        ("ArcFace model weight", (Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx").is_file()),
        ("FaceNet model weight", exists(".cache/torch/checkpoints/20180402-114759-vggface2.pt")),
    )

    complete = all(report(label, present) for label, present in required)
    for label, present in optional:
        report(label, present)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
