#!/usr/bin/env python3
"""Read-only completeness check for the research reproducibility contract.

The checker distinguishes a downloaded Git-LFS object from an LFS pointer. It
never downloads assets and is safe to run in a fresh clone.
"""

from __future__ import annotations

import sys
from pathlib import Path
import hashlib
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"
LFW_IMAGE_COUNT = 13_233
MANIFEST = json.loads((PROJECT_ROOT / "models/checkpoint_manifest.json").read_text())


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


def checkpoint_state(relative_path: str, model: str) -> tuple[bool, str]:
    path = PROJECT_ROOT / relative_path
    present, detail = asset_state(path)
    if not present:
        return present, detail
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    expected = MANIFEST[model]["sha256"]
    return digest == expected, f"{path} SHA-256 {digest}" + ("" if digest == expected else f" != {expected}")


def checksum_state(relative_path: str, expected: str) -> tuple[bool, str]:
    path = PROJECT_ROOT / relative_path
    present, detail = asset_state(path)
    if not present:
        return present, detail
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return digest == expected, f"{path} SHA-256 {digest}" + ("" if digest == expected else f" != {expected}")


def source_state(relative_path: str, model: str) -> tuple[bool, str]:
    path = PROJECT_ROOT / relative_path
    present, detail = asset_state(path)
    if not present:
        return present, detail
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    expected = MANIFEST[model]["model_source_sha256"]
    return digest == expected, f"{path} SHA-256 {digest}" + ("" if digest == expected else f" != {expected}")


def lfw_images_state() -> tuple[bool, str]:
    images = PROJECT_ROOT / "data/datasets/lfw/images"
    if not images.is_dir():
        return False, f"missing: {images} (expected {LFW_IMAGE_COUNT} JPEGs)"
    count = sum(1 for path in images.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg"})
    if count != LFW_IMAGE_COUNT:
        return False, f"incomplete: {images} ({count}/{LFW_IMAGE_COUNT} JPEGs)"
    hashes = PROJECT_ROOT / "data/datasets/lfw/images_sha256.txt"
    if not hashes.is_file():
        return False, f"missing LFW checksum manifest: {hashes}"
    expected = dict(line.split("  ", 1) for line in hashes.read_text(encoding="utf-8").splitlines())
    files = {p.relative_to(images).as_posix(): p for p in images.rglob("*.jpg")}
    if len(expected) != LFW_IMAGE_COUNT or set(files) != set(expected.values()):
        return False, "LFW file list differs from frozen checksum manifest"
    for digest, relative in expected.items():
        with files[relative].open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest:
            return False, f"LFW checksum mismatch: {relative}"
    return True, f"{images} ({count} JPEGs, SHA-256 manifest verified)"


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
        ("ArcFace / buffalo_l recognition model", checkpoint_state("models/insightface/buffalo_l/w600k_r50.onnx", "arcface")),
        ("detector", asset_state(insightface / "det_10g.onnx")),
        ("68-point landmark model", asset_state(insightface / "1k3d68.onnx")),
        ("FaceNet VGGFace2 weight", checkpoint_state("models/facenet/20180402-114759-vggface2.pt", "facenet")),
        ("AdaFace weight", checkpoint_state("models/adaface/adaface_ir50_ms1mv2.ckpt", "adaface")),
        ("MobileFaceNet weight", checkpoint_state("models/insightface/buffalo_sc/w600k_mbf.onnx", "mobilefacenet")),
        ("MobileFaceNet detector", checksum_state("models/insightface/buffalo_sc/det_500m.onnx", MANIFEST["mobilefacenet"]["detector_sha256"])),
        ("MagFace weight", checkpoint_state("models/magface/magface_iresnet100_ms1mv2.pth", "magface")),
        ("AdaFace source", source_state("models/third_party/adaface/net.py", "adaface")),
        ("MagFace source", source_state("models/third_party/magface/iresnet.py", "magface")),
        ("ArcFace calibration", repository_file("results/calibration/arcface_lfw_v1/thresholds.json")),
        ("FaceNet calibration", repository_file("results/calibration/facenet_lfw_v1/thresholds.json")),
        *((f"{name} v2 calibration", repository_file(f"results/calibration/{name}_lfw_v2/thresholds.json")) for name in MANIFEST),
        ("Development 100 split", repository_file("data/datasets/lfw/evaluation_splits/v2/development.csv")),
        ("OOS 228 split", repository_file("data/datasets/lfw/evaluation_splits/v2/oos.csv")),
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
