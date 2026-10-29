#!/usr/bin/env python3
"""Install the fixed runtime assets from their upstream publishers.

This script deliberately does not use or copy a user's caches.  It installs
the LFW images and the exact model files under this checkout, validates every
model checksum, and leaves them ignored by Git pending redistribution approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LFW_IMAGES = ROOT / "data/datasets/lfw/images"
BUFFALO_DIR = ROOT / "models/insightface/buffalo_l"
FACENET_PATH = ROOT / "models/facenet/20180402-114759-vggface2.pt"
MOBILE_PATH = ROOT / "models/insightface/buffalo_sc/w600k_mbf.onnx"
ADAFACE_PATH = ROOT / "models/adaface/adaface_ir50_ms1mv2.ckpt"
MAGFACE_PATH = ROOT / "models/magface/magface_iresnet100_ms1mv2.pth"
LFW_COUNT = 13_233
LFW_HASHES = ROOT / "data/datasets/lfw/images_sha256.txt"
LFW_ARCHIVE_SHA256 = "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a"
LFW_URLS = (
    "https://ndownloader.figshare.com/files/5976015",
    "https://huggingface.co/datasets/DerrickUnleashed/LFW/resolve/main/lfwfunneled.tgz",
)
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
FACENET_URL = "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"
MOBILE_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_sc.zip"
MANIFEST = json.loads((ROOT / "models/checkpoint_manifest.json").read_text())
CHECKSUMS = {
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "1k3d68.onnx": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
    "2d106det.onnx": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
    "genderage.onnx": "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
    FACENET_PATH.name: "281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)


def verify(path: Path) -> None:
    actual = sha256(path)
    expected = CHECKSUMS[path.name]
    if actual != expected:
        raise RuntimeError(f"checksum mismatch for {path}: {actual} != {expected}")


def verify_manifest(path: Path, model: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {model}: {path}")
    actual = sha256(path)
    expected = MANIFEST[model]["sha256"]
    if actual != expected:
        raise RuntimeError(f"checksum mismatch for {model}: {actual} != {expected}")


def verify_lfw() -> None:
    expected = {}
    for line in LFW_HASHES.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    actual_paths = {p.relative_to(LFW_IMAGES).as_posix() for p in LFW_IMAGES.rglob("*.jpg")}
    if len(expected) != LFW_COUNT or actual_paths != set(expected):
        raise RuntimeError("LFW file list differs from frozen checksum manifest")
    for relative, digest in expected.items():
        if sha256(LFW_IMAGES / relative) != digest:
            raise RuntimeError(f"LFW checksum mismatch: {relative}")


def install_lfw() -> None:
    if LFW_IMAGES.is_dir() and sum(1 for p in LFW_IMAGES.rglob("*.jpg")) == LFW_COUNT:
        print(f"LFW already complete: {LFW_IMAGES}")
        return
    # The mirror contains byte-for-byte the archive whose SHA-256 is published
    # in scikit-learn's _lfw.py. Its 13,233 JPEGs are also checked individually.
    with tempfile.TemporaryDirectory(prefix="advface-lfw-") as raw_tmp:
        tmp = Path(raw_tmp)
        archive = tmp / "lfw-funneled.tgz"
        failures = []
        for url in LFW_URLS:
            try:
                download(url, archive)
                if sha256(archive) != LFW_ARCHIVE_SHA256:
                    raise RuntimeError("LFW archive SHA-256 mismatch")
                break
            except Exception as exc:
                failures.append(f"{url}: {exc}")
                archive.unlink(missing_ok=True)
        else:
            raise RuntimeError("LFW download failed from all recorded sources:\n" + "\n".join(failures))
        source = tmp / "lfw_funneled"
        source.mkdir()
        with tarfile.open(archive, "r:gz") as package:
            for member in package:
                parts = Path(member.name).parts
                if not member.isfile() or len(parts) != 3 or parts[0] != "lfw_funneled" or not parts[-1].endswith(".jpg"):
                    continue
                if any(part in {".", ".."} for part in parts):
                    raise RuntimeError(f"unsafe LFW archive entry: {member.name}")
                destination = source / parts[1] / parts[2]
                destination.parent.mkdir(parents=True, exist_ok=True)
                with package.extractfile(member) as input_file, destination.open("wb") as output_file:
                    shutil.copyfileobj(input_file, output_file)
        if sum(1 for p in source.rglob("*.jpg")) != LFW_COUNT:
            raise RuntimeError("LFW archive has an unexpected JPEG count")
        LFW_IMAGES.parent.mkdir(parents=True, exist_ok=True)
        if LFW_IMAGES.exists() or LFW_IMAGES.is_symlink():
            if LFW_IMAGES.is_dir() and not LFW_IMAGES.is_symlink():
                shutil.rmtree(LFW_IMAGES)
            else:
                LFW_IMAGES.unlink()
        shutil.move(str(source), str(LFW_IMAGES))


def install_models() -> None:
    with tempfile.TemporaryDirectory(prefix="advface-assets-") as raw_tmp:
        tmp = Path(raw_tmp)
        archive = tmp / "buffalo_l.zip"
        if not all((BUFFALO_DIR / name).is_file() for name in CHECKSUMS if name != FACENET_PATH.name):
            download(BUFFALO_URL, archive)
            with zipfile.ZipFile(archive) as package:
                members = {Path(name).name: name for name in package.namelist() if name.endswith(".onnx")}
                missing = set(CHECKSUMS) - {FACENET_PATH.name} - set(members)
                if missing:
                    raise RuntimeError(f"buffalo_l archive missing: {sorted(missing)}")
                BUFFALO_DIR.mkdir(parents=True, exist_ok=True)
                for name, member in members.items():
                    if name in CHECKSUMS:
                        with package.open(member) as source, (BUFFALO_DIR / name).open("wb") as dest:
                            shutil.copyfileobj(source, dest)
        for name in CHECKSUMS:
            if name != FACENET_PATH.name:
                verify(BUFFALO_DIR / name)
        if not FACENET_PATH.is_file():
            FACENET_PATH.parent.mkdir(parents=True, exist_ok=True)
            download(FACENET_URL, FACENET_PATH)
        verify(FACENET_PATH)
        if not MOBILE_PATH.is_file():
            mobile_archive = tmp / "buffalo_sc.zip"
            download(MOBILE_URL, mobile_archive)
            expected_archive = MANIFEST["mobilefacenet"]["archive_sha256"]
            if sha256(mobile_archive) != expected_archive:
                raise RuntimeError("buffalo_sc archive checksum mismatch")
            with zipfile.ZipFile(mobile_archive) as package:
                matches = [name for name in package.namelist() if Path(name).name == MOBILE_PATH.name]
                if len(matches) != 1:
                    raise RuntimeError("buffalo_sc archive must contain one w600k_mbf.onnx")
                MOBILE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with package.open(matches[0]) as source, MOBILE_PATH.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
        verify_manifest(MOBILE_PATH, "mobilefacenet")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept-upstream-research-terms", action="store_true")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="check local assets without downloading")
    args = parser.parse_args()
    if not args.accept_upstream_research_terms and not args.verify_only:
        parser.error("pass --accept-upstream-research-terms after reviewing docs/REPRODUCIBILITY.md")
    if not args.verify_only:
        if not args.models_only:
            install_lfw()
        install_models()
    if not args.models_only and (not LFW_IMAGES.is_dir() or sum(1 for p in LFW_IMAGES.rglob("*.jpg")) != LFW_COUNT):
        raise RuntimeError("LFW is absent or incomplete")
    if not args.models_only:
        verify_lfw()
    for name in CHECKSUMS:
        verify(FACENET_PATH if name == FACENET_PATH.name else BUFFALO_DIR / name)
    for model, path in (("mobilefacenet", MOBILE_PATH), ("adaface", ADAFACE_PATH), ("magface", MAGFACE_PATH)):
        verify_manifest(path, model)
    for model, relative in (("adaface", "models/third_party/adaface/net.py"), ("magface", "models/third_party/magface/iresnet.py")):
        path = ROOT / relative
        if not path.is_file() or sha256(path) != MANIFEST[model]["model_source_sha256"]:
            raise RuntimeError(f"missing or checksum-mismatched {model} author source: {path}")
    print("All five model weights verified. Run: python scripts/check_research_environment.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
