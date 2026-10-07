#!/usr/bin/env python3
"""Install the fixed runtime assets from their upstream publishers.

This script deliberately does not use or copy a user's caches.  It installs
the LFW images and the exact model files under this checkout, validates every
model checksum, and leaves them ignored by Git pending redistribution approval.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LFW_IMAGES = ROOT / "data/datasets/lfw/images"
BUFFALO_DIR = ROOT / "models/insightface/buffalo_l"
FACENET_PATH = ROOT / "models/facenet/20180402-114759-vggface2.pt"
LFW_COUNT = 13_233
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
FACENET_URL = "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"
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


def install_lfw() -> None:
    if LFW_IMAGES.is_dir() and sum(1 for p in LFW_IMAGES.rglob("*.jpg")) == LFW_COUNT:
        print(f"LFW already complete: {LFW_IMAGES}")
        return
    # sklearn owns the official LFW fetch URL and archive layout.  Its cache is
    # deliberately rooted inside this checkout, then exposed at the canonical
    # experiment path without referring to any user-global cache.
    from sklearn.datasets import fetch_lfw_people

    data_home = ROOT / ".asset-downloads"
    fetch_lfw_people(data_home=data_home, funneled=True, download_if_missing=True)
    source = data_home / "lfw_home/lfw_funneled"
    if not source.is_dir() or sum(1 for p in source.rglob("*.jpg")) != LFW_COUNT:
        raise RuntimeError(f"upstream LFW download incomplete: {source}")
    LFW_IMAGES.parent.mkdir(parents=True, exist_ok=True)
    if LFW_IMAGES.exists() or LFW_IMAGES.is_symlink():
        if LFW_IMAGES.is_dir() and not LFW_IMAGES.is_symlink():
            shutil.rmtree(LFW_IMAGES)
        else:
            LFW_IMAGES.unlink()
    shutil.move(str(source), str(LFW_IMAGES))
    shutil.rmtree(data_home, ignore_errors=True)


def install_models() -> None:
    with tempfile.TemporaryDirectory(prefix="advface-assets-") as raw_tmp:
        tmp = Path(raw_tmp)
        archive = tmp / "buffalo_l.zip"
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
                    verify(BUFFALO_DIR / name)
        FACENET_PATH.parent.mkdir(parents=True, exist_ok=True)
        download(FACENET_URL, FACENET_PATH)
        verify(FACENET_PATH)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept-upstream-research-terms", action="store_true")
    parser.add_argument("--models-only", action="store_true")
    args = parser.parse_args()
    if not args.accept_upstream_research_terms:
        parser.error("pass --accept-upstream-research-terms after reviewing docs/REPRODUCIBILITY.md")
    if not args.models_only:
        install_lfw()
    install_models()
    print("Installed assets. Run: python scripts/check_research_environment.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
