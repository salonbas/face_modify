#!/usr/bin/env bash
# Install pinned author model definitions and AdaFace's MTCNN alignment assets.
# Model weights are downloaded separately from the links in checkpoint_manifest.
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root_dir/models/third_party"
mkdir -p "$dest"
if [[ ! -f "$dest/adaface/net.py" ]]; then
  tmp_adaface="$(mktemp -d)"
  git clone https://github.com/mk-minchul/AdaFace.git "$tmp_adaface/source"
  git -C "$tmp_adaface/source" checkout c60eaa786a42c03444f3df7096dbaf9d57ae010d
  mkdir -p "$dest/adaface"
  cp "$tmp_adaface/source/net.py" "$dest/adaface/"
  cp -R "$tmp_adaface/source/face_alignment" "$dest/adaface/"
  rm -rf "$tmp_adaface"
fi
if [[ ! -f "$dest/magface/iresnet.py" ]]; then
  tmp_magface="$(mktemp -d)"
  git clone https://github.com/IrvingMeng/MagFace.git "$tmp_magface/source"
  git -C "$tmp_magface/source" checkout 99bae614ac2643b9694bf18e0c5645272ff6acfa
  mkdir -p "$dest/magface"
  cp "$tmp_magface/source/models/iresnet.py" "$dest/magface/"
  rm -rf "$tmp_magface"
fi
echo "Pinned AdaFace and MagFace model sources are present in $dest"
