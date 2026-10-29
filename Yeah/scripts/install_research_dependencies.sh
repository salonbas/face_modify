#!/usr/bin/env bash
# Install the resolver-compatible Python 3.12 research environment.
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${1:-$root_dir/.venv/bin/python}"
"$python_bin" -m pip install 'torch==2.2.2+cpu' 'torchvision==0.17.2+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
"$python_bin" -m pip install -r "$root_dir/requirements.txt" -c "$root_dir/requirements-constraints.txt"
"$python_bin" -m pip install -e "$root_dir"
"$python_bin" -m pip check
"$python_bin" -c 'import facenet_pytorch, numpy, torch, torchvision, insightface, cv2, onnxruntime; print("dependencies import:", numpy.__version__, torch.__version__)'
