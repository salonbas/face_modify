#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

INPUT_DIR="photos"
OUTPUT_DIR="results/fgsm"

if [ ! -d ".venv" ]; then
  echo "找不到 .venv，請先建立 fgsm Python 環境。"
  exit 1
fi

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

shopt -s nullglob nocaseglob
images=(
  "$INPUT_DIR"/*.jpg
  "$INPUT_DIR"/*.jpeg
  "$INPUT_DIR"/*.png
  "$INPUT_DIR"/*.webp
)
shopt -u nocaseglob

if [ "${#images[@]}" -eq 0 ]; then
  echo "找不到圖片，請先把 jpg/jpeg/png/webp 放到 $INPUT_DIR/。"
  exit 0
fi

echo "找到 ${#images[@]} 張圖片，開始產生 FGSM 對抗樣本..."
".venv/bin/python" insightface_fgsm_sensitivity_cpu.py "${images[@]}" --output-dir "$OUTPUT_DIR" "$@"
