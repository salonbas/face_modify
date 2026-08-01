#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "找不到 .venv，請先建立 Fawkes Python 環境。"
  exit 1
fi

if [ ! -d "photos" ]; then
  mkdir -p "photos"
fi

".venv/bin/python" run_protection.py --directory ./photos/ --mode low
