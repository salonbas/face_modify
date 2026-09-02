#!/usr/bin/env python3
"""Calibrate canonical FaceNet VGGFace2 verification thresholds on LFW pairs."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import calibrate_threshold


def main() -> int:
    sys.argv[1:1] = ["--model", "facenet_vggface2", "--run-name", "facenet_lfw_v1"]
    return calibrate_threshold.main()


if __name__ == "__main__":
    raise SystemExit(main())
