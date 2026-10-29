#!/usr/bin/env python3
"""Run a frozen LFW verification protocol selected by a versioned config.

The v2 five-model executor deliberately remains fail-closed; selecting it does
not turn an engineering preflight into a formal benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = {
    "lfw_dev_20_four_method_v3": "run_lfw_two_phase_v4.py",
    "lfw_dev_100_five_model_v2": "run_lfw_five_model_benchmark.py",
}


def checked_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    protocol = config.get("protocol")
    if protocol not in PROTOCOLS or config.get("runner") != PROTOCOLS[protocol]:
        raise ValueError("unknown protocol or mismatched frozen runner")
    split = ROOT / config["split"]
    if hashlib.sha256(split.read_bytes()).hexdigest() != config["split_sha256"]:
        raise ValueError(f"frozen split SHA-256 mismatch: {split}")
    for relative in config["thresholds"].values():
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(f"missing threshold artifact: {relative}")
    if config.get("steps") != 200 or config.get("epsilon") != 0.04:
        raise ValueError("config differs from frozen 200-step, epsilon=0.04 contract")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--attack", choices=("pgd_full", "mi_fgsm"))
    parser.add_argument("--phase", choices=("generate", "evaluate", "regression", "all"), default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = checked_config(args.config)
    runner = ROOT / "scripts" / config["runner"]
    if config["protocol"] == "lfw_dev_20_four_method_v3":
        if args.preflight or args.execute or not args.attack:
            parser.error("v3 requires --attack; select --phase for generation, evaluation or regression")
        command = [sys.executable, str(runner), "--attack", args.attack, "--phase", args.phase]
    else:
        if args.attack or args.phase != "all":
            parser.error("v2 five-model benchmark accepts --preflight or --execute only")
        command = [sys.executable, str(runner), "--execute" if args.execute else "--preflight"]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
