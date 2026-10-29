#!/usr/bin/env python3
"""Read-only time-budget gate for the proposed LFW v2 five-model benchmark.

The historical v3 runtime field measures attack.apply only. It omits model
initialization, evaluation, perceptual metrics, and I/O, so its projection is
an attack-only floor, never an end-to-end estimate or permission to run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/transfer/lfw_attack_dev_20_four_method_v3/per_identity_results.csv"
METHODS = {"PGD", "MI-FGSM", "PGD+Mask", "MI-FGSM+Mask"}
OUT = ROOT / "results/transfer/lfw_dev_100_five_model_v1/preflight.json"


def read_source(path: Path, rows: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, object]]:
    if len(rows) != 20 or len({r["identity_id"] for r in rows}) != 20:
        raise ValueError(f"expected 20 distinct identities: {path}")
    if any(r["valid"].lower() != "true" or int(r["steps"]) != 200 or
           float(r["configured_epsilon"]) != 0.04 for r in rows):
        raise ValueError(f"formal attack contract failed: {path}")
    runtimes = {r["identity_id"]: float(r["runtime_seconds"]) for r in rows}
    if any(not 0 < seconds < 3600 for seconds in runtimes.values()):
        raise ValueError(f"invalid runtime: {path}")
    values = list(runtimes.values())
    return runtimes, {
        "n": len(values),
        "mean_attack_seconds": statistics.mean(values),
        "median_attack_seconds": statistics.median(values),
        "min_attack_seconds": min(values),
        "max_attack_seconds": max(values),
    }


def main() -> int:
    split = ROOT / "data/datasets/lfw/evaluation_splits/v2/development.csv"
    with split.open(newline="", encoding="utf-8") as handle:
        dev = list(csv.DictReader(handle))
    if len(dev) != 100 or len({r["identity_id"] for r in dev}) != 100 or any(r["split"] != "development" for r in dev):
        raise ValueError("LFW v2 Development must contain 100 distinct identities")
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 80 or {r["method"] for r in source_rows} != METHODS:
        raise ValueError("formal v3 result must have 80 rows across exactly four methods")
    timings, provenance = {}, {}
    for method in sorted(METHODS):
        timings[method], provenance[method] = read_source(SOURCE, [r for r in source_rows if r["method"] == method])
    common = set.intersection(*(set(values) for values in timings.values()))
    if len(common) != 20 or not common <= {r["identity_id"] for r in dev}:
        raise ValueError("historical 20-person pilot is not a subset of v2 Development")
    combined = sorted(sum(values[ident] for values in timings.values()) for ident in common)
    attack_only_mean_minutes = 100 * statistics.mean(combined) / 60
    attack_only_best_observed_minutes = 100 * combined[0] / 60
    # Conservative upper planning figure for the measured attack component.
    attack_only_p95_minutes = 100 * combined[18] / 60
    result = {
        "protocol": "LFW v2 Development 100; five-model benchmark preflight v1",
        "decision": "HARD_STOP",
        "reason": "Historical attack generation alone exceeds the 60-minute total budget; new victim checkpoints and end-to-end timing are unavailable.",
        "historical_attack_component": {
            "n_paired_identities": len(common),
            "per_identity_mean_seconds": statistics.mean(combined),
            "per_identity_min_seconds": combined[0],
            "per_identity_p95_seconds": combined[18],
            "projected_100_mean_minutes": attack_only_mean_minutes,
            "projected_100_best_observed_minutes": attack_only_best_observed_minutes,
            "projected_100_p95_minutes": attack_only_p95_minutes,
            "measurement_scope": "attack.apply runtime_seconds only; excludes initialization, detection outside apply, new victims, quality metrics, and serialization",
        },
        "complete_benchmark_estimate_minutes": None,
        "missing_stage_3_measurements": ["new victim inference", "model initialization", "mask generation separated", "SSIM/LPIPS/DISTS separated", "I/O and serialization", "peak memory", "end-to-end wall time"],
        "hardware_at_preflight": {
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "torch_version": torch.__version__, "torch_cpu_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "development_manifest": {"path": str(split.relative_to(ROOT)), "sha256": hashlib.sha256(split.read_bytes()).hexdigest(), "identities": len(dev)},
        "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "method_timings": provenance,
        "model_checkpoint_status": {name: (ROOT / path).is_file() for name, path in {
            "AdaFace": "models/adaface/adaface_ir50_ms1mv2.ckpt",
            "MobileFaceNet": "models/insightface/buffalo_sc/w600k_mbf.onnx",
            "MagFace": "models/magface/magface_iresnet100_ms1mv2.pth",
        }.items()},
        "full_run_authorized_by_gate": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"HARD STOP: attack-only mean {attack_only_mean_minutes:.1f} min; best observed {attack_only_best_observed_minutes:.1f} min; p95 {attack_only_p95_minutes:.1f} min")
    print(f"Evidence: {OUT}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
