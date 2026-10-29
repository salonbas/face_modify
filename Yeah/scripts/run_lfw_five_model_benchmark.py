#!/usr/bin/env python3
"""Fail-closed entry point for the proposed LFW v2 five-model benchmark.

The formal executor is intentionally unavailable until all readiness gates pass.
This script never creates adversarial images or reads OOS image data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/datasets/lfw/evaluation_splits/v2"
MODELS = {
    "arcface": ROOT / "models/insightface/buffalo_l/w600k_r50.onnx",
    "facenet": ROOT / "models/facenet/20180402-114759-vggface2.pt",
    "adaface": ROOT / "models/adaface/adaface_ir50_ms1mv2.ckpt",
    "mobilefacenet": ROOT / "models/insightface/buffalo_sc/w600k_mbf.onnx",
    "magface": ROOT / "models/magface/magface_iresnet100_ms1mv2.pth",
}
CALIBRATION = {
    "arcface": ROOT / "results/calibration/arcface_lfw_v2/thresholds.json",
    "facenet": ROOT / "results/calibration/facenet_lfw_v2/thresholds.json",
    "adaface": ROOT / "results/calibration/adaface_lfw_v2/thresholds.json",
    "mobilefacenet": ROOT / "results/calibration/mobilefacenet_lfw_v2/thresholds.json",
    "magface": ROOT / "results/calibration/magface_lfw_v2/thresholds.json",
}
CONTRACT = {"epsilon": 0.04, "alpha": 0.0002, "steps": 200,
            "momentum": 1.0, "methods": ["PGD", "MI-FGSM", "PGD+Mask", "MI-FGSM+Mask"]}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def preflight() -> dict:
    problems: list[str] = []
    evidence: dict = {"attack_contract": CONTRACT, "models": {}, "calibration": {}}
    try:
        dev, oos = rows(SPLITS / "development.csv"), rows(SPLITS / "oos.csv")
        dev_ids = {r["identity_id"] for r in dev}
        oos_ids = {r["identity_id"] for r in oos}
        if len(dev) != 100 or len(dev_ids) != 100 or any(r["split"] != "development" for r in dev):
            problems.append("Development manifest is not exactly 100 distinct development identities")
        if len(oos) != 228 or len(oos_ids) != 228 or dev_ids & oos_ids:
            problems.append("OOS manifest count or isolation failed")
        for r in dev:
            for kind in ("reference", "probe"):
                image = ROOT / r[f"{kind}_image"]
                if not image.is_file() or image.parent.name != r["identity_id"]:
                    problems.append(f"missing or wrongly paired Development image: {image}")
        for split in ("dev_train", "dev_test"):
            pairs = rows(SPLITS / "calibration_pairs" / f"{split}.csv")
            if not pairs or any(r["identity_a"] in dev_ids | oos_ids or
                                r["identity_b"] in dev_ids | oos_ids for r in pairs):
                problems.append(f"calibration {split} overlaps attack identities or is empty")
            evidence[f"calibration_{split}_pairs"] = len(pairs)
        evidence["development_count"] = len(dev)
        evidence["oos_manifest_count_only"] = len(oos)
        evidence["development_sha256"] = hashlib.sha256((SPLITS / "development.csv").read_bytes()).hexdigest()
    except (OSError, KeyError, ValueError) as exc:
        problems.append(f"manifest or calibration isolation check failed: {exc}")
    for name, path in MODELS.items():
        present = path.is_file()
        evidence["models"][name] = {"path": str(path.relative_to(ROOT)), "present": present}
        if not present:
            problems.append(f"missing {name} checkpoint: {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        evidence["models"][name]["sha256"] = digest
        metadata = ROOT / "models/checkpoint_manifest.json"
        expected = json.loads(metadata.read_text()).get(name, {}).get("sha256") if metadata.is_file() else None
        if expected != digest:
            problems.append(f"{name} checkpoint checksum is not verified by models/checkpoint_manifest.json")
    for name, source in {
        "adaface": ROOT / "models/third_party/adaface/net.py",
        "magface": ROOT / "models/third_party/magface/iresnet.py",
    }.items():
        if not source.is_file():
            problems.append(f"missing pinned {name} author model source")
        else:
            manifest = ROOT / "models/checkpoint_manifest.json"
            expected = json.loads(manifest.read_text()).get(name, {}).get("model_source_sha256") if manifest.is_file() else None
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                problems.append(f"{name} author model source checksum mismatch")
    # The author repositories license source code, but no separate grant for
    # the hosted pretrained bytes was published alongside these checkpoints.
    problems.append("AdaFace and MagFace pretrained-weight usage terms require review before formal use")
    for name, path in CALIBRATION.items():
        if not path.is_file():
            problems.append(f"missing {name} Calibration v2 threshold")
            continue
        try:
            op = json.loads(path.read_text(encoding="utf-8"))["far_1e-2"]
            threshold = float(op["threshold"])
            if not -1 <= threshold <= 1 or op.get("support_status") != "CALIBRATED":
                raise ValueError("invalid threshold or calibration status")
            evidence["calibration"][name] = threshold
            if name in ("adaface", "mobilefacenet", "magface"):
                meta = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
                if meta.get("checkpoint_sha256") != evidence["models"].get(name, {}).get("sha256") or not meta.get("preprocessing_version"):
                    problems.append(f"{name} calibration checkpoint or preprocessing provenance mismatch")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            problems.append(f"invalid {name} calibration: {exc}")
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        available = next((int(line.split()[1]) * 1024 for line in meminfo.read_text().splitlines()
                          if line.startswith("MemAvailable:")), 0)
    else:
        # macOS does not expose SC_AVPHYS_PAGES. Unknown capacity is a
        # preflight problem, never an uncaught exception or a pass.
        try:
            available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        except (ValueError, OSError):
            available = 0
    evidence["available_memory_bytes"] = available
    if available < 4 * 1024**3:
        problems.append("less than 4 GiB available memory or memory could not be measured")
    output = ROOT / "results/transfer/lfw_dev_100_five_model_v1/formal"
    evidence["formal_output"] = str(output.relative_to(ROOT))
    if output.exists() and any(output.iterdir()):
        problems.append("formal output already exists; resume validation is not implemented")
    timing = ROOT / "results/transfer/lfw_dev_100_five_model_v1/preflight.json"
    if timing.is_file():
        historical = json.loads(timing.read_text(encoding="utf-8"))
        evidence["historical_attack_only_mean_minutes"] = historical.get("historical_attack_component", {}).get("projected_100_mean_minutes")
    cpu_name = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").is_file() else platform.processor()
    evidence["hardware"] = {"processor": cpu_name, "logical_cpus": os.cpu_count(),
                            "kernel": platform.release()}
    eligibility = ROOT / "results/transfer/lfw_dev_100_five_model_v1/clean_eligibility.json"
    if not eligibility.is_file():
        problems.append("Development 100 five-model clean eligibility is missing")
    else:
        try:
            value = json.loads(eligibility.read_text(encoding="utf-8"))
            if value["summary"]["original_denominator"] != 100 or value["manifest_sha256"] != evidence["development_sha256"]:
                problems.append("Development 100 clean eligibility manifest or denominator mismatch")
            if len(value["rows"]) != 100 or set(value["summary"]["models"]) != set(MODELS):
                problems.append("Development 100 clean eligibility is incomplete")
            if value["thresholds"] != evidence["calibration"]:
                problems.append("Development 100 eligibility thresholds differ from calibration")
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f"invalid clean eligibility artifact: {exc}")
    # A measured complete five-model estimate is required; history is only an
    # attack-only lower bound and cannot authorize execution.
    problems.append("no validated five-model end-to-end estimate on the WSL CPU")
    problems.append("formal five-model execute/resume implementation has not passed safety validation")
    return {"decision": "NOT READY", "problems": problems, "evidence": evidence}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = preflight()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.execute:
        print("Formal execute refused: readiness and safe executor gates are incomplete.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
