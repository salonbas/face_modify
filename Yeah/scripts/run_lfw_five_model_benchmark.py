#!/usr/bin/env python3
"""Offline, resumable LFW v2 Development five-model benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import subprocess
import warnings
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
warnings.filterwarnings("ignore", message=r"`rcond` parameter will change to the default of machine precision",
                        category=FutureWarning, module=r"insightface\.utils\.transform")
warnings.filterwarnings("ignore", message=r"`estimate` is deprecated since version 0\.26",
                        category=FutureWarning, module=r"insightface\.utils\.face_align")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
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
    dev_ids: set[str] = set()
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
    for relative, expected in {
        "models/insightface/buffalo_l/det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
        "models/insightface/buffalo_l/1k3d68.onnx": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
        "models/insightface/buffalo_sc/det_500m.onnx": json.loads((ROOT / "models/checkpoint_manifest.json").read_text())["mobilefacenet"]["detector_sha256"],
    }.items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f"missing or invalid detector/alignment asset: {path}")
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
        if not available and sys.platform == "darwin":
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False)
            if vm.returncode == 0:
                import re
                page = re.search(r"page size of (\d+) bytes", vm.stdout)
                counts = [int(re.search(r"(\d+)", line.split(":", 1)[1]).group(1))
                          for line in vm.stdout.splitlines() if line.startswith(("Pages free:", "Pages inactive:", "Pages speculative:"))]
                available = int(page.group(1)) * sum(counts) if page else 0
    evidence["available_memory_bytes"] = available
    if available < 4 * 1024**3:
        problems.append("less than 4 GiB available memory or memory could not be measured")
    output = ROOT / "results/transfer/lfw_dev_100_five_model_v1/formal"
    evidence["formal_output"] = str(output.relative_to(ROOT))
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
            if value["summary"]["original_denominator"] != 100 or value["manifest_sha256"] != evidence.get("development_sha256"):
                problems.append("Development 100 clean eligibility manifest or denominator mismatch")
            if len(value["rows"]) != 100 or set(value["summary"]["models"]) != set(MODELS):
                problems.append("Development 100 clean eligibility is incomplete")
            if {r["identity_id"] for r in value["rows"]} != dev_ids:
                problems.append("Development 100 clean eligibility identities differ from split")
            if value["thresholds"] != evidence["calibration"]:
                problems.append("Development 100 eligibility thresholds differ from calibration")
            if value.get("checkpoint_sha256") != {name: item.get("sha256") for name, item in evidence["models"].items()}:
                problems.append("Development 100 eligibility checkpoint provenance differs from current models")
            import importlib.metadata
            versions = {name: importlib.metadata.version(name) for name in
                        ("torch", "numpy", "opencv-python", "onnxruntime", "insightface", "facenet-pytorch")}
            if value.get("environment") != versions:
                problems.append("Development 100 eligibility environment differs from current environment")
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f"invalid clean eligibility artifact: {exc}")
    for module in ("torch", "onnxruntime", "insightface", "facenet_pytorch", "skimage", "cv2"):
        try:
            __import__(module)
        except (ImportError, OSError) as exc:
            problems.append(f"dependency unavailable: {module}: {exc}")
    try:
        import torch
        cache = Path(torch.hub.get_dir()) / "checkpoints"
        evidence["perceptual_weights"] = {}
        for filename, expected in {
            "alexnet-owt-7be5be79.pth": "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
            "vgg16-397923af.pth": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
        }.items():
            path = cache / filename
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            evidence["perceptual_weights"][filename] = {"path": str(path), "sha256": actual}
            if actual != expected:
                problems.append(f"missing or invalid offline perceptual weight: {path}")
    except (ImportError, OSError) as exc:
        problems.append(f"perceptual weight verification failed: {exc}")
    dependency_check = subprocess.run([sys.executable, "-m", "pip", "check"],
                                      capture_output=True, text=True, check=False)
    evidence["pip_check"] = dependency_check.stdout.strip() or dependency_check.stderr.strip()
    if dependency_check.returncode:
        problems.append("installed dependency versions conflict; see evidence.pip_check")
    git_check = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True, check=False)
    evidence["tracked_worktree_clean"] = git_check.returncode == 0 and not git_check.stdout.strip()
    if not evidence["tracked_worktree_clean"]:
        problems.append("tracked source files are modified; commit before the formal run")
    return {"decision": "READY" if not problems else "NOT READY", "problems": problems, "evidence": evidence}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--identity", help="Development identity; engineering acceptance only")
    parser.add_argument("--method", choices=CONTRACT["methods"])
    parser.add_argument("--reevaluate", action="store_true", help="recompute victim scores from saved PNGs")
    args = parser.parse_args()
    if args.preflight:
        report = preflight()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["decision"] == "READY" else 2
    from formal_five_model import run
    return run(args, preflight)


if __name__ == "__main__":
    raise SystemExit(main())
