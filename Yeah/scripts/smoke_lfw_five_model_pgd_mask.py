#!/usr/bin/env python3
"""At most two Development identities, one formal 200-step PGD+Mask method."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from advface.attacks.registry import ATTACKS, AttackConfig
from advface.constraints import LandmarkSuperpixelMask
from advface.evaluation.multi_victim import MODELS, load_v2_thresholds, summarize
from advface.evaluation.similarity import cosine_similarity
from advface.image_io import load_bgr
from advface.models import load_embedder
from scripts.run_lfw_dev_v2 import metrics_from_png

OUT = ROOT / "results/smoke/lfw_five_model_pgd_mask_v1"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def run(limit: int) -> dict:
    manifest = ROOT / "data/datasets/lfw/evaluation_splits/v2/development.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))[:limit]
    thresholds = load_v2_thresholds(ROOT)
    started = time.perf_counter()
    models = {}
    init_by_model = {}
    for name in MODELS:
        t = time.perf_counter()
        models[name] = load_embedder(name)
        init_by_model[name] = time.perf_counter() - t
    init_seconds = time.perf_counter() - started
    records = []
    for pair in pairs:
        ident = pair["identity_id"]
        record_path = OUT / "records" / f"{ident}.json"
        if record_path.is_file():
            existing = json.loads(record_path.read_text())
            artifact = ROOT / existing["adversarial_image"]
            if existing["valid"] and artifact.is_file() and hashlib.sha256(artifact.read_bytes()).hexdigest() == existing["artifact_sha256"]:
                records.append(existing)
                continue
            raise RuntimeError(f"invalid existing record; refusing overwrite: {record_path}")
        a = load_bgr(ROOT / pair["reference_image"])
        b = load_bgr(ROOT / pair["probe_image"])
        t = time.perf_counter()
        attack = ATTACKS["pgd_full"].apply(b, app=models["arcface"].app,
                   config=AttackConfig(name="pgd_full", eps=0.04, steps=200,
                                       extra={"constraint": "landmark_superpixel_mask", "tv_weight": 0.0}))
        attack_seconds = time.perf_counter() - t
        if attack.parameters["linf_tensor"] > 0.040001:
            raise RuntimeError(f"tensor L-infinity exceeded: {ident}")
        target = OUT / "adversarial_probes" / f"{ident}_B_adv_pgd_mask.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError(f"unverified artifact exists: {target}")
        temp = target.with_name(target.stem + ".tmp.png")
        t = time.perf_counter()
        if not cv2.imwrite(str(temp), attack.adversarial_bgr):
            raise RuntimeError(f"PNG serialization failed: {ident}")
        os.replace(temp, target)
        adv = load_bgr(target)
        serialization_seconds = time.perf_counter() - t
        t = time.perf_counter()
        actual_linf, ssim, lpips, dists = metrics_from_png(ROOT / pair["probe_image"], target)
        metrics_seconds = time.perf_counter() - t
        t = time.perf_counter()
        mask = LandmarkSuperpixelMask().build(b)[0]
        mask_validation_seconds = time.perf_counter() - t
        outside = int(np.abs(adv.astype(np.int16) - b.astype(np.int16)).max(axis=2)[mask == 0].max(initial=0))
        if outside or actual_linf > 0.04 + 1 / 255 or not all(math.isfinite(x) for x in (ssim, lpips, dists)):
            raise RuntimeError(f"serialization or metric contract failed: {ident}")
        scores = {}
        inference_seconds = {}
        for name, model in models.items():
            t = time.perf_counter()
            ar = model.get_embedding(a, f"{ident}:A")
            br = model.get_embedding(b, f"{ident}:B")
            xr = model.get_embedding(adv, f"{ident}:B_adv")
            clean, attacked = float(cosine_similarity(ar, br)), float(cosine_similarity(ar, xr))
            eligible = clean >= thresholds[name]
            scores[name] = {"clean_cosine": clean, "adv_cosine": attacked,
                            "eligible": eligible, "success": eligible and attacked < thresholds[name]}
            inference_seconds[name] = time.perf_counter() - t
        record = {"identity_id": ident, "reference_image": pair["reference_image"],
                  "probe_image": pair["probe_image"], "adversarial_image": str(target.relative_to(ROOT)),
                  "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                  "attack": "PGD+Mask", "steps": 200, "epsilon": 0.04, "alpha": 0.0002,
                  "linf_tensor": attack.parameters["linf_tensor"], "linf_serialized": actual_linf,
                  "ssim": ssim, "lpips": lpips, "dists": dists,
                  "mask_outside_delta_max": outside, "attack_seconds": attack_seconds,
                  "serialization_seconds": serialization_seconds,
                  "perceptual_metrics_seconds": metrics_seconds,
                  "mask_validation_seconds": mask_validation_seconds,
                  "inference_seconds": inference_seconds,
                  "models": scores, "valid": True}
        atomic_json(record_path, record)
        records.append(record)
        print(f"{ident}: 200-step PGD+Mask completed in {attack_seconds:.2f} s", flush=True)
    report = {"scope": f"first {limit} Development identities; one attack only",
              "model_initialization_seconds": init_seconds, "elapsed_seconds": time.perf_counter() - started,
              "model_initialization_by_model_seconds": init_by_model,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "summary": summarize(records, thresholds), "records": records}
    atomic_json(OUT / f"summary_{limit}.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-identities", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    result = run(args.max_identities)
    print(json.dumps({"elapsed_seconds": result["elapsed_seconds"], "summary": result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
