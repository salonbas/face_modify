#!/usr/bin/env python3
"""Two-person MI-FGSM+Mask momentum ablation; output isolated from formal v3."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from advface.attacks.mi_fgsm import run_mi_fgsm
from advface.config import DEFAULT_DET_SIZE
from advface.constraints import LandmarkSuperpixelMask
from advface.image_io import load_bgr
from advface.models.insightface_app import create_face_app
from run_lfw_two_phase_v4 import FIELDS, evaluate, pairs

OUT = ROOT / "results/transfer/lfw_momentum_mask_v2"
ALLOWED = ("Michelle_Collins", "Meryl_Streep")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", choices=ALLOWED, required=True)
    parser.add_argument("--momentum", type=float, choices=(0.0, 0.5), required=True)
    args = parser.parse_args()
    pair = next(p for p in pairs() if p["identity_id"] == args.identity)
    out = OUT / f"mu_{str(args.momentum).replace('.', '_')}"
    record = out / "generation_records" / f"{args.identity}.json"
    target = out / "adversarial_probes" / f"{args.identity}_B_adv.png"
    if record.exists() or target.exists():
        raise SystemExit(f"refusing to overwrite: {record} / {target}")
    probe = load_bgr(ROOT / pair["probe_image"])
    app = create_face_app(det_size=DEFAULT_DET_SIZE)
    mask, _, _ = LandmarkSuperpixelMask().build(probe)
    start = time.perf_counter()
    adv, meta = run_mi_fgsm(probe, 0.04, app, steps=200, alpha=0.04 / 200, momentum=args.momentum,
                            precomputed_mask=mask, return_metadata=True)
    runtime = time.perf_counter() - start
    if meta["linf_tensor"] > 0.040001:
        raise RuntimeError(f"tensor L-inf violation: {meta['linf_tensor']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), adv):
        raise RuntimeError("PNG write failed")
    saved = load_bgr(target)
    outside = int(abs(saved.astype("int16") - probe.astype("int16")).max(axis=2)[mask == 0].max(initial=0))
    if outside:
        raise RuntimeError(f"outside mask delta: {outside}")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"identity_id": args.identity, "attack": "mi_fgsm", "constraint": "landmark_superpixel_mask",
                                  "momentum": args.momentum, "reference_image": pair["reference_image"], "probe_image": pair["probe_image"],
                                  "adversarial_image": str(target.relative_to(ROOT)), "configured_epsilon": 0.04, "steps": 200,
                                  "alpha": 0.04 / 200, "tv_weight": 0.0, "actual_linf_tensor": meta["linf_tensor"],
                                  "actual_linf_serialized": meta["linf_serialized"], "mask_outside_delta_max": outside,
                                  "runtime_seconds": runtime, "valid": True}, indent=2, sort_keys=True) + "\n")
    evaluate(out, "mi_fgsm", [pair])
    records = [json.loads(path.read_text()) for path in sorted((out / "evaluation_records").glob("*.json"))]
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    result = json.loads((out / "evaluation_records" / f"{args.identity}.json").read_text())
    print(json.dumps({"identity": args.identity, "momentum": args.momentum, "runtime_seconds": runtime,
                      "arcface_cosine": result["arcface_adv_cosine"], "arcface_success": result["arcface_threshold_crossing"],
                      "facenet_cosine": result["facenet_adv_cosine"], "facenet_success": result["facenet_transfer_crossing"],
                      "lpips": result["lpips"], "serialized_linf": result["actual_linf_serialized"]}, indent=2))


if __name__ == "__main__":
    main()
