#!/usr/bin/env python3
"""Diagnostic-only trajectories for the frozen LFW 20-identity protocol.

This never writes below the formal benchmark directory.  The attack API's
``diagnostics`` switch only observes detached tensors after the established
update values are computed; it is off by default in all production callers.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from advface.attacks.mi_fgsm import run_mi_fgsm
from advface.attacks.pgd import run_pgd_full
from advface.config import DEFAULT_DET_SIZE
from advface.constraints import LandmarkSuperpixelMask
from advface.image_io import load_bgr
from advface.models.insightface_app import create_face_app

EPS, STEPS = .04, 200
OUT = ROOT / "results/diagnostics/lfw_gradient_trajectory_v1"
METHODS = {"PGD": ("pgd", False), "MI-FGSM": ("mi", False), "PGD+Mask": ("pgd", True), "MI-FGSM+Mask": ("mi", True)}


def pairs():
    with (ROOT / "data/datasets/lfw/evaluation_splits/attack_dev.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 20 or any(x["split"] != "attack_dev" for x in rows):
        raise RuntimeError("frozen attack_dev contract failed")
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)


def attack(image, app, method: str):
    kind, masked = METHODS[method]
    constraint = LandmarkSuperpixelMask() if masked else None
    if kind == "pgd":
        result = run_pgd_full(image, [EPS], app, steps=STEPS, constraint=constraint, tv_weight=0., diagnostics=True)[0]
        return result.attacked_bgr, result.diagnostics, result.final_mask
    adv, meta = run_mi_fgsm(image, EPS, app, steps=STEPS, alpha=EPS/STEPS, momentum=1., constraint=constraint, tv_weight=0., return_metadata=True, diagnostics=True)
    return adv, meta["diagnostics"], meta.get("final_mask")


def regression(rows: list[dict]):
    """Exact ON/OFF byte comparison; downstream metrics are deterministic functions of B_adv."""
    selected = [r for r in rows if r["identity_id"] in {"Michelle_Collins", "Meryl_Streep"}]
    report = []
    app = create_face_app(det_size=DEFAULT_DET_SIZE)
    for row in selected:
        image = load_bgr(ROOT / row["probe_image"])
        for method, (kind, masked) in METHODS.items():
            constraint = LandmarkSuperpixelMask() if masked else None
            if kind == "pgd":
                off = run_pgd_full(image, [EPS], app, steps=STEPS, constraint=constraint, tv_weight=0., diagnostics=False)[0].attacked_bgr
            else:
                off = run_mi_fgsm(image, EPS, app, steps=STEPS, alpha=EPS/STEPS, momentum=1., constraint=constraint, tv_weight=0., return_metadata=False, diagnostics=False)
            on, diag, _ = attack(image, app, method)
            report.append({"identity_id": row["identity_id"], "method": method, "serialized_b_adv_exact": bool(np.array_equal(off, on)), "max_abs_rgb_difference": int(np.abs(off.astype(np.int16)-on.astype(np.int16)).max()), "trajectory_steps": len(diag["trajectory"]), "pass": bool(np.array_equal(off, on) and len(diag["trajectory"]) == STEPS)})
    write_csv(OUT / "regression_smoke.csv", report)
    (OUT / "regression_smoke.json").write_text(json.dumps({"setting": {"epsilon": EPS, "steps": STEPS, "tv_weight": 0.}, "results": report, "all_pass": all(x["pass"] for x in report)}, indent=2) + "\n", encoding="utf-8")
    if not all(x["pass"] for x in report):
        raise RuntimeError("diagnostic ON/OFF regression failed")


def full(rows: list[dict], only: str | None):
    requested = [r for r in rows if only is None or r["identity_id"] == only]
    if only and not requested: raise RuntimeError("identity not in frozen split")
    all_steps, summaries = [], []
    for row in requested:
        image = load_bgr(ROOT / row["probe_image"])
        app = create_face_app(det_size=DEFAULT_DET_SIZE)
        for method in METHODS:
            adv, diag, mask = attack(image, app, method)
            method_dir = OUT / "per_identity" / row["identity_id"] / method.replace("+", "_mask")
            method_dir.mkdir(parents=True, exist_ok=True)
            if mask is not None:
                cv2.imwrite(str(method_dir / "final_mask.png"), (mask * 255).astype(np.uint8))
                np.save(method_dir / "final_mask.npy", mask.astype(np.float32))
            trajectory = diag["trajectory"]
            for x in trajectory: x.update(identity_id=row["identity_id"], method=method)
            write_csv(method_dir / "trajectory.csv", trajectory)
            all_steps.extend(trajectory)
            summaries.append({"identity_id": row["identity_id"], "method": method, "steps": len(trajectory), "final_arcface_cosine": trajectory[-1]["arcface_cosine"], "final_linf": trajectory[-1]["current_linf"], "operation_order": diag["operation_order"], "final_mask_persisted": mask is not None})
    write_csv(OUT / "trajectory_all.csv", all_steps)
    write_csv(OUT / "diagnostic_summary.csv", summaries)
    (OUT / "provenance.json").write_text(json.dumps({"kind": "diagnostic / mechanism analysis", "formal_benchmark_replaced": False, "frozen_split": "attack_dev.csv", "identities": [r["identity_id"] for r in requested], "methods": list(METHODS), "epsilon": EPS, "steps": STEPS, "tv_weight": 0.}, indent=2) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["regression", "full"], required=True); p.add_argument("--identity")
    args = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    rows = pairs()
    if args.mode == "regression": regression(rows)
    else: full(rows, args.identity)

if __name__ == "__main__": main()
