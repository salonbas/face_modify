#!/usr/bin/env python3
"""Diagnostic-only trajectories for the frozen LFW 20-identity protocol.

This never writes below the formal benchmark directory.  The attack API's
``diagnostics`` switch only observes detached tensors after the established
update values are computed; it is off by default in all production callers.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
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
from advface.evaluation.similarity import cosine_similarity
from advface.evaluation.verification import PRIMARY_OPERATING_POINT, load_thresholds
from advface.image_io import load_bgr
from advface.models import load_embedder
from advface.models.insightface_app import create_face_app, get_embedding_from_bgr

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


def attack(image, app, method: str, precomputed_mask: np.ndarray | None = None):
    kind, masked = METHODS[method]
    constraint = LandmarkSuperpixelMask() if masked else None
    if kind == "pgd":
        result = run_pgd_full(image, [EPS], app, steps=STEPS, constraint=constraint, tv_weight=0., diagnostics=True, precomputed_mask=precomputed_mask)[0]
        return result.attacked_bgr, result.diagnostics, result.final_mask
    adv, meta = run_mi_fgsm(image, EPS, app, steps=STEPS, alpha=EPS/STEPS, momentum=1., constraint=constraint, tv_weight=0., return_metadata=True, diagnostics=True, precomputed_mask=precomputed_mask)
    return adv, meta["diagnostics"], meta.get("final_mask")


def png_hash(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG serialization failed")
    return hashlib.sha256(encoded.tobytes()).hexdigest()


def pair_metrics(reference: np.ndarray, probe: np.ndarray, adversarial: np.ndarray, arcface, facenet, source_threshold: float, victim_threshold: float) -> dict:
    ar, ab, ax = (get_embedding_from_bgr(arcface, image, label) for image, label in ((reference, "A"), (probe, "B"), (adversarial, "B_adv")))
    vr, vb, vx = (facenet.get_embedding(image, label=label) for image, label in ((reference, "A"), (probe, "B"), (adversarial, "B_adv")))
    ac, aa = float(cosine_similarity(ar, ab)), float(cosine_similarity(ar, ax))
    vc, va = float(cosine_similarity(vr, vb)), float(cosine_similarity(vr, vx))
    return {"arcface_clean_cosine": ac, "arcface_cosine": aa, "arcface_threshold_crossing": ac >= source_threshold and aa < source_threshold, "facenet_clean_cosine": vc, "facenet_cosine": va, "facenet_threshold_crossing": vc >= victim_threshold and va < victim_threshold, "linf_serialized": float(np.abs(adversarial.astype(np.int16) - probe.astype(np.int16)).max() / 255.)}


def evaluate_checkpoints(row: dict, method: str, checkpoints: list[tuple[int, np.ndarray]], checkpoint_dir: Path, arcface, facenet, embedding_cache: dict) -> list[dict]:
    """Evaluate saved snapshots only after the surrogate attack has finished."""
    source_thresholds, _ = load_thresholds(ROOT / "results/calibration/arcface_lfw_v1")
    victim_thresholds, _ = load_thresholds(ROOT / "results/calibration/facenet_lfw_v1")
    reference, probe = load_bgr(ROOT / row["reference_image"]), load_bgr(ROOT / row["probe_image"])
    output = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    decoded = []
    paths = []
    for step, image in checkpoints:
        path = checkpoint_dir / f"step_{step:03d}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed checkpoint serialization: {path}")
        paths.append((step, path)); decoded.append(load_bgr(path))
    key = row["identity_id"]
    if key not in embedding_cache:
        embedding_cache[key] = (
            get_embedding_from_bgr(arcface, reference, "A"),
            get_embedding_from_bgr(arcface, probe, "B"),
            facenet.get_embedding(reference, label="A"),
            facenet.get_embedding(probe, label="B"),
        )
    ar, ab, vr, vb = embedding_cache[key]
    facenet_adv = facenet.get_embeddings(decoded, label=f"{key} trajectory")
    ac, vc = float(cosine_similarity(ar, ab)), float(cosine_similarity(vr, vb))
    for (step, path), saved, vx in zip(paths, decoded, facenet_adv):
        ax = get_embedding_from_bgr(arcface, saved, f"B_adv step {step}")
        aa, va = float(cosine_similarity(ar, ax)), float(cosine_similarity(vr, vx))
        output.append({"identity_id": key, "method": method, "step": step, "checkpoint_image": str(path.relative_to(ROOT)), "arcface_clean_cosine": ac, "arcface_cosine": aa, "arcface_threshold_crossing": ac >= source_thresholds[PRIMARY_OPERATING_POINT] and aa < source_thresholds[PRIMARY_OPERATING_POINT], "facenet_clean_cosine": vc, "facenet_cosine": va, "facenet_threshold_crossing": vc >= victim_thresholds[PRIMARY_OPERATING_POINT] and va < victim_thresholds[PRIMARY_OPERATING_POINT], "linf_serialized": float(np.abs(saved.astype(np.int16) - probe.astype(np.int16)).max() / 255.)})
    return output


def regression(rows: list[dict]):
    """Exact ON/OFF PNG and pair-metric comparison on two frozen identities."""
    selected = [r for r in rows if r["identity_id"] in {"Michelle_Collins", "Meryl_Streep"}]
    report = []
    source_thresholds, _ = load_thresholds(ROOT / "results/calibration/arcface_lfw_v1")
    victim_thresholds, _ = load_thresholds(ROOT / "results/calibration/facenet_lfw_v1")
    # Keep the attack phase surrogate-only.  The evaluator models are created
    # only after all ON/OFF images have been generated and the attack app has
    # been released.
    app = create_face_app(det_size=DEFAULT_DET_SIZE)
    generated = []
    try:
        for row in selected:
            image, reference = load_bgr(ROOT / row["probe_image"]), load_bgr(ROOT / row["reference_image"])
            for method, (kind, masked) in METHODS.items():
                constraint = LandmarkSuperpixelMask() if masked else None
                if kind == "pgd":
                    off = run_pgd_full(image, [EPS], app, steps=STEPS, constraint=constraint, tv_weight=0., diagnostics=False)[0].attacked_bgr
                else:
                    off = run_mi_fgsm(image, EPS, app, steps=STEPS, alpha=EPS/STEPS, momentum=1., constraint=constraint, tv_weight=0., return_metadata=False, diagnostics=False)
                on, diag, _ = attack(image, app, method)
                generated.append((row["identity_id"], method, reference, image, off, on, len(diag["trajectory"])))
    finally:
        del app
        gc.collect()
    arcface, facenet = create_face_app(det_size=DEFAULT_DET_SIZE), load_embedder("facenet_vggface2")
    try:
        for identity_id, method, reference, image, off, on, trajectory_steps in generated:
            off_metrics = pair_metrics(reference, image, off, arcface, facenet, source_thresholds[PRIMARY_OPERATING_POINT], victim_thresholds[PRIMARY_OPERATING_POINT])
            on_metrics = pair_metrics(reference, image, on, arcface, facenet, source_thresholds[PRIMARY_OPERATING_POINT], victim_thresholds[PRIMARY_OPERATING_POINT])
            equal = bool(np.array_equal(off, on))
            metrics_equal = off_metrics == on_metrics
            report.append({"identity_id": identity_id, "method": method, "serialized_png_sha256_off": png_hash(off), "serialized_png_sha256_on": png_hash(on), "serialized_b_adv_exact": equal, "max_abs_rgb_difference": int(np.abs(off.astype(np.int16)-on.astype(np.int16)).max()), "final_arcface_cosine_off": off_metrics["arcface_cosine"], "final_arcface_cosine_on": on_metrics["arcface_cosine"], "final_facenet_cosine_off": off_metrics["facenet_cosine"], "final_facenet_cosine_on": on_metrics["facenet_cosine"], "linf_off": off_metrics["linf_serialized"], "linf_on": on_metrics["linf_serialized"], "arcface_threshold_crossing_equal": off_metrics["arcface_threshold_crossing"] == on_metrics["arcface_threshold_crossing"], "facenet_threshold_crossing_equal": off_metrics["facenet_threshold_crossing"] == on_metrics["facenet_threshold_crossing"], "trajectory_steps": trajectory_steps, "pass": bool(equal and metrics_equal and trajectory_steps == STEPS)})
    finally:
        del arcface, facenet
        gc.collect()
    write_csv(OUT / "regression_smoke.csv", report)
    (OUT / "regression_smoke.json").write_text(json.dumps({"setting": {"epsilon": EPS, "steps": STEPS, "tv_weight": 0.}, "results": report, "all_pass": all(x["pass"] for x in report)}, indent=2) + "\n", encoding="utf-8")
    if not all(x["pass"] for x in report):
        raise RuntimeError("diagnostic ON/OFF regression failed")


def regression_attack(rows: list[dict], identity: str, method: str, diagnostics: bool) -> None:
    """One bounded attack invocation for hosts that cap foreground jobs at 30 s."""
    row = next((r for r in rows if r["identity_id"] == identity), None)
    if row is None or method not in METHODS:
        raise RuntimeError("invalid frozen regression identity or method")
    image, app = load_bgr(ROOT / row["probe_image"]), create_face_app(det_size=DEFAULT_DET_SIZE)
    kind, masked = METHODS[method]
    constraint = LandmarkSuperpixelMask() if masked else None
    if diagnostics:
        output, diag, _ = attack(image, app, method)
        if len(diag["trajectory"]) != STEPS:
            raise RuntimeError("diagnostic trajectory length mismatch")
    elif kind == "pgd":
        output = run_pgd_full(image, [EPS], app, steps=STEPS, constraint=constraint, tv_weight=0., diagnostics=False)[0].attacked_bgr
    else:
        output = run_mi_fgsm(image, EPS, app, steps=STEPS, alpha=EPS/STEPS, momentum=1., constraint=constraint, tv_weight=0., return_metadata=False, diagnostics=False)
    target = OUT / "regression_artifacts" / identity / method.replace("+", "_mask") / ("on.png" if diagnostics else "off.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), output):
        raise RuntimeError("regression PNG serialization failed")


def regression_evaluate(rows: list[dict], identity: str, method: str) -> None:
    row = next((r for r in rows if r["identity_id"] == identity), None)
    if row is None or method not in METHODS:
        raise RuntimeError("invalid frozen regression identity or method")
    artifact_dir = OUT / "regression_artifacts" / identity / method.replace("+", "_mask")
    off, on = load_bgr(artifact_dir / "off.png"), load_bgr(artifact_dir / "on.png")
    source_thresholds, _ = load_thresholds(ROOT / "results/calibration/arcface_lfw_v1")
    victim_thresholds, _ = load_thresholds(ROOT / "results/calibration/facenet_lfw_v1")
    reference, probe = load_bgr(ROOT / row["reference_image"]), load_bgr(ROOT / row["probe_image"])
    arcface, facenet = create_face_app(det_size=DEFAULT_DET_SIZE), load_embedder("facenet_vggface2")
    try:
        off_metrics = pair_metrics(reference, probe, off, arcface, facenet, source_thresholds[PRIMARY_OPERATING_POINT], victim_thresholds[PRIMARY_OPERATING_POINT])
        on_metrics = pair_metrics(reference, probe, on, arcface, facenet, source_thresholds[PRIMARY_OPERATING_POINT], victim_thresholds[PRIMARY_OPERATING_POINT])
    finally:
        del arcface, facenet
        gc.collect()
    equal = bool(np.array_equal(off, on))
    result = {"identity_id": identity, "method": method, "serialized_png_sha256_off": png_hash(off), "serialized_png_sha256_on": png_hash(on), "serialized_b_adv_exact": equal, "max_abs_rgb_difference": int(np.abs(off.astype(np.int16) - on.astype(np.int16)).max()), "final_arcface_cosine_off": off_metrics["arcface_cosine"], "final_arcface_cosine_on": on_metrics["arcface_cosine"], "final_facenet_cosine_off": off_metrics["facenet_cosine"], "final_facenet_cosine_on": on_metrics["facenet_cosine"], "linf_off": off_metrics["linf_serialized"], "linf_on": on_metrics["linf_serialized"], "arcface_threshold_crossing_equal": off_metrics["arcface_threshold_crossing"] == on_metrics["arcface_threshold_crossing"], "facenet_threshold_crossing_equal": off_metrics["facenet_threshold_crossing"] == on_metrics["facenet_threshold_crossing"], "pass": bool(equal and off_metrics == on_metrics)}
    target = artifact_dir / "evaluation.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def full(rows: list[dict], only: str | None):
    requested = [r for r in rows if only is None or r["identity_id"] == only]
    if only and not requested: raise RuntimeError("identity not in frozen split")
    all_steps, summaries, checkpoint_metrics = [], [], []
    app = create_face_app(det_size=DEFAULT_DET_SIZE)
    arcface, facenet = create_face_app(det_size=DEFAULT_DET_SIZE), load_embedder("facenet_vggface2")
    embedding_cache = {}
    for row in requested:
        image = load_bgr(ROOT / row["probe_image"])
        shared_mask = LandmarkSuperpixelMask().build(image)[0]
        pending_evaluation = []
        for method in METHODS:
            adv, diag, mask = attack(image, app, method, shared_mask if METHODS[method][1] else None)
            method_dir = OUT / "per_identity" / row["identity_id"] / method.replace("+", "_mask")
            method_dir.mkdir(parents=True, exist_ok=True)
            if mask is not None:
                cv2.imwrite(str(method_dir / "final_mask.png"), (mask * 255).astype(np.uint8))
                np.save(method_dir / "final_mask.npy", mask.astype(np.float32))
            trajectory = diag["trajectory"]
            for x in trajectory: x.update(identity_id=row["identity_id"], method=method)
            write_csv(method_dir / "trajectory.csv", trajectory)
            all_steps.extend(trajectory)
            pending_evaluation.append((method, diag["checkpoints"], method_dir / "checkpoints", trajectory[-1]["arcface_cosine"], mask is not None, diag["operation_order"], trajectory[-1]["current_linf"]))
        for method, checkpoints, checkpoint_dir, objective_cosine, persisted_mask, operation_order, final_linf in pending_evaluation:
            # This runs after attack completion and after releasing the attack
            # surrogate.  FaceNet remains strictly evaluation-only.
            metrics = evaluate_checkpoints(row, method, checkpoints, checkpoint_dir, arcface, facenet, embedding_cache)
            checkpoint_metrics.extend(metrics)
            final = metrics[-1]
            summaries.append({"identity_id": row["identity_id"], "method": method, "steps": STEPS, "final_attack_objective_cosine": objective_cosine, "final_pair_arcface_cosine": final["arcface_cosine"], "final_pair_facenet_cosine": final["facenet_cosine"], "final_linf": final_linf, "operation_order": operation_order, "final_mask_persisted": persisted_mask})
    del app, arcface, facenet
    gc.collect()
    write_csv(OUT / "trajectory_all.csv", all_steps)
    write_csv(OUT / "diagnostic_summary.csv", summaries)
    write_csv(OUT / "checkpoint_metrics.csv", checkpoint_metrics)
    (OUT / "provenance.json").write_text(json.dumps({"kind": "diagnostic / mechanism analysis", "formal_benchmark_replaced": False, "frozen_split": "attack_dev.csv", "identities": [r["identity_id"] for r in requested], "methods": list(METHODS), "epsilon": EPS, "steps": STEPS, "tv_weight": 0.}, indent=2) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["regression", "regression-attack", "regression-evaluate", "full"], required=True); p.add_argument("--identity"); p.add_argument("--method", choices=list(METHODS)); p.add_argument("--diagnostics", action="store_true"); p.add_argument("--cpu-threads", type=int, default=min(os.cpu_count() or 1, 8))
    args = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if args.cpu_threads < 1:
        p.error("--cpu-threads must be positive")
    # CPU scheduling only; it does not alter epsilon, steps, masks, objective,
    # or update order.  Regression still gates diagnostic ON/OFF equivalence.
    import torch
    torch.set_num_threads(args.cpu_threads)
    rows = pairs()
    if args.mode == "regression": regression(rows)
    elif args.mode == "regression-attack":
        if not args.identity or not args.method: p.error("regression-attack requires --identity and --method")
        regression_attack(rows, args.identity, args.method, args.diagnostics)
    elif args.mode == "regression-evaluate":
        if not args.identity or not args.method: p.error("regression-evaluate requires --identity and --method")
        regression_evaluate(rows, args.identity, args.method)
    else: full(rows, args.identity)

if __name__ == "__main__": main()
