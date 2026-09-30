#!/usr/bin/env python3
"""Resumable, one-subprocess-per-identity LFW development v2 runner.

Every worker writes B_adv, decodes that exact PNG, and computes all reported
metrics from it.  Workers deliberately exit after one identity so heavyweight
Torch/ONNX/perceptual objects cannot accumulate on the CPU host.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from advface.attacks.registry import ATTACKS, AttackConfig
from advface.config import DEFAULT_DET_SIZE
from advface.constraints import LandmarkSuperpixelMask
from advface.evaluation.similarity import cosine_similarity
from advface.evaluation.verification import PRIMARY_OPERATING_POINT, load_thresholds
from advface.image_io import load_bgr
from advface.models import load_embedder
from advface.models.insightface_app import create_face_app, get_embedding_from_bgr

EPS = 0.04
STEPS = 200
FIELDS = ("identity_id", "attack", "constraint", "reference_image", "probe_image", "adversarial_image", "configured_epsilon", "steps", "tv_weight", "actual_linf_tensor", "actual_linf_serialized", "arcface_clean_cosine", "arcface_adv_cosine", "arcface_cosine_drop", "arcface_threshold_crossing", "facenet_clean_cosine", "facenet_adv_cosine", "facenet_cosine_drop", "facenet_transfer_crossing", "ssim", "lpips", "dists", "mask_outside_delta_max", "runtime_seconds", "valid", "failure_reason")


def read_split() -> list[dict[str, str]]:
    with (ROOT / "data/datasets/lfw/evaluation_splits/attack_dev.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 20 or any(r["split"] != "attack_dev" for r in rows):
        raise RuntimeError("frozen attack_dev.csv must contain exactly its 20 development identities")
    return rows


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def perceptual_tensor(value: np.ndarray):
    import torch
    tensor = torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))).unsqueeze(0).float() / 255
    height, width = tensor.shape[-2:]
    scale = min(1.0, 256.0 / max(height, width))
    if scale < 1:
        import torch.nn.functional as F
        tensor = F.interpolate(tensor, size=(round(height * scale), round(width * scale)), mode="bilinear", align_corners=False)
    return tensor


def metrics_from_png(probe_path: Path, adv_path: Path) -> tuple[float, float, float, float]:
    """All four pixel/perceptual values use the decoded serialized artifact."""
    import torch
    import lpips
    from DISTS_pytorch import DISTS
    clean, saved = rgb(probe_path), rgb(adv_path)
    actual = float(np.abs(saved.astype(np.int16) - clean.astype(np.int16)).max() / 255)
    ssim = float(structural_similarity(clean.astype(np.float32) / 255, saved.astype(np.float32) / 255, channel_axis=2, data_range=1.0))
    clean_t, saved_t = perceptual_tensor(clean), perceptual_tensor(saved)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()
    with torch.inference_mode():
        lpips_value = float(lpips_model(clean_t * 2 - 1, saved_t * 2 - 1).item())
    del lpips_model
    gc.collect()
    dists_model = DISTS().eval()
    with torch.inference_mode():
        dists_value = float(dists_model(clean_t, saved_t).item())
    del dists_model, clean_t, saved_t
    gc.collect()
    return actual, ssim, lpips_value, dists_value


def write_visuals(out: Path, ident: str, original: np.ndarray, adv: np.ndarray, constraint: LandmarkSuperpixelMask) -> float:
    visual = out / "visualizations" / ident
    visual.mkdir(parents=True, exist_ok=True)
    final, initial, points = constraint.build(original)
    labels = constraint.segmentation(original)
    delta = np.abs(adv.astype(np.int16) - original.astype(np.int16)).max(axis=2)
    cv2.imwrite(str(visual / "original.png"), original)
    cv2.imwrite(str(visual / "landmark_indices_overlay.png"), constraint.landmark_region_overlay(original, points))
    cv2.imwrite(str(visual / "initial_polygon_mask.png"), (initial * 255).astype(np.uint8))
    cv2.imwrite(str(visual / "slic_segmentation.png"), constraint.colorize_segments(labels))
    cv2.imwrite(str(visual / "final_mask.png"), (final * 255).astype(np.uint8))
    cv2.imwrite(str(visual / "perturbation.png"), (np.clip(delta * 255 / max(int(delta.max()), 1), 0, 255)).astype(np.uint8))
    cv2.imwrite(str(visual / "B_adv.png"), adv)
    return float(delta[final == 0].max(initial=0))


def worker(args: argparse.Namespace, pair: dict[str, str], out: Path) -> int:
    ident = pair["identity_id"]
    row: dict[str, Any] = {field: "" for field in FIELDS}
    constrained = args.constraint == "landmark_superpixel_mask"
    row.update(identity_id=ident, attack=args.attack, constraint=args.constraint or "none", reference_image=pair["reference_image"], probe_image=pair["probe_image"], configured_epsilon=EPS, steps=args.steps, tv_weight=0.0, valid=False, failure_reason="")
    try:
        source_thresholds, _ = load_thresholds(ROOT / "results/calibration/arcface_lfw_v1")
        victim_thresholds, _ = load_thresholds(ROOT / "results/calibration/facenet_lfw_v1")
        a, b = load_bgr(ROOT / pair["reference_image"]), load_bgr(ROOT / pair["probe_image"])
        app = create_face_app(det_size=DEFAULT_DET_SIZE)
        started = time.perf_counter()
        extra = {"tv_weight": 0.0}
        if constrained:
            extra["constraint"] = args.constraint
        attack = ATTACKS[args.attack].apply(b, app=app, config=AttackConfig(name=args.attack, eps=EPS, steps=args.steps, momentum=1.0, extra=extra))
        row["runtime_seconds"] = time.perf_counter() - started
        tensor_linf = float(attack.parameters["linf_tensor"])
        if tensor_linf > EPS + 1e-6:
            raise RuntimeError(f"tensor L∞ contract failed: {tensor_linf}")
        adv_dir = out / "adversarial_probes"; adv_dir.mkdir(parents=True, exist_ok=True)
        suffix = "pgd_mask" if constrained else args.attack
        adv_path = adv_dir / f"{ident}_B_adv_{suffix}.png"
        if not cv2.imwrite(str(adv_path), attack.adversarial_bgr):
            raise RuntimeError("failed to serialize B_adv PNG")
        # Re-open once; every score and perceptual value below is from this PNG.
        adv = load_bgr(adv_path)
        ar, ab, ax = (get_embedding_from_bgr(app, image, f"{ident}:{label}") for image, label in ((a, "A"), (b, "B"), (adv, "B_adv")))
        ac, aa = float(cosine_similarity(ar, ab)), float(cosine_similarity(ar, ax))
        del app, ar, ab, ax
        gc.collect()
        victim = load_embedder("facenet_vggface2")
        vr, vb, vx = (victim.get_embedding(image, label=f"{ident}:{label}") for image, label in ((a, "A"), (b, "B"), (adv, "B_adv")))
        vc, va = float(cosine_similarity(vr, vb)), float(cosine_similarity(vr, vx))
        del victim, vr, vb, vx
        gc.collect()
        actual, ssim, lpips_value, dists_value = metrics_from_png(ROOT / pair["probe_image"], adv_path)
        outside = ""
        if constrained:
            outside = write_visuals(out, ident, b, adv, LandmarkSuperpixelMask())
            if outside != 0:
                raise RuntimeError(f"serialized perturbation outside final mask: {outside}")
        row.update(adversarial_image=str(adv_path.relative_to(ROOT)), actual_linf_tensor=tensor_linf, actual_linf_serialized=actual, arcface_clean_cosine=ac, arcface_adv_cosine=aa, arcface_cosine_drop=ac-aa, arcface_threshold_crossing=ac >= source_thresholds[PRIMARY_OPERATING_POINT] and aa < source_thresholds[PRIMARY_OPERATING_POINT], facenet_clean_cosine=vc, facenet_adv_cosine=va, facenet_cosine_drop=vc-va, facenet_transfer_crossing=vc >= victim_thresholds[PRIMARY_OPERATING_POINT] and va < victim_thresholds[PRIMARY_OPERATING_POINT], ssim=ssim, lpips=lpips_value, dists=dists_value, mask_outside_delta_max=outside, valid=True)
    except Exception as exc:
        row["failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Process exit is the hard memory boundary; this is useful if invoked directly too.
        gc.collect()
    record = out / "records"; record.mkdir(parents=True, exist_ok=True)
    (record / f"{ident}.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if row["valid"] else 1


def aggregate(out: Path, args: argparse.Namespace, pairs: list[dict[str, str]]) -> int:
    rows = []
    for pair in pairs:
        record = out / "records" / f"{pair['identity_id']}.json"
        if record.exists(): rows.append(json.loads(record.read_text(encoding="utf-8")))
    with (out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    valid = [r for r in rows if r["valid"] is True]
    mean = lambda key: float(np.mean([float(r[key]) for r in valid])) if valid else None
    summary = {"title": args.title, "formal": args.formal, "scope": f"attack_dev.csv frozen development identities; n={len(pairs)}", "completeness": {"expected": len(pairs), "records": len(rows), "valid": len(valid), "failures": len(rows)-len(valid), "complete": len(valid) == len(pairs)}, "attack": {"name": args.attack, "configured_epsilon": EPS, "steps": args.steps, "constraint": args.constraint or None, "tv_weight": 0.0}, "provenance": {"single_serialized_artifact_per_observation": True, "all_metrics_read_from_decoded_B_adv_png": True, "per_identity_subprocess": True, "resume_skips_valid_completed_records": True}, "aggregate": {"arcface_asr": sum(bool(r["arcface_threshold_crossing"]) for r in valid) / len(valid) if valid else None, "facenet_asr": sum(bool(r["facenet_transfer_crossing"]) for r in valid) / len(valid) if valid else None, **{key: mean(key) for key in ("actual_linf_serialized", "arcface_cosine_drop", "facenet_cosine_drop", "ssim", "lpips", "dists")}}}
    if args.constraint:
        summary["landmark_ordering_validation"] = {
            "model": "InsightFace buffalo_l/1k3d68.onnx",
            "status": "implementation_assumption verified by saved overlay; no official semantic index mapping was found in the model artifact",
            "assumed_zero_based_regions": {"eyebrows": "17-26", "nose": "27-35", "eyes": "36-47", "mouth": "48-67"},
            "inspection_files": "visualizations/<identity>/landmark_indices_overlay.png",
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["completeness"]["complete"] else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attack", choices=("pgd_full", "mi_fgsm"), default="pgd_full")
    p.add_argument("--constraint", choices=("landmark_superpixel_mask",), default=None)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--max-identities", type=int, default=20, choices=(1, 5, 20))
    p.add_argument("--identity", default=None, help="internal worker mode: exactly one attack_dev identity")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--formal", action="store_true")
    p.add_argument("--title", default="LFW attack_dev development run")
    args = p.parse_args()
    if args.steps <= 0 or args.steps > STEPS: raise SystemExit("steps must be in 1..200")
    if args.formal and args.steps != STEPS: raise SystemExit("formal runs require 200 steps")
    if args.formal and args.max_identities not in (5, 20): raise SystemExit("formal runs are either the 20-identity baseline or fixed 5-identity mask ablation")
    pairs = read_split(); selected = {r["identity_id"]: r for r in pairs}
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if args.identity:
        if args.identity not in selected: raise SystemExit("identity is not in frozen attack_dev.csv")
        return worker(args, selected[args.identity], out)
    pairs = pairs[:args.max_identities]
    out.mkdir(parents=True, exist_ok=True)
    config = {"split": "data/datasets/lfw/evaluation_splits/attack_dev.csv", "identity_count": len(pairs), "attack": args.attack, "configured_epsilon": EPS, "steps": args.steps, "constraint": args.constraint, "tv_weight": 0.0, "execution": "sequential independent subprocess per identity", "formal": args.formal}
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for pair in pairs:
        record = out / "records" / f"{pair['identity_id']}.json"
        if record.exists():
            existing = json.loads(record.read_text(encoding="utf-8"))
            artifact = ROOT / existing.get("adversarial_image", "")
            if existing.get("valid") is True and artifact.is_file():
                print(f"resume: {pair['identity_id']} already complete")
                continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "--attack", args.attack, "--steps", str(args.steps), "--identity", pair["identity_id"], "--output-dir", str(out), "--title", args.title]
        if args.constraint: cmd += ["--constraint", args.constraint]
        if args.formal: cmd += ["--formal"]
        print(f"running isolated worker: {pair['identity_id']}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=False)
    return aggregate(out, args, pairs)


if __name__ == "__main__":
    raise SystemExit(main())
