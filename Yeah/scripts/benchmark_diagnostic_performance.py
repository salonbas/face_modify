#!/usr/bin/env python3
"""Reproducible performance audit for the Michelle Collins 200-step attacks."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
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
from advface.models import load_embedder
from advface.models.arcface_torch import load_arcface_torch
from advface.models.insightface_app import create_face_app, get_embedding_from_bgr

METHODS = {"PGD": ("pgd", False), "MI-FGSM": ("mi", False), "PGD+Mask": ("pgd", True), "MI-FGSM+Mask": ("mi", True)}


def stamp(device: str) -> float:
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)
    return time.perf_counter()


def michelle() -> dict[str, str]:
    with (ROOT / "data/datasets/lfw/evaluation_splits/attack_dev.csv").open(newline="", encoding="utf-8") as handle:
        return next(row for row in csv.DictReader(handle) if row["identity_id"] == "Michelle_Collins")


def attack(image, app, method: str, steps: int, diagnostics: bool, device: str, mask):
    kind, masked = METHODS[method]
    kwargs = dict(steps=steps, device=device, constraint=LandmarkSuperpixelMask() if masked else None,
                  precomputed_mask=mask if masked else None, tv_weight=0., diagnostics=diagnostics)
    if kind == "pgd":
        result = run_pgd_full(image, [.04], app, **kwargs)[0]
        return result.attacked_bgr, result.diagnostics
    return run_mi_fgsm(image, .04, app, alpha=.04 / steps, momentum=1., return_metadata=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--method", choices=list(METHODS), action="append")
    parser.add_argument("--diagnostics", choices=("off", "on", "both"), default="both")
    parser.add_argument("--output", type=Path, default=ROOT / "results/diagnostics/performance_audit_michelle.json")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    row = michelle(); reference = load_bgr(ROOT / row["reference_image"]); probe = load_bgr(ROOT / row["probe_image"])
    runtime = {"torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
               "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
               "cuda_available_in_this_process": torch.cuda.is_available(), "device": args.device}
    t0 = stamp(args.device); attack_app = create_face_app(det_size=DEFAULT_DET_SIZE); load_arcface_torch(args.device); model_loading = stamp(args.device) - t0
    t0 = stamp(args.device); shared_mask = LandmarkSuperpixelMask().build(probe)[0]; mask_seconds = stamp(args.device) - t0
    modes = (False, True) if args.diagnostics == "both" else (args.diagnostics == "on",)
    results = []
    out_dir = args.output.parent / "performance_checkpoints"
    for method in args.method or METHODS:
        for diagnostics in modes:
            t0 = stamp(args.device); value = attack(probe, attack_app, method, args.steps, diagnostics, args.device, shared_mask); attack_seconds = stamp(args.device) - t0
            if METHODS[method][0] == "pgd": adv, diag = value
            else: adv, metadata = value; diag = metadata.get("diagnostics")
            checkpoints = diag["checkpoints"] if diag else [(args.steps, adv)]
            paths = []; t0 = time.perf_counter()
            target = out_dir / method.replace("+", "_") / ("on" if diagnostics else "off"); target.mkdir(parents=True, exist_ok=True)
            for step, image in checkpoints:
                path = target / f"step_{step:03d}.png"
                if not cv2.imwrite(str(path), image): raise RuntimeError(f"failed to write {path}")
                paths.append(path)
            serialization_seconds = time.perf_counter() - t0
            decoded = [load_bgr(path) for path in paths]
            t0 = time.perf_counter(); arcface_eval = create_face_app(det_size=DEFAULT_DET_SIZE); arc_load = time.perf_counter() - t0
            t0 = time.perf_counter(); get_embedding_from_bgr(arcface_eval, reference, "A"); get_embedding_from_bgr(arcface_eval, decoded[-1], "B_adv"); arc_eval = time.perf_counter() - t0
            t0 = time.perf_counter(); facenet = load_embedder("facenet_vggface2", device=args.device); face_load = time.perf_counter() - t0
            t0 = stamp(args.device); facenet.get_embedding(reference, "A"); facenet.get_embeddings(decoded, "trajectory"); face_eval = stamp(args.device) - t0
            perceptual_load = perceptual_eval = None
            try:
                t0 = time.perf_counter(); import lpips; from DISTS_pytorch import DISTS
                lp, ds = lpips.LPIPS(net="alex", verbose=False).eval().to(args.device), DISTS().eval().to(args.device); perceptual_load = time.perf_counter() - t0
                a = torch.from_numpy(cv2.cvtColor(probe, cv2.COLOR_BGR2RGB)).permute(2,0,1)[None].float().to(args.device) / 255
                b = torch.from_numpy(cv2.cvtColor(decoded[-1], cv2.COLOR_BGR2RGB)).permute(2,0,1)[None].float().to(args.device) / 255
                t0 = stamp(args.device)
                with torch.inference_mode(): lp(a * 2 - 1, b * 2 - 1); ds(a, b)
                perceptual_eval = stamp(args.device) - t0
            except (ImportError, RuntimeError) as exc:
                perceptual_eval = f"unavailable: {exc}"
            total = model_loading + mask_seconds + attack_seconds + serialization_seconds + arc_load + arc_eval + face_load + face_eval
            if isinstance(perceptual_load, float): total += perceptual_load + perceptual_eval
            results.append({"method": method, "diagnostics": diagnostics, "steps": args.steps, "model_loading_seconds": model_loading,
                "landmark_slic_mask_seconds": mask_seconds, "attack_seconds": attack_seconds, "ms_per_step": attack_seconds * 1000 / args.steps,
                "checkpoint_serialization_seconds": serialization_seconds, "arcface_model_loading_seconds": arc_load, "arcface_evaluation_seconds": arc_eval,
                "facenet_model_loading_seconds": face_load, "facenet_checkpoint_evaluation_seconds": face_eval,
                "perceptual_model_loading_seconds": perceptual_load, "perceptual_metric_evaluation_seconds": perceptual_eval, "total_accounted_seconds": total})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"identity": "Michelle_Collins", "runtime": runtime, "results": results}, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__": main()
