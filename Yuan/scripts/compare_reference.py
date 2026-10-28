"""Paired MI-FGSM self-reference vs A-reference development experiment.

Run from any directory. Inputs are read-only; outputs stay under Yuan/results.
FaceNet participates only in evaluation, never attack gradients.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEPENDENCIES = ("numpy", "cv2", "torch", "torchvision", "insightface",
                "onnxruntime", "onnx2torch", "facenet_pytorch")


def read_pairs(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"identity_id", "reference_image", "probe_image"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"split must contain nonempty rows with {sorted(required)}")
    ids = [row["identity_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate identities in split")
    return rows


def input_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def preflight(root, split, limit):
    issues = []
    missing = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
    if missing:
        issues.append("Missing Python packages: " + ", ".join(missing))
    pairs = []
    if not split.is_file():
        issues.append(f"Missing development split: {split}")
    else:
        try:
            pairs = read_pairs(split)[:limit] if limit else read_pairs(split)
            for pair in pairs:
                for field in ("reference_image", "probe_image"):
                    path = input_path(root, pair[field])
                    if not path.is_file():
                        issues.append(f"Missing image: {path}")
        except (ValueError, OSError) as exc:
            issues.append(str(exc))
    for model in ("arcface", "facenet"):
        for filename in ("thresholds.json", "summary.json"):
            path = root / f"results/calibration/{model}_lfw_v1" / filename
            if not path.is_file():
                issues.append(f"Missing calibration artifact: {path}")
    # Require local weights; this experiment does not download assets implicitly.
    for filename in ("det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx", "1k3d68.onnx", "genderage.onnx"):
        path = root / "models/insightface/buffalo_l" / filename
        if not path.is_file():
            issues.append(f"Missing InsightFace model: {path}")
    weights = root / "models/facenet/20180402-114759-vggface2.pt"
    if not weights.is_file():
        issues.append(f"Missing FaceNet model (copy existing cache here): {weights}")
    return pairs, issues


def aggregate(rows, mode):
    selected = [row for row in rows if row["mode"] == mode]
    summary = {"n": len(selected)}
    for model in ("source", "victim"):
        eligible = [row for row in selected if row[f"{model}_clean_valid"]]
        count = sum(row[f"{model}_boundary_crossed"] for row in eligible)
        summary[model] = {
            "eligible": len(eligible), "successes": count,
            "asr": count / len(eligible) if eligible else None,
            "mean_cosine_drop": (sum(row[f"{model}_drop"] for row in eligible)
                                 / len(eligible)) if eligible else None,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "Yeah")
    parser.add_argument("--split", type=Path, help="development CSV; paths relative to data-root")
    parser.add_argument("--limit", type=int, default=0, help="0 = all development identities")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eps", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--mask", action="store_true", help="landmark-superpixel gradient mask, shared by self/pair")
    parser.add_argument("--di", action="store_true", help="random shrink/pad after alignment, fixed reference")
    parser.add_argument("--di-probability", type=float, default=0.7)
    parser.add_argument("--di-min-scale", type=float, default=0.9)
    parser.add_argument("--ti", action="store_true", help="Gaussian TI on full-image gradient before mask")
    parser.add_argument("--ti-kernel-size", type=int, default=5)
    parser.add_argument("--ti-sigma", type=float, default=1.0)
    parser.add_argument("--pair-only", action="store_true", help="run A-reference only; default compares self and pair")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for torch and ONNX Runtime")
    args = parser.parse_args()
    modes = ("pair",) if args.pair_only else ("self", "pair")
    if args.ti_kernel_size < 1 or args.ti_kernel_size % 2 != 1 or not math.isfinite(args.ti_sigma) or args.ti_sigma <= 0:
        parser.error("TI requires positive odd kernel-size and finite sigma > 0")
    if not 0 <= args.di_probability <= 1 or not 0 < args.di_min_scale <= 1:
        parser.error("DI: probability in [0,1], min-scale in (0,1]")
    if args.steps <= 0 or not 0 < args.eps <= 1 or args.limit < 0 or args.threads <= 0:
        parser.error("steps > 0, 0 < eps <= 1, limit >= 0 required")
    data_root = args.data_root.resolve()
    split = (args.split or data_root / "data/datasets/lfw/evaluation_splits/attack_dev.csv").resolve()
    if split.name != "attack_dev.csv":
        parser.error("this runner accepts attack_dev.csv only; held-out splits are not for tuning")
    pairs, issues = preflight(data_root, split, args.limit)
    print(json.dumps({"local_package": str(ROOT / "advface"), "pairs": len(pairs),
                      "ready": not issues, "issues": issues}, indent=2))
    if issues:
        return 2
    if args.check_only:
        return 0

    os.environ["YUAN_INSIGHTFACE_ROOT"] = str(data_root / "models/insightface")
    os.environ["YUAN_FACENET_WEIGHT"] = str(data_root / "models/facenet/20180402-114759-vggface2.pt")
    os.environ["YUAN_CPU_THREADS"] = str(args.threads)

    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from advface.attacks.mi_fgsm import run_mi_fgsm
    from advface.evaluation.verification import evaluate_verification_pair, load_thresholds
    from advface.image_io import load_bgr
    from advface.models.facenet import FaceNetEmbedder
    from advface.models.insightface_app import create_face_app, get_embedding_from_bgr

    prefix = "reference_" + ("di_" if args.di else "") + ("ti_" if args.ti else "") + ("mask_" if args.mask else "")
    output = ROOT / "results" / (prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    source_thresholds, source_artifact = load_thresholds(data_root / "results/calibration/arcface_lfw_v1")
    victim_thresholds, victim_artifact = load_thresholds(data_root / "results/calibration/facenet_lfw_v1")
    config = {"experimental": True, "formal": False, "split": str(split),
              "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
              "identities": [p["identity_id"] for p in pairs], "eps": args.eps,
              "steps": args.steps, "alpha": args.eps / args.steps, "momentum": 1.0,
              "seed": args.seed, "mask": "landmark_superpixel_mask" if args.mask else None,
              "mask_options": {"n_segments": 100, "compactness": 10.0, "sigma": 0.0} if args.mask else None,
              "tv_weight": 0.0, "random_start": False, "threads": args.threads,
              "reference_modes": list(modes),
              "ti": {"enabled": args.ti, "kernel_size": args.ti_kernel_size if args.ti else 1,
                     "sigma_pixels": args.ti_sigma, "padding": "replicate",
                     "domain": "full-image gradient", "channels": "independent depthwise",
                     "order": "DI backward -> TI -> mask -> L1 normalization -> momentum -> sign -> projection"},
              "di": {"enabled": args.di, "probability": args.di_probability if args.di else 0.0,
                     "min_scale": args.di_min_scale, "seed": args.seed,
                     "location": "normalized aligned B crop only; A/reference unchanged",
                     "padding": "neutral RGB 127.5", "variant": "shrink+random pad to fixed 112x112"},
              "runtime": {"python": sys.version, "torch": torch.__version__,
                          "numpy": np.__version__, "opencv": cv2.__version__,
                          "cuda_available": torch.cuda.is_available(),
                          "cudnn_deterministic": True, "cudnn_benchmark": False},
              "victim_gradient_participation": False,
              "source_calibration": str(source_artifact), "victim_calibration": str(victim_artifact),
              "objectives": {"self": "min cos(ArcFace(B_adv), ArcFace(B))",
                             "pair": "min cos(ArcFace(B_adv), ArcFace(A))"}}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    app = create_face_app()
    if args.mask:
        from advface.constraints.landmark_superpixel import LandmarkSuperpixelMask
        constraint = LandmarkSuperpixelMask(landmark_app=app)
    rows, failures = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    victim = FaceNetEmbedder(device=device)
    for index, pair in enumerate(pairs):
        # Append both modes only when the entire pair succeeds, keeping matched samples.
        pair_rows = []
        try:
            a = load_bgr(input_path(data_root, pair["reference_image"]))
            b = load_bgr(input_path(data_root, pair["probe_image"]))
            final_mask = None
            mask_path = None
            if args.mask:
                final_mask, initial_mask, landmarks = constraint.build(b)
                if not np.any(final_mask):
                    raise ValueError("empty landmark-superpixel mask")
                mask_path = output / f"{index:03d}_mask.png"
                artifacts = {mask_path: (final_mask * 255).astype(np.uint8),
                    output / f"{index:03d}_initial_mask.png": (initial_mask * 255).astype(np.uint8),
                    output / f"{index:03d}_mask_overlay.png": constraint.overlay(b, final_mask, initial_mask, landmarks),
                    output / f"{index:03d}_landmark_indices.png": constraint.landmark_region_overlay(b, landmarks)}
                for artifact_path, image in artifacts.items():
                    if not cv2.imwrite(str(artifact_path), image):
                        raise OSError(f"could not write {artifact_path}")
                np.save(output / f"{index:03d}_landmarks.npy", landmarks)
            sa, sb = [get_embedding_from_bgr(app, image, label) for image, label in ((a, "A"), (b, "B"))]
            va, vb = [victim.get_embedding(image, label) for image, label in ((a, "A"), (b, "B"))]
            for mode in modes:
                print(f"Starting {index + 1}/{len(pairs)} {pair['identity_id']} {mode}", flush=True)
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                adv, metadata = run_mi_fgsm(b, args.eps, app, steps=args.steps,
                    alpha=args.eps / args.steps, momentum=1.0, device=device,
                    reference_image=a if mode == "pair" else None, return_metadata=True,
                    precomputed_mask=final_mask, di_probability=args.di_probability if args.di else 0.0,
                    di_min_scale=args.di_min_scale, di_seed=args.seed,
                    ti_kernel_size=args.ti_kernel_size if args.ti else 1, ti_sigma=args.ti_sigma)
                if metadata["linf_tensor"] > args.eps + 1e-6:
                    raise ValueError("tensor epsilon exceeded")
                path = output / f"{index:03d}_{mode}.png"
                if not cv2.imwrite(str(path), adv):
                    raise OSError(f"could not write {path}")
                decoded = load_bgr(path)
                actual_linf = float(np.abs(decoded.astype(np.int16) - b.astype(np.int16)).max() / 255)
                if actual_linf > args.eps + 1 / 255 + 1e-6:
                    raise ValueError("serialized epsilon + quantization tolerance exceeded")
                row = {"identity_id": pair["identity_id"], "mode": mode,
                       "adversarial_image": str(path), "linf_tensor": metadata["linf_tensor"],
                       "linf_serialized": actual_linf,
                       "rgb_mae": float(np.abs(decoded.astype(float) - b.astype(float)).mean())}
                row.update({key: metadata[key] for key in (
                    "initial_objective_cosine", "final_objective_cosine", "initial_gradient_l1")})
                row["di_applied_steps"] = metadata["di_applied_steps"]
                row["ti_kernel_size"] = metadata["ti_kernel_size"]
                row["ti_sigma"] = metadata["ti_sigma"]
                if final_mask is not None:
                    outside = np.abs(decoded.astype(np.int16) - b.astype(np.int16))[final_mask == 0]
                    outside_max = int(outside.max()) if outside.size else 0
                    if outside_max != 0:
                        raise ValueError(f"mask outside pixel delta != 0: {outside_max}")
                    row.update(mask_image=str(mask_path), mask_coverage=float(final_mask.mean()),
                               mask_outside_delta_max=outside_max)
                for name, ref, probe, adv_emb, thresholds in (
                    ("source", sa, sb, get_embedding_from_bgr(app, decoded, "B_adv"), source_thresholds),
                    ("victim", va, vb, victim.get_embedding(decoded, "B_adv"), victim_thresholds)):
                    metrics = evaluate_verification_pair(ref, probe, adv_emb, thresholds)
                    row.update({f"{name}_clean_valid": bool(metrics["clean_pair_valid"]),
                                f"{name}_boundary_crossed": bool(metrics["crossed_primary_boundary"]),
                                f"{name}_clean_cosine": float(metrics["clean_cosine_A_B"]),
                                f"{name}_adv_cosine": float(metrics["modified_cosine_A_Bmodified"]),
                                f"{name}_drop": float(metrics["verification_similarity_change"])})
                pair_rows.append(row)
            rows.extend(pair_rows)
        except Exception as exc:
            failures.append({"identity_id": pair["identity_id"], "error": str(exc)})
        # Persist after each identity so interrupted runs retain usable diagnostics.
        if rows:
            with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        summary = {"experimental": True, "formal": False, "requested": len(pairs),
                   "completed_pairs": len(rows) // len(modes), "failures": failures,
                   "self": aggregate(rows, "self"), "pair": aggregate(rows, "pair"),
                   "perceptual_metrics": "RGB MAE only; SSIM/LPIPS/DISTS not evaluated"}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"{index + 1}/{len(pairs)} {pair['identity_id']}: {len(pair_rows) == len(modes)}", flush=True)
    print(f"Results: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
