#!/usr/bin/env python3
"""Five-pair integration smoke: PGD Full versus PGD Full + landmark mask.

This is intentionally not an Authority Table experiment.  It validates that a
single saved B_adv PNG is read back for both verification and perceptual
metrics, while leaving the frozen v1 baseline untouched.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from time import perf_counter

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

EPS, STEPS = 0.04, 200
IDENTITIES = ("Michelle_Collins", "Meryl_Streep", "Jolanta_Kwasniewski", "Dino_de_Laurentis", "Cyndi_Thompson")
FIELDS = ("identity_id", "attack", "reference_image", "probe_image", "adversarial_image", "configured_epsilon", "actual_linf_serialized", "actual_linf_tensor", "arcface_clean_cosine", "arcface_adv_cosine", "arcface_cosine_drop", "arcface_threshold_crossing", "facenet_clean_cosine", "facenet_adv_cosine", "facenet_cosine_drop", "facenet_transfer_crossing", "ssim", "lpips", "dists", "mask_outside_delta_max", "runtime_seconds", "valid", "failure_reason")


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def deep_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))).unsqueeze(0).float() / 255


def mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> int:
    out = ROOT / "results/smoke/lfw_pgd_landmark_superpixel_mask_v1"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True); (out / "adversarial_probes").mkdir(); (out / "visualizations").mkdir()
    split = list(csv.DictReader((ROOT / "data/datasets/lfw/evaluation_splits/attack_dev.csv").open(encoding="utf-8")))[:5]
    if tuple(row["identity_id"] for row in split) != IDENTITIES:
        raise RuntimeError("frozen five-identity regression subset changed")
    source_thresholds, _ = load_thresholds(ROOT / "results/calibration/arcface_lfw_v1")
    victim_thresholds, _ = load_thresholds(ROOT / "results/calibration/facenet_lfw_v1")
    app, victim = create_face_app(det_size=DEFAULT_DET_SIZE), load_embedder("facenet_vggface2")
    rows = []
    for kind, extra in (("PGD", {}), ("PGD+Mask", {"constraint": "landmark_superpixel_mask"})):
        for pair in split:
            ident = pair["identity_id"]; row = {key: "" for key in FIELDS}
            row.update(identity_id=ident, attack=kind, reference_image=pair["reference_image"], probe_image=pair["probe_image"], configured_epsilon=EPS, valid=False, failure_reason="")
            try:
                a, b = load_bgr(ROOT / pair["reference_image"]), load_bgr(ROOT / pair["probe_image"])
                started = perf_counter()
                attack = ATTACKS["pgd_full"].apply(b, app=app, config=AttackConfig(name="pgd_full", eps=EPS, steps=STEPS, extra=extra))
                row["runtime_seconds"] = perf_counter() - started
                if attack.parameters["linf_tensor"] > EPS + 1e-6:
                    raise RuntimeError("tensor epsilon contract failed")
                adv_path = out / "adversarial_probes" / f"{ident}_{kind.replace('+', '_')}_B_adv.png"
                cv2.imwrite(str(adv_path), attack.adversarial_bgr)
                # Decode the saved PNG: every reported metric below refers to
                # this same immutable artifact, never an in-memory/regenerated image.
                adv = load_bgr(adv_path)
                ar, ab, ax = (get_embedding_from_bgr(app, image, f"{ident}:{label}") for image, label in ((a, "A"), (b, "B"), (adv, "B_adv")))
                vr, vb, vx = (victim.get_embedding(image, label=f"{ident}:{label}") for image, label in ((a, "A"), (b, "B"), (adv, "B_adv")))
                ac, aa = float(cosine_similarity(ar, ab)), float(cosine_similarity(ar, ax))
                vc, va = float(cosine_similarity(vr, vb)), float(cosine_similarity(vr, vx))
                b_rgb, adv_rgb = rgb(ROOT / pair["probe_image"]), rgb(adv_path)
                outside = 0.0
                if kind == "PGD+Mask":
                    constraint = LandmarkSuperpixelMask(); final, initial, points = constraint.build(b)
                    delta = np.abs(adv.astype(np.int16) - b.astype(np.int16)).max(axis=2)
                    outside = float(delta[final == 0].max(initial=0))
                    cv2.imwrite(str(out / "visualizations" / f"{ident}_original.png"), b)
                    cv2.imwrite(str(out / "visualizations" / f"{ident}_final_mask.png"), (final * 255).astype(np.uint8))
                    cv2.imwrite(str(out / "visualizations" / f"{ident}_mask_overlay.png"), constraint.overlay(b, final, initial, points))
                    cv2.imwrite(str(out / "visualizations" / f"{ident}_perturbation.png"), (np.clip(delta * 255 / max(int(delta.max()), 1), 0, 255)).astype(np.uint8))
                    cv2.imwrite(str(out / "visualizations" / f"{ident}_adversarial.png"), adv)
                row.update(adversarial_image=str(adv_path.relative_to(ROOT)), actual_linf_serialized=float(np.abs(adv.astype(np.int16)-b.astype(np.int16)).max()/255), actual_linf_tensor=attack.parameters["linf_tensor"], arcface_clean_cosine=ac, arcface_adv_cosine=aa, arcface_cosine_drop=ac-aa, arcface_threshold_crossing=ac >= source_thresholds[PRIMARY_OPERATING_POINT] and aa < source_thresholds[PRIMARY_OPERATING_POINT], facenet_clean_cosine=vc, facenet_adv_cosine=va, facenet_cosine_drop=vc-va, facenet_transfer_crossing=vc >= victim_thresholds[PRIMARY_OPERATING_POINT] and va < victim_thresholds[PRIMARY_OPERATING_POINT], ssim=float(structural_similarity(b_rgb.astype(np.float32)/255, adv_rgb.astype(np.float32)/255, channel_axis=2, data_range=1.0)), mask_outside_delta_max=outside, valid=True)
            except Exception as exc:
                row["failure_reason"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    # Deep perceptual networks are deliberately loaded one at a time, after
    # ArcFace/FaceNet scoring has completed, to fit the CPU research host.
    del app, victim
    import gc
    import lpips
    import torch
    from DISTS_pytorch import DISTS
    lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()
    with torch.inference_mode():
        for row in rows:
            if row["valid"]:
                probe, adv = rgb(ROOT / row["probe_image"]), rgb(ROOT / row["adversarial_image"])
                row["lpips"] = float(lpips_model(deep_tensor(probe) * 2 - 1, deep_tensor(adv) * 2 - 1).item())
    del lpips_model; gc.collect()
    dists_model = DISTS().eval()
    with torch.inference_mode():
        for row in rows:
            if row["valid"]:
                probe, adv = rgb(ROOT / row["probe_image"]), rgb(ROOT / row["adversarial_image"])
                row["dists"] = float(dists_model(deep_tensor(probe), deep_tensor(adv)).item())
    del dists_model; gc.collect()
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    summary = {"title": "LFW PGD landmark-superpixel mask smoke v1", "formal": False, "reason_not_formal": "engineering smoke on frozen 5-identity regression subset", "mask": {"model": "InsightFace buffalo_l/1k3d68.onnx", "regions": "eyebrows, eyes, nose, mouth", "slic": {"n_segments": 100, "compactness": 10.0, "sigma": 0.0}}, "sanity": {"same_saved_png_used_for_metrics": True, "epsilon_tensor_valid": all(float(r["actual_linf_tensor"]) <= EPS + 1e-6 for r in rows if r["valid"]), "masked_outside_serialized_delta_is_zero": all(float(r["mask_outside_delta_max"]) == 0 for r in rows if r["attack"] == "PGD+Mask" and r["valid"])}, "aggregate": {}}
    for kind in ("PGD", "PGD+Mask"):
        valid = [r for r in rows if r["attack"] == kind and r["valid"]]
        summary["aggregate"][kind] = {"n": len(valid), "arcface_asr": sum(r["arcface_threshold_crossing"] is True for r in valid)/len(valid), "facenet_asr": sum(r["facenet_transfer_crossing"] is True for r in valid)/len(valid), **{key: mean(valid, key) for key in ("arcface_cosine_drop", "facenet_cosine_drop", "actual_linf_serialized", "ssim", "lpips", "dists")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0 if all(r["valid"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
