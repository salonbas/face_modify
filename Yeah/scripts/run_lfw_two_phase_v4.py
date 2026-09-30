#!/usr/bin/env python3
"""Two-phase, resumable formal LFW mask runner.

Attack workers are isolated because ArcFace Torch and landmark-mask resources can
grow over repeated attacks.  They write only serialized B_adv plus a small
generation record.  A separate evaluator keeps ArcFace, FaceNet, LPIPS, and
DISTS resident while evaluating every serialized artifact.

Existing formal records and PNGs are never modified.  Evaluation output is
append-only under ``evaluation_records``; ``results.csv`` is written only for
a method that does not already have one (the new MI-FGSM+Mask method).
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

EPS, STEPS = 0.04, 200
FIELDS = ("identity_id", "attack", "constraint", "reference_image", "probe_image", "adversarial_image", "configured_epsilon", "steps", "tv_weight", "actual_linf_tensor", "actual_linf_serialized", "arcface_clean_cosine", "arcface_adv_cosine", "arcface_cosine_drop", "arcface_threshold_crossing", "facenet_clean_cosine", "facenet_adv_cosine", "facenet_cosine_drop", "facenet_transfer_crossing", "ssim", "lpips", "dists", "mask_outside_delta_max", "runtime_seconds", "valid", "failure_reason")


def pairs() -> list[dict[str, str]]:
    with (ROOT / "data/datasets/lfw/evaluation_splits/attack_dev.csv").open(encoding="utf-8", newline="") as f:
        result = list(csv.DictReader(f))
    if len(result) != 20 or any(r["split"] != "attack_dev" for r in result):
        raise RuntimeError("frozen attack_dev.csv contract failed")
    return result


def method_dir(method: str) -> Path:
    name = "pgd" if method == "pgd_full" else method
    return ROOT / "results/transfer/lfw_attack_dev_20_four_method_v3" / f"{name}_landmark_superpixel_mask"


def adv_path(out: Path, ident: str, attack: str) -> Path:
    return out / "adversarial_probes" / f"{ident}_B_adv_{'pgd_mask' if attack == 'pgd_full' else 'mi_fgsm_mask'}.png"


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def existing_complete(out: Path, ident: str, attack: str) -> bool:
    # Old v3 worker records are authoritative for completed PGD+Mask.  New
    # generation records are used for MI-FGSM+Mask.  Never infer completion
    # from a bare PNG, which could be a partial/corrupt write.
    for record in (out / "records" / f"{ident}.json", out / "generation_records" / f"{ident}.json"):
        value = load_json(record)
        if value and value.get("valid") is True and (ROOT / value.get("adversarial_image", "")).is_file():
            return True
    return False


def worker(args: argparse.Namespace, pair: dict[str, str]) -> int:
    out, ident = Path(args.output_dir), pair["identity_id"]
    target = adv_path(out, ident, args.attack)
    record_path = out / "generation_records" / f"{ident}.json"
    if existing_complete(out, ident, args.attack):
        return 0
    if target.exists():
        raise RuntimeError(f"refusing to overwrite pre-existing artifact: {target}")
    row: dict[str, Any] = {"identity_id": ident, "attack": args.attack, "constraint": "landmark_superpixel_mask", "reference_image": pair["reference_image"], "probe_image": pair["probe_image"], "adversarial_image": str(target.relative_to(ROOT)), "configured_epsilon": EPS, "steps": STEPS, "tv_weight": 0.0, "valid": False, "failure_reason": ""}
    try:
        probe = load_bgr(ROOT / pair["probe_image"])
        # This process intentionally owns only attack-surrogate and landmark
        # resources.  It does not create FaceNet, LPIPS, or DISTS.
        app = create_face_app(det_size=DEFAULT_DET_SIZE)
        started = time.perf_counter()
        attack = ATTACKS[args.attack].apply(probe, app=app, config=AttackConfig(name=args.attack, eps=EPS, steps=STEPS, momentum=1.0, extra={"constraint": "landmark_superpixel_mask", "tv_weight": 0.0}))
        runtime = time.perf_counter() - started
        tensor_linf = float(attack.parameters["linf_tensor"])
        if tensor_linf > EPS + 1e-6:
            raise RuntimeError(f"tensor L∞ contract failed: {tensor_linf}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), attack.adversarial_bgr):
            raise RuntimeError("failed to serialize B_adv PNG")
        saved = load_bgr(target)
        final, _, _ = LandmarkSuperpixelMask().build(probe)
        outside = float(np.abs(saved.astype(np.int16) - probe.astype(np.int16)).max(axis=2)[final == 0].max(initial=0))
        if outside != 0:
            raise RuntimeError(f"serialized perturbation outside final mask: {outside}")
        row.update(actual_linf_tensor=tensor_linf, mask_outside_delta_max=outside, runtime_seconds=runtime, valid=True)
    except Exception as exc:
        row["failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        gc.collect()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if record_path.exists():
        raise RuntimeError(f"refusing to overwrite generation record: {record_path}")
    record_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if row["valid"] else 1


def tensor(rgb: np.ndarray):
    import torch
    value = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).float() / 255
    h, w = value.shape[-2:]
    if max(h, w) > 256:
        import torch.nn.functional as F
        value = F.interpolate(value, size=(round(h * 256 / max(h, w)), round(w * 256 / max(h, w))), mode="bilinear", align_corners=False)
    return value


def evaluation_source(out: Path, ident: str) -> dict[str, Any]:
    for candidate in (out / "generation_records" / f"{ident}.json", out / "records" / f"{ident}.json"):
        value = load_json(candidate)
        if value and value.get("valid") is True:
            return value
    raise RuntimeError(f"{out.name}/{ident}: no valid generation/formal record")


def evaluate(out: Path, attack: str, selected: list[dict[str, str]], *, overwrite_evaluation: bool = False, refresh_results: bool = False) -> None:
    """Evaluate in one process: metric model construction occurs exactly once."""
    import torch
    import lpips
    from DISTS_pytorch import DISTS
    src_thresholds, _ = load_thresholds(ROOT / "results/calibration/arcface_lfw_v1")
    victim_thresholds, _ = load_thresholds(ROOT / "results/calibration/facenet_lfw_v1")
    arcface = create_face_app(det_size=DEFAULT_DET_SIZE)
    facenet = load_embedder("facenet_vggface2")
    lpips_model, dists_model = lpips.LPIPS(net="alex", verbose=False).eval(), DISTS().eval()
    eval_dir = out / "evaluation_records"; eval_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    try:
        for pair in selected:
            ident, eval_path = pair["identity_id"], eval_dir / f"{pair['identity_id']}.json"
            if eval_path.exists() and not overwrite_evaluation:
                rows.append(json.loads(eval_path.read_text(encoding="utf-8"))); continue
            source = evaluation_source(out, ident)
            probe_path, saved_path = ROOT / pair["probe_image"], ROOT / source["adversarial_image"]
            clean_rgb, saved_rgb = np.asarray(Image.open(probe_path).convert("RGB"), dtype=np.uint8), np.asarray(Image.open(saved_path).convert("RGB"), dtype=np.uint8)
            a, b, adv = load_bgr(ROOT / pair["reference_image"]), load_bgr(probe_path), load_bgr(saved_path)
            ar, ab, ax = (get_embedding_from_bgr(arcface, image, f"{ident}:{label}") for image, label in ((a, "A"), (b, "B"), (adv, "B_adv")))
            vr, vb, vx = (facenet.get_embedding(image, label=f"{ident}:{label}") for image, label in ((a, "A"), (b, "B"), (adv, "B_adv")))
            ac, aa, vc, va = map(float, (cosine_similarity(ar, ab), cosine_similarity(ar, ax), cosine_similarity(vr, vb), cosine_similarity(vr, vx)))
            clean_t, saved_t = tensor(clean_rgb), tensor(saved_rgb)
            with torch.inference_mode():
                lp, ds = float(lpips_model(clean_t * 2 - 1, saved_t * 2 - 1).item()), float(dists_model(clean_t, saved_t).item())
            actual = float(np.abs(saved_rgb.astype(np.int16) - clean_rgb.astype(np.int16)).max() / 255)
            row = {field: source.get(field, "") for field in FIELDS}
            row.update(identity_id=ident, attack=attack, constraint="landmark_superpixel_mask", reference_image=pair["reference_image"], probe_image=pair["probe_image"], adversarial_image=source["adversarial_image"], configured_epsilon=EPS, steps=STEPS, tv_weight=0.0, actual_linf_serialized=actual, arcface_clean_cosine=ac, arcface_adv_cosine=aa, arcface_cosine_drop=ac-aa, arcface_threshold_crossing=ac >= src_thresholds[PRIMARY_OPERATING_POINT] and aa < src_thresholds[PRIMARY_OPERATING_POINT], facenet_clean_cosine=vc, facenet_adv_cosine=va, facenet_cosine_drop=vc-va, facenet_transfer_crossing=vc >= victim_thresholds[PRIMARY_OPERATING_POINT] and va < victim_thresholds[PRIMARY_OPERATING_POINT], ssim=float(structural_similarity(clean_rgb.astype(np.float32)/255, saved_rgb.astype(np.float32)/255, channel_axis=2, data_range=1.0)), lpips=lp, dists=ds, valid=True, failure_reason="")
            eval_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rows.append(row)
            del a, b, adv, ar, ab, ax, vr, vb, vx, clean_t, saved_t
            gc.collect()
    finally:
        del arcface, facenet, lpips_model, dists_model
        gc.collect()
    # Preserve formal results.csv; it is existing protocol evidence.  This
    # writes the missing MI-FGSM+Mask table used by the legacy aggregator.
    if not (out / "results.csv").exists() or refresh_results:
        with (out / "results.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)


def regression(out: Path, identities: list[str]) -> None:
    report: dict[str, Any] = {"passed": True, "identities": {}}
    for ident in identities:
        old, new = load_json(out / "records" / f"{ident}.json"), load_json(out / "evaluation_records" / f"{ident}.json")
        if not old or not new: raise RuntimeError(f"missing regression input for {ident}")
        deltas = {key: abs(float(old[key]) - float(new[key])) for key in ("actual_linf_serialized", "arcface_clean_cosine", "arcface_adv_cosine", "facenet_clean_cosine", "facenet_adv_cosine", "ssim", "lpips", "dists")}
        passed = all(v <= 1e-6 for v in deltas.values()) and all(bool(old[k]) == bool(new[k]) for k in ("arcface_threshold_crossing", "facenet_transfer_crossing"))
        report["identities"][ident] = {"passed": passed, "absolute_deltas": deltas}; report["passed"] &= passed
    (out / "evaluator_regression.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]: raise RuntimeError("evaluator regression mismatch")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--attack", choices=("pgd_full", "mi_fgsm"), required=True); p.add_argument("--phase", choices=("generate", "evaluate", "regression", "all"), default="all"); p.add_argument("--identity"); p.add_argument("--regression", action="store_true"); p.add_argument("--overwrite-evaluation", action="store_true"); p.add_argument("--refresh-results", action="store_true")
    args = p.parse_args(); out = method_dir(args.attack); all_pairs = pairs(); chosen = [x for x in all_pairs if not args.identity or x["identity_id"] == args.identity]
    if args.identity and not chosen: raise SystemExit("identity is not in frozen split")
    if args.identity: return worker(argparse.Namespace(**vars(args), output_dir=str(out)), chosen[0])
    if args.phase in ("generate", "all"):
        out.mkdir(parents=True, exist_ok=True)
        for pair in chosen:
            if existing_complete(out, pair["identity_id"], args.attack): print(f"resume: {pair['identity_id']} already complete"); continue
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--attack", args.attack, "--identity", pair["identity_id"]], cwd=ROOT, check=True)
    if args.refresh_results and args.attack != "mi_fgsm":
        raise SystemExit("--refresh-results is reserved for the MI-FGSM+Mask evaluator table")
    if args.phase in ("evaluate", "all"):
        evaluate(out, args.attack, chosen, overwrite_evaluation=args.overwrite_evaluation, refresh_results=args.refresh_results)
    if args.regression or args.phase == "regression": regression(out, [x["identity_id"] for x in chosen[:2]])
    return 0


if __name__ == "__main__": raise SystemExit(main())
