"""單一 benchmark experiment unit。"""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from advface.attacks.registry import get_attack_spec
from advface.benchmark.failures import (
    INVALID_EMBEDDING,
    INVALID_LINF,
    MULTIPLE_FACE_AMBIGUOUS,
    NO_FACE_DETECTED,
    VICTIM_EXCEPTION,
    BenchmarkFailure,
    classify_attack_exception,
    classify_exception,
)
from advface.config import (
    PIXEL_MAX,
    SIMILARITY_THRESHOLD,
    SURROGATE_THRESHOLD_STATUS,
    VICTIM_THRESHOLD_STATUS,
)
from advface.evaluation.perturbation import perturbation_metrics as _perturbation_metrics
from advface.evaluation.perturbation import validate_linf
from advface.experiments.runner import run_experiment
from advface.image_io import load_bgr
from advface.models.insightface_app import InsightFaceEmbedder


def perturbation_metrics(original_bgr: np.ndarray, adversarial_bgr: np.ndarray) -> tuple[float, float]:
    m = _perturbation_metrics(original_bgr, adversarial_bgr)
    return m["linf"], m["l2_pixel"]


def inspect_surrogate_faces(embedder: InsightFaceEmbedder, img_bgr: np.ndarray, label: str) -> None:
    faces = embedder.app.get(img_bgr)
    if not faces:
        raise BenchmarkFailure(NO_FACE_DETECTED, f"沒有偵測到人臉：{label}")
    if len(faces) > 1:
        raise BenchmarkFailure(
            MULTIPLE_FACE_AMBIGUOUS,
            f"偵測到 {len(faces)} 張臉：{label}",
        )


def empty_result_row(unit, *, run_id: str, seed: int, device: str) -> dict[str, Any]:
    attack = unit.attack
    steps = int(unit.steps)
    eps = float(unit.eps)
    spec = get_attack_spec(attack)
    alpha_px = None if not spec.requires_steps else (eps * PIXEL_MAX) / max(steps, 1)
    return {
        "unit_key": unit.key,
        "image_id": unit.image_id,
        "image_path": unit.image_path,
        "identity_id": unit.identity_id,
        "source": unit.source,
        "attack": attack,
        "eps": eps,
        "steps": steps,
        "alpha_px": alpha_px,
        "random_start": False,
        "seed": int(seed),
        "surrogate_model": "insightface_buffalo_l",
        "victim_model": "facenet_vggface2",
        "run_id": run_id,
        "device": device,
        "status": "failed",
        "failure_code": "",
        "failure_message": "",
        "runtime_sec": None,
        "linf": None,
        "l2_pixel": None,
        "linf_ok": None,
        "surrogate_cosine": None,
        "surrogate_euclidean": None,
        "surrogate_threshold": float(SIMILARITY_THRESHOLD),
        "threshold_status_surrogate": SURROGATE_THRESHOLD_STATUS,
        "whitebox_success": None,
        "victim_cosine": None,
        "victim_euclidean": None,
        "victim_cosine_drop": None,
        "victim_threshold": None,
        "threshold_status_victim": VICTIM_THRESHOLD_STATUS,
    }


def run_experiment_unit(
    unit,
    *,
    original_bgr: Optional[np.ndarray],
    surrogate: InsightFaceEmbedder,
    victim,
    device: str,
    seed: int,
    run_id: str,
    save_image: bool = False,
) -> tuple[dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    row = empty_result_row(unit, run_id=run_id, seed=seed, device=device)
    t0 = time.perf_counter()
    adv_bgr = None
    orig = original_bgr
    try:
        if orig is None:
            orig = load_bgr(unit.image_path)
        inspect_surrogate_faces(surrogate, orig, unit.image_id)
        result = run_experiment(
            orig,
            attack=unit.attack,
            surrogate=surrogate,
            victims={"victim": victim},
            eps=unit.eps,
            steps=unit.steps,
            device=device,
            seed=seed,
            app=surrogate.app,
        )
        adv_bgr = result.adversarial_bgr
        spec = get_attack_spec(unit.attack)
        linf = float(result.perturbation["linf"])
        l2_pixel = float(result.perturbation["l2_pixel"])
        row["linf"] = linf
        row["l2_pixel"] = l2_pixel
        row["linf_constraint_domain"] = spec.linf_constraint_domain
        if spec.linf_constraint_domain == "aligned_crop_paste":
            linf_ok = True
            row["linf_note"] = (
                "full-image L∞ may exceed eps due to affine paste in canonical FGSM"
            )
        else:
            linf_ok = validate_linf(linf, unit.eps)
            row["linf_note"] = ""
        row["linf_ok"] = bool(linf_ok)

        sur = result.surrogate
        vic = result.primary_victim
        if vic is None:
            raise RuntimeError("canonical runner 未回傳 victim 評估")
        if sur.error:
            row["failure_code"] = INVALID_EMBEDDING
            row["failure_message"] = str(sur.error)
            row["status"] = "failed"
            row["runtime_sec"] = time.perf_counter() - t0
            return row, orig, adv_bgr

        row["surrogate_cosine"] = float(sur.cosine_after)
        row["surrogate_euclidean"] = float(sur.euclidean_distance) if sur.euclidean_distance is not None else None
        row["whitebox_success"] = bool(sur.success)

        if vic.error:
            row["failure_code"] = VICTIM_EXCEPTION
            row["failure_message"] = str(vic.error)
            row["status"] = "failed"
            row["runtime_sec"] = time.perf_counter() - t0
            return row, orig, adv_bgr

        row["victim_cosine"] = float(vic.cosine_after)
        row["victim_euclidean"] = float(vic.euclidean_distance) if vic.euclidean_distance is not None else None
        # Level A：1 - cosine(original, adversarial)
        row["victim_cosine_drop"] = float(1.0 - vic.cosine_after)

        if not linf_ok:
            row["status"] = "invalid"
            row["failure_code"] = INVALID_LINF
            row["failure_message"] = f"linf={linf:.6f} > eps={unit.eps} + tol"
        else:
            row["status"] = "completed"
            row["failure_code"] = ""
            row["failure_message"] = ""
        row["runtime_sec"] = time.perf_counter() - t0
        if not save_image:
            pass
        return row, orig, adv_bgr
    except BenchmarkFailure as e:
        row["status"] = "failed"
        row["failure_code"] = e.code
        row["failure_message"] = e.message
        row["runtime_sec"] = time.perf_counter() - t0
        return row, orig, adv_bgr
    except Exception as e:
        row["status"] = "failed"
        if orig is None:
            row["failure_code"] = classify_exception(e)
        else:
            row["failure_code"] = classify_attack_exception(e)
        row["failure_message"] = f"{type(e).__name__}: {e}"
        row["runtime_sec"] = time.perf_counter() - t0
        return row, orig, adv_bgr
