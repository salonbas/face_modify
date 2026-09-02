"""Evaluation helpers for calibrated same-identity verification attacks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from advface.evaluation.similarity import cosine_similarity, euclidean_distance


PRIMARY_OPERATING_POINT = "far_1e-2"


def load_thresholds(source: str | Path) -> tuple[dict[str, float], Path]:
    """Load the immutable calibration artifact; never recalibrate here."""
    path = Path(source)
    artifact = path if path.name == "thresholds.json" else path / "thresholds.json"
    if not artifact.is_file():
        raise FileNotFoundError(f"calibration thresholds artifact not found: {artifact}")
    raw = json.loads(artifact.read_text(encoding="utf-8"))
    required = (PRIMARY_OPERATING_POINT, "eer", "far_1e-3")
    missing = [name for name in required if name not in raw or "threshold" not in raw[name]]
    if missing:
        raise ValueError(f"invalid calibration artifact; missing operating points: {', '.join(missing)}")
    values = {name: float(raw[name]["threshold"]) for name in required}
    summary = artifact.parent / "summary.json"
    if not summary.is_file():
        raise FileNotFoundError(f"calibration summary artifact not found: {summary}")
    legacy = json.loads(summary.read_text(encoding="utf-8"))["operating_points"]["legacy_0.4"]["threshold"]
    values["legacy_0.4"] = float(legacy)
    return values, artifact.resolve()


def evaluate_verification_pair(
    reference_embedding: np.ndarray,
    probe_embedding: np.ndarray,
    adversarial_embedding: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, float | bool]:
    """Score A/B, B/B_adv, and A/B_adv with the frozen decision convention."""
    clean = cosine_similarity(reference_embedding, probe_embedding)
    self_adv = cosine_similarity(probe_embedding, adversarial_embedding)
    verification_adv = cosine_similarity(reference_embedding, adversarial_embedding)
    primary = float(thresholds[PRIMARY_OPERATING_POINT])
    clean_pass = clean >= primary
    return {
        "clean_cosine_A_B": clean,
        "clean_euclidean_A_B": euclidean_distance(reference_embedding, probe_embedding),
        "self_cosine_B_Bmodified": self_adv,
        "self_euclidean_B_Bmodified": euclidean_distance(probe_embedding, adversarial_embedding),
        "modified_cosine_A_Bmodified": verification_adv,
        "modified_euclidean_A_Bmodified": euclidean_distance(reference_embedding, adversarial_embedding),
        "self_similarity_change": 1.0 - self_adv,
        "verification_similarity_change": clean - verification_adv,
        "primary_threshold": primary,
        "clean_pair_valid": clean_pass,
        "crossed_primary_boundary": clean_pass and verification_adv < primary,
        "success_at_eer_threshold": clean >= thresholds["eer"] and verification_adv < thresholds["eer"],
        "success_at_far_1e-3_exploratory": clean >= thresholds["far_1e-3"] and verification_adv < thresholds["far_1e-3"],
    }
