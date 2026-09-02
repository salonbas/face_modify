"""Empirical face-verification threshold calibration helpers.

The decision convention throughout this module is deliberately explicit:
``score >= threshold`` accepts a pair as the same identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def labels_from_pairs(rows: Iterable[dict[str, object]]) -> np.ndarray:
    """Return genuine-positive labels and reject malformed pair labels."""
    labels: list[bool] = []
    for row in rows:
        value = row["same_identity"]
        if isinstance(value, bool):
            labels.append(value)
        elif str(value).strip().lower() in {"true", "false"}:
            labels.append(str(value).strip().lower() == "true")
        else:
            raise ValueError(f"invalid same_identity value: {value!r}")
    return np.asarray(labels, dtype=bool)


def operating_metrics(scores: np.ndarray, genuine: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Metrics for the acceptance rule score >= threshold."""
    scores = np.asarray(scores, dtype=float)
    genuine = np.asarray(genuine, dtype=bool)
    if scores.shape != genuine.shape or scores.ndim != 1:
        raise ValueError("scores and genuine labels must be matching 1-D arrays")
    positives = int(genuine.sum())
    negatives = int((~genuine).sum())
    if not positives or not negatives:
        raise ValueError("both genuine and impostor pairs are required")
    accepted = scores >= threshold
    tp = int(np.logical_and(accepted, genuine).sum())
    fp = int(np.logical_and(accepted, ~genuine).sum())
    fn = positives - tp
    tn = negatives - fp
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "far": fp / negatives, "fpr": fp / negatives,
        "tar": tp / positives, "tpr": tp / positives,
        "frr": fn / positives, "fnr": fn / positives,
        "accuracy": (tp + tn) / len(scores),
    }


def roc_rows(scores: np.ndarray, genuine: np.ndarray) -> list[dict[str, float | int]]:
    """Return all empirical ROC operating points, including reject-all."""
    scores = np.asarray(scores, dtype=float)
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1]]
    return [operating_metrics(scores, genuine, float(t)) for t in thresholds]


def eer(scores: np.ndarray, genuine: np.ndarray) -> dict[str, float | int]:
    """Discrete EER: select the ROC point minimizing |FAR-FRR|, deterministically."""
    rows = roc_rows(scores, genuine)
    # The tuple makes ties reproducible: lower error sum, then higher threshold.
    best = min(rows, key=lambda row: (abs(float(row["far"]) - float(row["frr"])), float(row["far"]) + float(row["frr"]), -float(row["threshold"])))
    return {**best, "eer": (float(best["far"]) + float(best["frr"])) / 2}


def far_support_status(impostor_count: int, target_far: float, minimum_expected_false_accepts: float = 5.0) -> str:
    if impostor_count <= 0:
        return "UNSUPPORTED"
    if impostor_count * target_far >= minimum_expected_false_accepts:
        return "CALIBRATED"
    return "EXPLORATORY"


def select_far_threshold(impostor_scores: np.ndarray, target_far: float) -> dict[str, float | int | str]:
    """Conservatively choose a threshold with empirical FAR <= target_far.

    The selected boundary is just above the score which would create the
    (floor(alpha*N)+1)-th false accept. This handles score ties correctly.
    """
    values = np.asarray(impostor_scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("impostor_scores must be a non-empty finite 1-D array")
    if not 0 < target_far < 1:
        raise ValueError("target_far must be in (0, 1)")
    allowed = int(np.floor(target_far * len(values)))
    descending = np.sort(values)[::-1]
    threshold = float(np.nextafter(descending[allowed], np.inf)) if allowed < len(values) else float(-np.inf)
    actual_far = float(np.mean(values >= threshold))
    return {
        "threshold": threshold, "target_far": float(target_far), "dev_far": actual_far,
        "impostor_count": int(len(values)), "far_resolution": 1.0 / len(values),
        "allowed_false_accepts": allowed,
        "support_status": far_support_status(len(values), target_far),
        "selection_method": "conservative empirical order statistic: threshold just above the (floor(alpha*N)+1)-th highest impostor score",
    }


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values)),
        "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values)),
        **{f"p{p:02d}": float(np.percentile(values, p)) for p in (1, 5, 25, 75, 95, 99)},
    }
