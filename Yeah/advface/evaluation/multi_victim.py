"""Eligibility-aware verification summaries for paired multi-victim attacks."""
from __future__ import annotations

import math
import json
from pathlib import Path

MODELS = ("arcface", "facenet", "adaface", "mobilefacenet", "magface")


def load_v2_thresholds(root: Path) -> dict[str, float]:
    """Read the frozen FAR=1e-2 operating point for all five v2 victims."""
    values = {}
    for name in MODELS:
        path = root / f"results/calibration/{name}_lfw_v2/thresholds.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        point = raw["far_1e-2"]
        if point.get("target_far") != 0.01 or point.get("support_status") != "CALIBRATED":
            raise ValueError(f"invalid v2 calibration operating point: {path}")
        value = float(point["threshold"])
        if not math.isfinite(value):
            raise ValueError(f"nonfinite v2 threshold: {path}")
        values[name] = value
    return values


def summarize(rows: list[dict], thresholds: dict[str, float]) -> dict:
    if set(thresholds) != set(MODELS):
        raise ValueError("all five calibrated model thresholds are required")
    if len({r["identity_id"] for r in rows}) != len(rows):
        raise ValueError("identity rows must be distinct")
    for row in rows:
        for name in MODELS:
            value = row["models"][name]
            clean = value["clean_cosine"]
            if clean is None:
                eligible = False
                if not value.get("error"):
                    raise ValueError(f"missing clean cosine without an error for {name}")
            else:
                clean = float(clean)
                if not math.isfinite(clean):
                    raise ValueError(f"nonfinite clean cosine for {name}")
                eligible = clean >= thresholds[name]
            if value.get("eligible") != eligible:
                raise ValueError(f"incorrect clean eligibility for {name}")
            if "adv_cosine" in value:
                adv = float(value["adv_cosine"])
                if not math.isfinite(adv):
                    raise ValueError(f"nonfinite adversarial cosine for {name}")
                if value.get("success") != (eligible and adv < thresholds[name]):
                    raise ValueError(f"incorrect attack success for {name}")
    paired = [r for r in rows if all(r["models"][m]["eligible"] for m in MODELS)]
    result = {"original_denominator": len(rows), "joint_eligible": len(paired), "models": {}}
    for name in MODELS:
        valid_clean = [r for r in rows if r["models"][name]["clean_cosine"] is not None]
        eligible = [r for r in rows if r["models"][name]["eligible"]]
        attacked = [r for r in eligible if "adv_cosine" in r["models"][name]]
        paired_attacked = [r for r in paired if "adv_cosine" in r["models"][name]]
        result["models"][name] = {
            "clean_valid": len(valid_clean),
            "clean_errors": len(rows) - len(valid_clean),
            "clean_eligible": len(eligible),
            "clean_tar": len(eligible) / len(rows) if rows else None,
            "conditional_asr": sum(r["models"][name]["success"] for r in attacked) / len(attacked) if attacked else None,
            "conditional_denominator": len(attacked),
            "paired_asr": sum(r["models"][name]["success"] for r in paired_attacked) / len(paired_attacked) if paired_attacked else None,
            "paired_denominator": len(paired_attacked),
        }
    return result
