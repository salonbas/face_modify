"""Benchmark 彙總統計（含 ALL ATTEMPTS 與 WHITE-BOX SUCCESSFUL ONLY）。"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

import numpy as np


def _floats(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for r in rows:
        v = r.get(key)
        if v is None or v == "":
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if x == x:
            out.append(x)
    return out


def _stat_block(xs: list[float]) -> dict[str, Optional[float]]:
    if not xs:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def _flatten_stats(prefix: str, block: dict[str, Optional[float]]) -> dict[str, Optional[float]]:
    return {f"{prefix}_{k}": v for k, v in block.items()}


def is_valid_row(row: dict[str, Any]) -> bool:
    return str(row.get("status")) == "completed" and bool(row.get("linf_ok", False))


def is_whitebox_success(row: dict[str, Any]) -> bool:
    return is_valid_row(row) and bool(row.get("whitebox_success"))


def config_key(row: dict[str, Any]) -> tuple:
    attack = str(row.get("attack"))
    eps = float(row.get("eps"))
    steps = int(row.get("steps") or 0)
    return (attack, eps, steps)


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_attempted = len(rows)
    valid = [r for r in rows if is_valid_row(r)]
    n_valid = len(valid)
    n_wb = sum(1 for r in valid if bool(r.get("whitebox_success")))
    rate = (n_wb / n_valid) if n_valid else None
    out: dict[str, Any] = {
        "n_attempted": n_attempted,
        "n_valid": n_valid,
        "n_whitebox_success": n_wb,
        "whitebox_success_rate": rate,
    }
    out.update(_flatten_stats("surrogate_cosine", _stat_block(_floats(valid, "surrogate_cosine"))))
    out.update(_flatten_stats("victim_cosine", _stat_block(_floats(valid, "victim_cosine"))))
    out.update(_flatten_stats("victim_cosine_drop", _stat_block(_floats(valid, "victim_cosine_drop"))))
    out.update(_flatten_stats("surrogate_euclidean", _stat_block(_floats(valid, "surrogate_euclidean"))))
    out.update(_flatten_stats("victim_euclidean", _stat_block(_floats(valid, "victim_euclidean"))))
    out.update(_flatten_stats("linf", _stat_block(_floats(valid, "linf"))))
    out.update(_flatten_stats("runtime", _stat_block(_floats(rows, "runtime_sec"))))
    return out


def summarize_config_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_stats = summarize_subset(rows)
    wb_rows = [r for r in rows if is_whitebox_success(r)]
    wb_stats = summarize_subset(wb_rows)
    attack, eps, steps = config_key(rows[0]) if rows else ("", 0.0, 0)
    payload: dict[str, Any] = {
        "attack": attack,
        "eps": eps,
        "steps": steps,
        "all": all_stats,
        "whitebox_success_only": wb_stats,
    }
    # flat columns for CSV
    for k, v in all_stats.items():
        payload[f"all_{k}"] = v
    for k, v in wb_stats.items():
        payload[f"wb_{k}"] = v
    return payload


def _qualitative_drop(mean_drop: Optional[float]) -> str:
    if mean_drop is None:
        return "unknown"
    x = float(mean_drop)
    if x < 0.05:
        return "weak"
    if x < 0.20:
        return "moderate"
    return "strong"


def build_summary(
    rows: list[dict[str, Any]],
    *,
    n_images: int,
    n_planned: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    valid = [r for r in rows if is_valid_row(r)]
    failed = [r for r in rows if str(r.get("status")) == "failed"]
    invalid = [r for r in rows if str(r.get("status")) == "invalid"]
    codes = Counter(str(r.get("failure_code") or "OTHER") for r in failed + invalid)

    by_cfg: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cfg[config_key(r)].append(r)
    per_config = [summarize_config_rows(v) for _, v in sorted(by_cfg.items())]

    strongest = None
    weakest = None
    for c in per_config:
        drop = c["all"].get("victim_cosine_drop_mean")
        if drop is None:
            continue
        item = {
            "attack": c["attack"],
            "eps": c["eps"],
            "steps": c["steps"],
            "victim_cosine_drop_mean": drop,
            "whitebox_success_rate": c["all"].get("whitebox_success_rate"),
            "n_valid": c["all"].get("n_valid"),
        }
        if strongest is None or drop > strongest["victim_cosine_drop_mean"]:
            strongest = item
        if weakest is None or drop < weakest["victim_cosine_drop_mean"]:
            weakest = item

    overall_drop = summarize_subset(rows)["victim_cosine_drop_mean"]
    overall_wb = summarize_subset(rows)["whitebox_success_rate"]

    distributions = {
        "surrogate_cosine": _floats(valid, "surrogate_cosine"),
        "victim_cosine": _floats(valid, "victim_cosine"),
        "victim_cosine_drop": _floats(valid, "victim_cosine_drop"),
        "victim_euclidean": _floats(valid, "victim_euclidean"),
        "surrogate_euclidean": _floats(valid, "surrogate_euclidean"),
    }

    example_keys = _select_example_keys(valid, failed)

    executive = {
        "dataset_size": n_images,
        "n_planned": n_planned,
        "n_completed_rows": sum(1 for r in rows if str(r.get("status")) in ("completed", "invalid", "failed")),
        "n_valid": len(valid),
        "n_failed": len(failed),
        "n_invalid": len(invalid),
        "n_attack_configurations": len(per_config),
        "surrogate": config.get("surrogate_model", "insightface_buffalo_l"),
        "victim": config.get("victim_model", "facenet_vggface2"),
        "overall_whitebox_success_rate": overall_wb,
        "victim_mean_cosine_drop": overall_drop,
        "victim_transfer_effect_qualitative": _qualitative_drop(overall_drop),
        "strongest_transfer_configuration": strongest,
        "weakest_transfer_configuration": weakest,
        "transfer_success_rate": None,
        "transfer_success_rate_display": "N/A — victim threshold not calibrated",
    }

    return {
        "executive": executive,
        "per_config": per_config,
        "failure_counts": dict(codes),
        "distributions": distributions,
        "example_keys": example_keys,
        "limitations": [
            "Single surrogate model (InsightFace / ArcFace buffalo_l).",
            "Single victim model (FaceNet VGGFace2).",
            "Victim threshold not calibrated; no binary transfer success rate.",
            "No targeted attack.",
            "No mapper.",
            "No query-based attack.",
            "Dataset scope is limited to the current benchmark manifest.",
            "Surrogate cosine threshold 0.4 is provisional.",
            "Results describe observed transfer effect under this dataset, model pair, and attack grid only.",
        ],
        "research_question": (
            "Vanilla white-box adversarial attacks on a larger face sample: "
            "do they transfer across models, and how does the effect change with "
            "attack type, epsilon, and PGD steps?"
        ),
    }


def _select_example_keys(valid: list[dict[str, Any]], failed: list[dict[str, Any]]) -> dict[str, Any]:
    def _best(rows, key, reverse=False):
        xs = [r for r in rows if r.get(key) is not None]
        if not xs:
            return None
        xs = sorted(xs, key=lambda r: float(r[key]), reverse=reverse)
        return xs[0].get("unit_key")

    median_key = None
    drops = [(float(r["victim_cosine_drop"]), r.get("unit_key")) for r in valid if r.get("victim_cosine_drop") is not None]
    if drops:
        drops.sort(key=lambda t: t[0])
        median_key = drops[len(drops) // 2][1]

    fail_key = failed[0].get("unit_key") if failed else None
    return {
        "strongest_surrogate_attack": _best(valid, "surrogate_cosine", reverse=False),
        "weakest_surrogate_attack": _best(valid, "surrogate_cosine", reverse=True),
        "strongest_victim_effect": _best(valid, "victim_cosine_drop", reverse=True),
        "weakest_victim_effect": _best(valid, "victim_cosine_drop", reverse=False),
        "median_victim_effect": median_key,
        "selected_failure": fail_key,
    }


SUMMARY_CSV_FIELDS = [
    "attack",
    "eps",
    "steps",
    "all_n_attempted",
    "all_n_valid",
    "all_n_whitebox_success",
    "all_whitebox_success_rate",
    "all_surrogate_cosine_mean",
    "all_surrogate_cosine_median",
    "all_surrogate_cosine_p25",
    "all_surrogate_cosine_p75",
    "all_victim_cosine_mean",
    "all_victim_cosine_median",
    "all_victim_cosine_p25",
    "all_victim_cosine_p75",
    "all_victim_cosine_drop_mean",
    "all_victim_cosine_drop_median",
    "all_surrogate_euclidean_mean",
    "all_victim_euclidean_mean",
    "all_linf_mean",
    "all_runtime_mean",
    "all_surrogate_cosine_p05",
    "all_surrogate_cosine_p95",
    "all_surrogate_cosine_std",
    "wb_n_attempted",
    "wb_n_valid",
    "wb_victim_cosine_drop_mean",
    "wb_victim_cosine_drop_median",
    "wb_victim_cosine_mean",
    "wb_victim_euclidean_mean",
]
