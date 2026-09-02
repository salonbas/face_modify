#!/usr/bin/env python3
"""Evaluate existing PGD Full on frozen LFW verification development pairs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from advface.attacks.registry import ATTACKS, AttackConfig
from advface.config import DEFAULT_DET_SIZE, project_root
from advface.evaluation.perturbation import perturbation_metrics, validate_linf
from advface.evaluation.verification import PRIMARY_OPERATING_POINT, evaluate_verification_pair, load_thresholds
from advface.image_io import load_bgr
from advface.models.insightface_app import create_face_app, get_embedding_from_bgr

RESULT_FIELDS = (
    "identity_id", "reference_image", "probe_image", "clean_cosine_A_B", "clean_euclidean_A_B",
    "self_cosine_B_Bmodified", "self_euclidean_B_Bmodified", "modified_cosine_A_Bmodified",
    "modified_euclidean_A_Bmodified", "self_similarity_change", "verification_similarity_change",
    "primary_threshold", "clean_pair_valid", "crossed_primary_boundary", "success_at_eer_threshold",
    "success_at_far_1e-3_exploratory", "linf_tensor", "linf_serialized", "linf_serialized_valid",
    "runtime_seconds", "perturbation_configuration", "valid", "failure_reason",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # This v1 executable intentionally cannot run the frozen held-out set.
    parser.add_argument("--split", choices=["development"], default="development")
    parser.add_argument("--attack", choices=["pgd_full"], default="pgd_full")
    parser.add_argument("--max-identities", type=int, default=5)
    parser.add_argument("--eps", type=float, default=0.04)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--threshold-source", type=Path, default=None)
    parser.add_argument("--run-name", default="lfw_verification_pgd_v1")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="one identity, up to five PGD steps; integration only")
    return parser.parse_args()


def value(row: dict[str, Any], key: str) -> float:
    raw = row.get(key, "")
    return float(raw) if raw not in ("", None) else float("nan")


def print_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    counts, aggregate = summary["counts"], summary["aggregate"]
    print("=" * 40); print("LFW Verification Robustness Evaluation"); print("=" * 40)
    print("Calibration:")
    print(f"threshold source = {summary['calibration']['source']}")
    print(f"operating point = {PRIMARY_OPERATING_POINT}")
    print(f"primary threshold = {summary['calibration']['primary_threshold']:.6f}")
    print("Dataset:")
    for key in ("development", "test", "reserve"):
        print(f"{key} identities = {summary['split_counts'][key]}")
    print("Evaluated development pairs:")
    print(f"attempted = {counts['attempted']}\nclean-valid = {counts['clean_valid']}\ninvalid = {counts['clean_invalid']}\nfailures = {counts['failures']}")
    print("For each evaluated pair:")
    for row in rows:
        if row["valid"]:
            print(f"{row['identity_id']}: A-B={value(row, 'clean_cosine_A_B'):.6f} "
                  f"B-B_modified={value(row, 'self_cosine_B_Bmodified'):.6f} "
                  f"A-B_modified={value(row, 'modified_cosine_A_Bmodified'):.6f} "
                  f"change={value(row, 'verification_similarity_change'):.6f} "
                  f"crossed={row['crossed_primary_boundary']}")
        else:
            print(f"{row['identity_id']}: failure={row['failure_reason']}")
    print("Aggregate:")
    for key in ("mean_clean_verification_cosine", "mean_modified_verification_cosine", "mean_verification_similarity_change", "median_verification_similarity_change"):
        print(f"{key} = {aggregate[key]}")
    print("boundary crossing:")
    print(f"{aggregate['boundary_crossing_count']} / {counts['clean_valid']}")
    print(f"rate = {aggregate['boundary_crossing_rate']}")
    print("=" * 40)


def main() -> int:
    args = parse_args()
    if args.max_identities < 1:
        raise SystemExit("--max-identities must be positive")
    if args.smoke:
        args.max_identities = 1
        args.steps = min(args.steps, 5)
        if args.run_name == "lfw_verification_pgd_v1":
            args.run_name = "lfw_verification_pgd_smoke"
    root = project_root()
    split_dir = root / "data/datasets/lfw/evaluation_splits"
    split_rows = read_csv(split_dir / f"{args.split}.csv")
    selected = split_rows[:args.max_identities]
    if len(selected) != args.max_identities:
        raise SystemExit(f"requested {args.max_identities} identities but split has {len(selected)}")
    threshold_source = args.threshold_source or root / "results/calibration/arcface_lfw_v1"
    thresholds, threshold_artifact = load_thresholds(threshold_source)
    out = args.output_dir or root / "results/robustness" / args.run_name
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty run directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    app = create_face_app(det_size=DEFAULT_DET_SIZE)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for pair in selected:
        row: dict[str, Any] = {field: "" for field in RESULT_FIELDS}
        row.update({"identity_id": pair["identity_id"], "reference_image": pair["reference_image"],
                    "probe_image": pair["probe_image"], "valid": False, "failure_reason": ""})
        try:
            reference = load_bgr(root / pair["reference_image"])
            probe = load_bgr(root / pair["probe_image"])
            ref_emb = get_embedding_from_bgr(app, reference, f"{pair['identity_id']}:A")
            probe_emb = get_embedding_from_bgr(app, probe, f"{pair['identity_id']}:B")
            clean = evaluate_verification_pair(ref_emb, probe_emb, probe_emb, thresholds)
            row.update(clean)
            if not row["clean_pair_valid"]:
                row.update({"valid": True, "failure_reason": "clean_pair_invalid_no_modification"})
                results.append(row)
                continue
            started = time.perf_counter()
            output = ATTACKS[args.attack].apply(probe, app=app, config=AttackConfig(name=args.attack, eps=args.eps, steps=args.steps))
            runtime = time.perf_counter() - started
            modified_emb = get_embedding_from_bgr(app, output.adversarial_bgr, f"{pair['identity_id']}:B_modified")
            row.update(evaluate_verification_pair(ref_emb, probe_emb, modified_emb, thresholds))
            serialized = perturbation_metrics(probe, output.adversarial_bgr)["linf"]
            row.update({"linf_tensor": output.parameters.get("linf_tensor"), "linf_serialized": serialized,
                        "linf_serialized_valid": validate_linf(serialized, args.eps), "runtime_seconds": runtime,
                        "perturbation_configuration": json.dumps(output.parameters, sort_keys=True), "valid": True})
        except Exception as exc:
            row["failure_reason"] = f"{type(exc).__name__}: {exc}"
            failures.append({"identity_id": pair["identity_id"], "reference_image": pair["reference_image"], "probe_image": pair["probe_image"], "failure_reason": row["failure_reason"]})
        results.append(row)
    valid = [row for row in results if row["valid"]]
    clean_valid = [row for row in valid if row.get("clean_pair_valid") is True]
    def mean_metric(name: str) -> float | None:
        values = [value(row, name) for row in clean_valid]
        return float(np.mean(values)) if values else None
    changes = [value(row, "verification_similarity_change") for row in clean_valid]
    crossings = sum(bool(row.get("crossed_primary_boundary")) for row in clean_valid)
    split_counts = {name: len(read_csv(split_dir / f"{name}.csv")) for name in ("development", "test", "reserve")}
    summary = {"title": "LFW Face-Verification Robustness Evaluation v1", "scope": "offline robustness measurement; small development capability observation only",
               "calibration": {"source": str(threshold_artifact), "operating_point": PRIMARY_OPERATING_POINT,
                               "primary_threshold": thresholds[PRIMARY_OPERATING_POINT], "eer_threshold": thresholds["eer"],
                               "far_1e-3_exploratory_threshold": thresholds["far_1e-3"], "historical_0.4_threshold": thresholds["legacy_0.4"]},
               "split_counts": split_counts, "counts": {"attempted": len(results), "clean_valid": len(clean_valid),
               "clean_invalid": len(valid) - len(clean_valid), "failures": len(failures)},
               "aggregate": {"mean_clean_verification_cosine": mean_metric("clean_cosine_A_B"),
               "mean_modified_verification_cosine": mean_metric("modified_cosine_A_Bmodified"),
               "mean_verification_similarity_change": mean_metric("verification_similarity_change"),
               "median_verification_similarity_change": float(np.median(changes)) if changes else None,
               "boundary_crossing_count": crossings, "boundary_crossing_rate": crossings / len(clean_valid) if clean_valid else None}}
    config = {"split": args.split, "max_identities": args.max_identities, "attack": args.attack, "eps": args.eps,
              "steps": args.steps, "smoke": args.smoke, "threshold_source": str(threshold_artifact),
              "perturbation_objective": "existing PGD Full self-representation objective; unchanged"}
    write_csv(out / "results.csv", results, RESULT_FIELDS)
    write_csv(out / "failures.csv", failures, ("identity_id", "reference_image", "probe_image", "failure_reason"))
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print_summary(summary, results)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
