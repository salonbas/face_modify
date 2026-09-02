#!/usr/bin/env python3
"""Run the frozen five-pair MI-FGSM ArcFace -> FaceNet verification check.

FaceNet is instantiated only after the ArcFace attack finishes and is used only
for embeddings.  This runner deliberately exposes no held-out split or tuning
arguments.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from advface.attacks.registry import ATTACKS, AttackConfig
from advface.config import DEFAULT_DET_SIZE, project_root
from advface.evaluation.similarity import cosine_similarity, euclidean_distance
from advface.evaluation.verification import PRIMARY_OPERATING_POINT, load_thresholds
from advface.image_io import load_bgr
from advface.models import load_embedder
from advface.models.insightface_app import create_face_app, get_embedding_from_bgr

EPS = 0.04
STEPS = 200
MOMENTUM = 1.0
TOLERANCE = 1e-6
RESULT_FIELDS = (
    "identity_id", "reference_image", "probe_image", "adversarial_image",
    "source_clean_cosine", "source_clean_euclidean", "source_self_adv_cosine", "source_self_adv_euclidean",
    "source_verification_adv_cosine", "source_verification_adv_euclidean", "source_verification_drop",
    "source_threshold", "source_boundary_crossed",
    "victim_clean_cosine", "victim_clean_euclidean", "victim_self_adv_cosine", "victim_self_adv_euclidean",
    "victim_self_similarity_drop", "victim_verification_adv_cosine", "victim_verification_adv_euclidean",
    "victim_verification_drop", "victim_threshold", "victim_clean_valid", "victim_boundary_crossed",
    "transfer_boundary_crossed", "linf_tensor", "linf_serialized", "linf_tensor_valid", "runtime_seconds",
    "attack_parameters", "valid", "failure_reason",
)
COMPARISON_FIELDS = (
    "identity_id", "pgd_source_self_adv_cosine", "pgd_source_A_B_adv", "pgd_source_boundary_crossed",
    "pgd_victim_self_adv_cosine", "pgd_victim_A_B_adv", "pgd_victim_verification_drop", "pgd_victim_boundary_crossed",
    "mi_fgsm_source_self_adv_cosine", "mi_fgsm_source_A_B_adv", "mi_fgsm_source_boundary_crossed",
    "mi_fgsm_victim_self_adv_cosine", "mi_fgsm_victim_A_B_adv", "mi_fgsm_victim_verification_drop", "mi_fgsm_victim_boundary_crossed",
    "delta_source_A_B_adv", "delta_victim_self_cosine", "delta_victim_A_B_adv", "delta_victim_verification_drop",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    return float(cosine_similarity(a, b)), float(euclidean_distance(a, b))


def number(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def average(rows: list[dict[str, Any]], key: str) -> float | None:
    return float(np.mean([number(row, key) for row in rows])) if rows else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["development"], default="development")
    parser.add_argument("--smoke", action="store_true", help="one identity and at most five steps; integration only")
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def comparison_rows(pgd: dict[str, dict[str, str]], mi_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mi in mi_rows:
        if not mi["valid"]:
            continue
        old = pgd[mi["identity_id"]]
        row = {
            "identity_id": mi["identity_id"],
            "pgd_source_self_adv_cosine": old["source_self_B_Badv"], "pgd_source_A_B_adv": old["source_adv_A_Badv"],
            "pgd_source_boundary_crossed": old["source_boundary_crossed"], "pgd_victim_self_adv_cosine": old["victim_self_B_Badv"],
            "pgd_victim_A_B_adv": old["victim_adv_A_Badv"],
            "pgd_victim_verification_drop": number(old, "victim_clean_A_B") - number(old, "victim_adv_A_Badv"),
            "pgd_victim_boundary_crossed": old["victim_boundary_crossed"],
            "mi_fgsm_source_self_adv_cosine": mi["source_self_adv_cosine"], "mi_fgsm_source_A_B_adv": mi["source_verification_adv_cosine"],
            "mi_fgsm_source_boundary_crossed": mi["source_boundary_crossed"], "mi_fgsm_victim_self_adv_cosine": mi["victim_self_adv_cosine"],
            "mi_fgsm_victim_A_B_adv": mi["victim_verification_adv_cosine"],
            "mi_fgsm_victim_verification_drop": mi["victim_verification_drop"], "mi_fgsm_victim_boundary_crossed": mi["victim_boundary_crossed"],
        }
        row.update(delta_source_A_B_adv=number(row, "mi_fgsm_source_A_B_adv") - number(row, "pgd_source_A_B_adv"),
                   delta_victim_self_cosine=number(row, "mi_fgsm_victim_self_adv_cosine") - number(row, "pgd_victim_self_adv_cosine"),
                   delta_victim_A_B_adv=number(row, "mi_fgsm_victim_A_B_adv") - number(row, "pgd_victim_A_B_adv"),
                   delta_victim_verification_drop=number(row, "mi_fgsm_victim_verification_drop") - number(row, "pgd_victim_verification_drop"))
        rows.append(row)
    return rows


def print_final(pgd_summary: dict[str, Any], summary: dict[str, Any], comparison: list[dict[str, Any]], config: dict[str, Any]) -> None:
    print("=" * 52); print("PGD vs MI-FGSM — Transfer Verification Development Check"); print("=" * 52)
    print(f"PGD: eps={EPS} steps={STEPS} alpha={EPS / STEPS}")
    attack = config["attack"]
    print(f"MI-FGSM: eps={attack['eps']} steps={attack['steps']} alpha={attack['alpha']} momentum={attack['momentum']}")
    print(f"ArcFace threshold = {summary['source']['threshold']:.6f}")
    print(f"FaceNet threshold = {summary['victim']['threshold']:.6f}")
    print("-" * 52)
    for row in comparison:
        print(f"{row['identity_id']}: PGD ArcFace={number(row, 'pgd_source_A_B_adv'):.6f}, FaceNet B-Badv={number(row, 'pgd_victim_self_adv_cosine'):.6f}, FaceNet A-Badv={number(row, 'pgd_victim_A_B_adv'):.6f}, crossed={row['pgd_victim_boundary_crossed']}; "
              f"MI ArcFace={number(row, 'mi_fgsm_source_A_B_adv'):.6f}, FaceNet B-Badv={number(row, 'mi_fgsm_victim_self_adv_cosine'):.6f}, FaceNet A-Badv={number(row, 'mi_fgsm_victim_A_B_adv'):.6f}, crossed={row['mi_fgsm_victim_boundary_crossed']}")
    print("-" * 52)
    print(f"SOURCE: PGD success={pgd_summary['counts']['source_boundary_crossed']}/5 mean A-Badv={average(comparison, 'pgd_source_A_B_adv'):.6f}; MI success={summary['counts']['source_boundary_crossed']}/5 mean A-Badv={average(comparison, 'mi_fgsm_source_A_B_adv'):.6f}")
    print(f"VICTIM REPRESENTATION: PGD mean B-Badv={average(comparison, 'pgd_victim_self_adv_cosine'):.6f}; MI mean B-Badv={average(comparison, 'mi_fgsm_victim_self_adv_cosine'):.6f}")
    print(f"VICTIM VERIFICATION: PGD mean A-Badv={average(comparison, 'pgd_victim_A_B_adv'):.6f}, drop={average(comparison, 'pgd_victim_verification_drop'):.6f}, crossing={pgd_summary['counts']['transfer_success']}/{pgd_summary['counts']['eligible']}; MI mean A-Badv={average(comparison, 'mi_fgsm_victim_A_B_adv'):.6f}, drop={average(comparison, 'mi_fgsm_victim_verification_drop'):.6f}, crossing={summary['counts']['transfer_success']}/{summary['counts']['eligible']}")
    print("No generalization beyond these five development identities."); print("=" * 52)


def main() -> int:
    args = parse_args()
    if args.eps != EPS or args.steps != STEPS:
        raise SystemExit("v1 is frozen to eps=0.04 and steps=200; smoke automatically uses at most five steps")
    root = project_root()
    selected = read_csv(root / "data/datasets/lfw/evaluation_splits/development.csv")[:1 if args.smoke else 5]
    if len(selected) != (1 if args.smoke else 5):
        raise SystemExit("frozen development selection unavailable")
    steps = min(STEPS, 5) if args.smoke else STEPS
    output = args.output_dir or root / "results/transfer" / ("mi_fgsm_arcface_to_facenet_verification_smoke" if args.smoke else "mi_fgsm_arcface_to_facenet_verification_v1")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty run directory: {output}")
    source_thresholds, source_artifact = load_thresholds(root / "results/calibration/arcface_lfw_v1")
    victim_thresholds, victim_artifact = load_thresholds(root / "results/calibration/facenet_lfw_v1")
    pgd_dir = root / "results/transfer/pgd_arcface_to_facenet_verification_v1"
    pgd = {row["identity_id"]: row for row in read_csv(pgd_dir / "results.csv")}
    missing = [row["identity_id"] for row in selected if row["identity_id"] not in pgd]
    if missing:
        raise SystemExit(f"frozen PGD results lack selected identities: {missing}")
    output.mkdir(parents=True); (output / "adversarial_probes").mkdir()
    app = create_face_app(det_size=DEFAULT_DET_SIZE)
    victim = load_embedder("facenet_vggface2")  # Evaluation-only: never passed to ATTACKS or autograd.
    rows: list[dict[str, Any]] = []; failures: list[dict[str, str]] = []
    for pair in selected:
        identity = pair["identity_id"]
        row: dict[str, Any] = {field: "" for field in RESULT_FIELDS}
        row.update(identity_id=identity, reference_image=pair["reference_image"], probe_image=pair["probe_image"], valid=False, failure_reason="")
        try:
            reference, probe = load_bgr(root / pair["reference_image"]), load_bgr(root / pair["probe_image"])
            started = time.perf_counter()
            attack = ATTACKS["mi_fgsm"].apply(probe, app=app, config=AttackConfig(name="mi_fgsm", eps=EPS, steps=steps, momentum=MOMENTUM))
            runtime = time.perf_counter() - started
            linf_tensor, linf_serialized = attack.parameters.get("linf_tensor"), attack.parameters.get("linf_serialized")
            if linf_tensor is None or float(linf_tensor) > EPS + TOLERANCE:
                raise RuntimeError(f"tensor L-infinity constraint failed: {linf_tensor} > {EPS} + {TOLERANCE}")
            adversarial = attack.adversarial_bgr
            path = output / "adversarial_probes" / f"{identity}_B_adv_MI.png"; cv2.imwrite(str(path), adversarial)
            sa, sb, sx = (get_embedding_from_bgr(app, image, f"{identity}:{label}") for image, label in ((reference, "A"), (probe, "B"), (adversarial, "B_adv_MI")))
            va, vb, vx = (victim.get_embedding(image, label=f"{identity}:{label}") for image, label in ((reference, "A"), (probe, "B"), (adversarial, "B_adv_MI")))
            source_clean, source_clean_e = score(sa, sb); source_self, source_self_e = score(sb, sx); source_adv, source_adv_e = score(sa, sx)
            victim_clean, victim_clean_e = score(va, vb); victim_self, victim_self_e = score(vb, vx); victim_adv, victim_adv_e = score(va, vx)
            source_crossed = source_clean >= source_thresholds[PRIMARY_OPERATING_POINT] and source_adv < source_thresholds[PRIMARY_OPERATING_POINT]
            victim_valid = victim_clean >= victim_thresholds[PRIMARY_OPERATING_POINT]
            victim_crossed = source_crossed and victim_valid and victim_adv < victim_thresholds[PRIMARY_OPERATING_POINT]
            row.update(adversarial_image=str(path.relative_to(root)), source_clean_cosine=source_clean, source_clean_euclidean=source_clean_e,
                       source_self_adv_cosine=source_self, source_self_adv_euclidean=source_self_e, source_verification_adv_cosine=source_adv,
                       source_verification_adv_euclidean=source_adv_e, source_verification_drop=source_clean-source_adv,
                       source_threshold=source_thresholds[PRIMARY_OPERATING_POINT], source_boundary_crossed=source_crossed,
                       victim_clean_cosine=victim_clean, victim_clean_euclidean=victim_clean_e, victim_self_adv_cosine=victim_self,
                       victim_self_adv_euclidean=victim_self_e, victim_self_similarity_drop=1-victim_self, victim_verification_adv_cosine=victim_adv,
                       victim_verification_adv_euclidean=victim_adv_e, victim_verification_drop=victim_clean-victim_adv,
                       victim_threshold=victim_thresholds[PRIMARY_OPERATING_POINT], victim_clean_valid=victim_valid, victim_boundary_crossed=victim_crossed,
                       transfer_boundary_crossed=victim_crossed, linf_tensor=linf_tensor, linf_serialized=linf_serialized, linf_tensor_valid=True,
                       runtime_seconds=runtime, attack_parameters=json.dumps(attack.parameters, sort_keys=True), valid=True)
        except Exception as exc:
            row["failure_reason"] = f"{type(exc).__name__}: {exc}"; failures.append({"identity_id": identity, "failure_reason": row["failure_reason"]})
        rows.append(row)
    valid = [row for row in rows if row["valid"]]; eligible = [row for row in valid if row["source_boundary_crossed"] and row["victim_clean_valid"]]
    summary = {"title": "ArcFace -> FaceNet MI-FGSM Transfer Verification v1", "scope": "smoke integration only" if args.smoke else "five frozen development identities only; not a general transfer-rate estimate",
               "source": {"model": "ArcFace", "threshold_source": str(source_artifact), "threshold": source_thresholds[PRIMARY_OPERATING_POINT], "operating_point": PRIMARY_OPERATING_POINT},
               "victim": {"model": "facenet_vggface2", "threshold_source": str(victim_artifact), "threshold": victim_thresholds[PRIMARY_OPERATING_POINT], "operating_point": PRIMARY_OPERATING_POINT, "gradient_participation": False},
               "counts": {"attempted": len(rows), "valid": len(valid), "eligible": len(eligible), "source_boundary_crossed": sum(row["source_boundary_crossed"] for row in valid), "transfer_success": sum(row["transfer_boundary_crossed"] for row in eligible), "failures": len(failures)},
               "aggregate": {"mean_source_A_B_adv": average(valid, "source_verification_adv_cosine"), "mean_source_verification_drop": average(valid, "source_verification_drop"), "mean_victim_self_B_Badv_cosine": average(valid, "victim_self_adv_cosine"), "mean_victim_self_cosine_drop": average(valid, "victim_self_similarity_drop"), "mean_victim_adv_A_Badv": average(valid, "victim_verification_adv_cosine"), "mean_victim_verification_drop": average(valid, "victim_verification_drop")}}
    config = {"split": "development", "identities": [row["identity_id"] for row in selected], "smoke": args.smoke, "attack": {"name": "mi_fgsm", "eps": EPS, "steps": steps, "alpha": EPS / steps, "momentum": MOMENTUM, "random_start": False, "gradient_normalization": "per-image L1 norm", "objective": "canonical ArcFace self-reference cosine minimization"}, "victim_gradient_participation": False, "pgd_authority": str(pgd_dir), "tensor_linf_tolerance": TOLERANCE}
    write_csv(output / "results.csv", rows, RESULT_FIELDS); write_csv(output / "failures.csv", failures, ("identity_id", "failure_reason"))
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.smoke:
        comparison = comparison_rows(pgd, rows); write_csv(output / "comparison_pgd_vs_mi_fgsm.csv", comparison, COMPARISON_FIELDS)
        pgd_summary = json.loads((pgd_dir / "summary.json").read_text(encoding="utf-8")); print_final(pgd_summary, summary, comparison, config)
    else:
        print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
