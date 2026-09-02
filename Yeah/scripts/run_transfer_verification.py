#!/usr/bin/env python3
"""ArcFace PGD Full -> FaceNet calibrated reference/probe transfer check."""
from __future__ import annotations

import argparse
import csv
import json
import sys
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

FIELDS = ("identity_id", "reference_image", "probe_image", "adversarial_image", "source_clean_A_B",
          "source_self_B_Badv", "source_adv_A_Badv", "source_threshold", "source_boundary_crossed",
          "victim_clean_A_B", "victim_clean_euclidean", "victim_self_B_Badv", "victim_self_euclidean",
          "victim_adv_A_Badv", "victim_adv_euclidean", "victim_threshold", "victim_clean_valid",
          "victim_boundary_crossed", "transfer_verification_success", "valid", "failure_reason")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def score(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    return float(cosine_similarity(a, b)), float(euclidean_distance(a, b))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["development"], default="development")
    p.add_argument("--max-identities", type=int, default=5)
    p.add_argument("--eps", type=float, default=0.04); p.add_argument("--steps", type=int, default=200)
    p.add_argument("--source-run", type=Path, default=None); p.add_argument("--victim-calibration", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_identities != 5 or args.eps != 0.04 or args.steps != 200:
        raise SystemExit("v1 is frozen to exactly five development identities and PGD Full eps=0.04, steps=200")
    root = project_root(); out = args.output_dir or root / "results/transfer/pgd_arcface_to_facenet_verification_v1"
    if out.exists() and any(out.iterdir()): raise SystemExit(f"refusing to overwrite non-empty run directory: {out}")
    victim_source = args.victim_calibration or root / "results/calibration/facenet_lfw_v1"
    victim_thresholds, victim_artifact = load_thresholds(victim_source)
    source_dir = args.source_run or root / "results/robustness/lfw_verification_pgd_v1"
    source_rows = {r["identity_id"]: r for r in read_csv(source_dir / "results.csv")}
    selected = read_csv(root / "data/datasets/lfw/evaluation_splits/development.csv")[:5]
    if len(selected) != 5: raise SystemExit("frozen development selection unavailable")
    missing = [p["identity_id"] for p in selected if p["identity_id"] not in source_rows]
    if missing: raise SystemExit(f"source authority lacks selected identities: {missing}")
    out.mkdir(parents=True); (out / "adversarial_probes").mkdir()
    source_thresholds, source_artifact = load_thresholds(root / "results/calibration/arcface_lfw_v1")
    app = create_face_app(det_size=DEFAULT_DET_SIZE)  # ArcFace only: source scoring and PGD gradients.
    victim = load_embedder("facenet_vggface2")       # Never passed to ATTACKS.
    rows: list[dict[str, Any]] = []; failures: list[dict[str, str]] = []
    for pair in selected:
        row = {key: "" for key in FIELDS}; ident = pair["identity_id"]
        row.update(identity_id=ident, reference_image=pair["reference_image"], probe_image=pair["probe_image"], valid=False, failure_reason="")
        try:
            a, b = load_bgr(root / pair["reference_image"]), load_bgr(root / pair["probe_image"])
            # No legacy B_adv artifact exists; regenerate exactly once under the frozen source config.
            badv = ATTACKS["pgd_full"].apply(b, app=app, config=AttackConfig(name="pgd_full", eps=0.04, steps=200)).adversarial_bgr
            path = out / "adversarial_probes" / f"{ident}_B_adv.png"; cv2.imwrite(str(path), badv); row["adversarial_image"] = str(path.relative_to(root))
            sa, sb, sx = (get_embedding_from_bgr(app, image, f"{ident}:{label}") for image, label in ((a, "A"), (b, "B"), (badv, "B_adv")))
            va, vb, vx = (victim.get_embedding(image, label=f"{ident}:{label}") for image, label in ((a, "A"), (b, "B"), (badv, "B_adv")))
            sclean, _ = score(sa, sb); sself, _ = score(sb, sx); sadv, _ = score(sa, sx)
            vclean, vclean_e = score(va, vb); vself, vself_e = score(vb, vx); vadv, vadv_e = score(va, vx)
            source_crossed = sclean >= source_thresholds[PRIMARY_OPERATING_POINT] and sadv < source_thresholds[PRIMARY_OPERATING_POINT]
            clean_valid = vclean >= victim_thresholds[PRIMARY_OPERATING_POINT]
            victim_crossed = clean_valid and vadv < victim_thresholds[PRIMARY_OPERATING_POINT]
            row.update(source_clean_A_B=sclean, source_self_B_Badv=sself, source_adv_A_Badv=sadv,
                       source_threshold=source_thresholds[PRIMARY_OPERATING_POINT], source_boundary_crossed=source_crossed,
                       victim_clean_A_B=vclean, victim_clean_euclidean=vclean_e, victim_self_B_Badv=vself, victim_self_euclidean=vself_e,
                       victim_adv_A_Badv=vadv, victim_adv_euclidean=vadv_e, victim_threshold=victim_thresholds[PRIMARY_OPERATING_POINT],
                       victim_clean_valid=clean_valid, victim_boundary_crossed=victim_crossed,
                       transfer_verification_success=source_crossed and victim_crossed, valid=True)
        except Exception as exc:
            row["failure_reason"] = f"{type(exc).__name__}: {exc}"; failures.append({"identity_id": ident, "failure_reason": row["failure_reason"]})
        rows.append(row)
    valid = [r for r in rows if r["valid"]]; clean = [r for r in valid if r["victim_clean_valid"] is True]
    mean = lambda key, values: float(np.mean([float(r[key]) for r in values])) if values else None
    eligible = [r for r in clean if r["source_boundary_crossed"] is True]; success = sum(r["transfer_verification_success"] is True for r in eligible)
    summary = {"title": "ArcFace -> FaceNet PGD Transfer Verification v1", "scope": "five frozen development identities only; not a general transfer-rate estimate",
      "source": {"model": "ArcFace", "threshold_source": str(source_artifact), "threshold": source_thresholds[PRIMARY_OPERATING_POINT], "operating_point": PRIMARY_OPERATING_POINT},
      "victim": {"model": "facenet_vggface2", "threshold_source": str(victim_artifact), "threshold": victim_thresholds[PRIMARY_OPERATING_POINT], "operating_point": PRIMARY_OPERATING_POINT, "gradient_participation": False},
      "counts": {"attempted": len(rows), "valid": len(valid), "victim_clean_valid": len(clean), "victim_clean_invalid": len(valid)-len(clean), "eligible": len(eligible), "source_boundary_crossed": sum(r["source_boundary_crossed"] is True for r in clean), "transfer_success": success, "failures": len(failures)},
      "aggregate": {"mean_victim_self_B_Badv_cosine": mean("victim_self_B_Badv", clean), "mean_victim_self_cosine_drop": (1 - mean("victim_self_B_Badv", clean)) if clean else None, "mean_victim_clean_A_B": mean("victim_clean_A_B", clean), "mean_victim_adv_A_Badv": mean("victim_adv_A_Badv", clean), "mean_victim_verification_drop": (mean("victim_clean_A_B", clean)-mean("victim_adv_A_Badv", clean)) if clean else None, "true_boundary_crossing_rate": success / len(eligible) if eligible else None}}
    config = {"split": "development", "identities": [p["identity_id"] for p in selected], "attack": {"name": "pgd_full", "eps": 0.04, "steps": 200, "objective": "unchanged ArcFace self-similarity minimization"}, "source_run": str(source_dir), "adversarial_probes": "regenerated because source run does not persist B_adv artifacts", "victim_gradient_participation": False}
    write_csv(out / "results.csv", rows, FIELDS); write_csv(out / "failures.csv", failures, ("identity_id", "failure_reason"))
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True)+"\n", encoding="utf-8"); (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0 if not failures else 1


if __name__ == "__main__": raise SystemExit(main())
