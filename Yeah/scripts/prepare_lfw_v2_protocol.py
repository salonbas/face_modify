#!/usr/bin/env python3
"""Freeze a 100-person development and calibration-isolated LFW v2 protocol.

Uses only frozen metadata and previously computed calibration scores. Never
loads attack images or evaluates an OOS attack.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from advface.calibration import operating_metrics, select_far_threshold

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/datasets/lfw/evaluation_splits"
OUT = SPLITS / "v2"
SEED = 20261029


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = {name: read(SPLITS / f"attack_{name}.csv") for name in ("dev", "test", "reserve")}
    pairs = read(ROOT / "data/datasets/lfw/pairs/dev_train.csv") + read(ROOT / "data/datasets/lfw/pairs/dev_test.csv")
    calibration_ids = {p[k] for p in pairs for k in ("identity_a", "identity_b")}
    calibration_images = {p[k] for p in pairs for k in ("image_a", "image_b")}
    all_rows = old["dev"] + old["test"] + old["reserve"]
    by_id = {r["identity_id"]: r for r in all_rows}
    assert len(all_rows) == len(by_id) == 328
    old_overlap = {name: sorted({r["identity_id"] for r in rows} & calibration_ids) for name, rows in old.items()}
    assert [len(old_overlap[k]) for k in ("dev", "test", "reserve")] == [1, 1, 9]
    # All previously calibrated identities are assigned to development. The
    # legacy pilot remains intact; old test/reserve files are never changed.
    mandatory = set(r["identity_id"] for r in old["dev"]) | set().union(*map(set, old_overlap.values()))
    # Draw the remaining development IDs from the old reserve, leaving the
    # historical 100-person test intact except for its one contaminated member.
    pool = sorted({r["identity_id"] for r in old["reserve"]} - mandatory)
    random.Random(SEED).shuffle(pool)
    dev_ids = mandatory | set(pool[:100 - len(mandatory)])
    oos_ids = set(by_id) - dev_ids
    assert len(dev_ids) == 100 and len(oos_ids) == 228
    assert not (oos_ids & calibration_ids)
    fields = list(all_rows[0])
    for name, ids in (("development", dev_ids), ("oos", oos_ids)):
        rows = [{**by_id[ident], "split": name} for ident in sorted(ids)]
        write(OUT / f"{name}.csv", rows, fields)
    images_dev = {by_id[i][k] for i in dev_ids for k in ("reference_image_id", "probe_image_id")}
    images_oos = {by_id[i][k] for i in oos_ids for k in ("reference_image_id", "probe_image_id")}
    assert not (images_dev & images_oos)
    # v2 definitions remove *every* attack identity, including pairs whose v1
    # model scoring failed. Source pair order and pair IDs remain traceable.
    kept = [p for p in pairs if p["identity_a"] not in by_id and p["identity_b"] not in by_id]
    kept_ids = {p[k] for p in kept for k in ("identity_a", "identity_b")}
    kept_images = {p[k] for p in kept for k in ("image_a", "image_b")}
    assert not kept_ids & set(by_id)
    assert not kept_images & (images_dev | images_oos)
    pair_fields = list(pairs[0])
    for name in ("dev_train", "dev_test"):
        write(OUT / "calibration_pairs" / f"{name}.csv", [p for p in kept if p["source_split"] == name], pair_fields)
    audit = {
        "protocol": "LFW attack/calibration v2", "seed": SEED,
        "old_split_counts": {k: len(v) for k, v in old.items()},
        "old_calibration_identity_overlap": old_overlap,
        "old_calibration_ab_image_overlap": {k: sorted({r[x] for r in v for x in ("reference_image_id", "probe_image_id")} & calibration_images) for k, v in old.items()},
        "development_count": len(dev_ids), "oos_count": len(oos_ids),
        "development_from_old": {k: len(dev_ids & {r["identity_id"] for r in v}) for k, v in old.items()},
        "oos_from_old": {k: len(oos_ids & {r["identity_id"] for r in v}) for k, v in old.items()},
        "pilot_20_subset_development": {r["identity_id"] for r in old["dev"]} <= dev_ids,
        "v2_dev_oos_identity_overlap": len(dev_ids & oos_ids),
        "v2_dev_oos_ab_image_overlap": len(images_dev & images_oos),
        "v2_calibration_attack_identity_overlap": len(kept_ids & set(by_id)),
        "v2_calibration_attack_image_overlap": len(kept_images & (images_dev | images_oos)),
        "v1_calibration_oos_identity_overlap": len(oos_ids & calibration_ids),
        "pairing": "Preserved exact A/B IDs from frozen v1 CSVs",
        "selection": "old dev 20 plus all 11 v1 calibration-overlap identities; fill to 100 from sorted remaining old reserve IDs shuffled with Python Random(seed)",
        "oos_use": "No attack evaluation, method tuning, victim selection, or eligibility screening before protocol freeze",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [*(SPLITS / f"attack_{n}.csv" for n in old), ROOT / "data/datasets/lfw/pairs/dev_train.csv", ROOT / "data/datasets/lfw/pairs/dev_test.csv"]},
    }
    for model in ("arcface", "facenet"):
        source = ROOT / f"results/calibration/{model}_lfw_v1/pair_scores.csv"
        scores = read(source)
        assert len(scores) == len(pairs)
        retained = [r for r in scores if r["identity_a"] not in by_id and r["identity_b"] not in by_id]
        assert len(retained) == len(kept)
        assert [(r["image_a"], r["image_b"]) for r in retained] == [(r["image_a"], r["image_b"]) for r in kept]
        cal_dir = ROOT / f"results/calibration/{model}_lfw_v2"
        write(cal_dir / "pair_scores.csv", retained, list(scores[0]))
        arrays = {}
        counts = {}
        for split in ("dev_train", "dev_test"):
            section = [r for r in retained if r["split"] == split]
            valid = [r for r in section if r["valid"].lower() == "true"]
            arrays[split] = (np.array([float(r["cosine_similarity"]) for r in valid]), np.array([r["same_identity"].lower() == "true" for r in valid]))
            counts[split] = {"defined": len(section), "valid": len(valid), "genuine": int(arrays[split][1].sum()), "impostor": int((~arrays[split][1]).sum())}
        train_scores, train_labels = arrays["dev_train"]
        point = select_far_threshold(train_scores[~train_labels], 1e-2)
        point["development"] = operating_metrics(*arrays["dev_train"], point["threshold"])
        point["held_out_test"] = operating_metrics(*arrays["dev_test"], point["threshold"])
        (cal_dir / "thresholds.json").write_text(json.dumps({"far_1e-2": point}, indent=2, sort_keys=True) + "\n")
        metadata = {"protocol": "LFW calibration v2: official dev train/test pair definitions filtered to exclude all 328 attack identities", "model": model, "target_far": 0.01, "threshold_selected_on": "valid dev_train scores only", "test_usage": "evaluation only", "source_pair_scores": str(source.relative_to(ROOT)), "source_pair_scores_sha256": digest(source), "source_model_config": f"results/calibration/{model}_lfw_v1/config.json", "pair_counts": counts, "attack_split": str(OUT.relative_to(ROOT)), "score_reuse": "Existing v1 cosine scores retained for unchanged pairs; no new embedding inference", "identity_image_overlap_with_attack": 0}
        (cal_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        audit[f"{model}_v2"] = {"threshold": point["threshold"], "counts": counts}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
