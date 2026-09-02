import csv
import json
from pathlib import Path

import numpy as np

from advface.evaluation.verification import evaluate_verification_pair, load_thresholds
from scripts.prepare_lfw_attack_splits import main as prepare_splits
from scripts.run_verification_robustness import parse_args


ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_evaluation_splits_are_disjoint_and_calibration_disjoint():
    split_dir = ROOT / "data/datasets/lfw/evaluation_splits"
    development, test, reserve = (rows(split_dir / f"{name}.csv") for name in ("development", "test", "reserve"))
    sets = [{row["identity_id"] for row in group} for group in (development, test, reserve)]
    assert [len(group) for group in (development, test, reserve)] == [20, 100, 208]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    calibration = rows(ROOT / "results/calibration/arcface_lfw_v1/pair_scores.csv")
    calibrated = {row[key] for row in calibration if row["valid"].lower() == "true" for key in ("identity_a", "identity_b")}
    assert not calibrated & set.union(*sets)


def test_split_generation_is_deterministic_and_pairs_are_same_identity(tmp_path, monkeypatch):
    output = tmp_path / "splits"
    monkeypatch.setattr("sys.argv", ["prepare_lfw_attack_splits.py", "--output-dir", str(output)])
    assert prepare_splits() == 0
    first = (output / "development.csv").read_bytes()
    monkeypatch.setattr("sys.argv", ["prepare_lfw_attack_splits.py", "--output-dir", str(output)])
    assert prepare_splits() == 0
    assert (output / "development.csv").read_bytes() == first
    for row in rows(output / "development.csv"):
        assert row["reference_image_id"] != row["probe_image_id"]
        assert Path(row["reference_image"]).parent.name == row["identity_id"]
        assert Path(row["probe_image"]).parent.name == row["identity_id"]


def test_threshold_is_loaded_from_artifact_not_hardcoded(tmp_path):
    thresholds = {"far_1e-2": {"threshold": 0.123}, "eer": {"threshold": 0.2}, "far_1e-3": {"threshold": 0.3}}
    (tmp_path / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")
    (tmp_path / "summary.json").write_text(json.dumps({"operating_points": {"legacy_0.4": {"threshold": 0.456}}}), encoding="utf-8")
    loaded, _ = load_thresholds(tmp_path)
    assert loaded == {"far_1e-2": 0.123, "eer": 0.2, "far_1e-3": 0.3, "legacy_0.4": 0.456}


def test_clean_invalid_is_excluded_and_below_threshold_crosses_boundary():
    # Unit vectors make cosine values explicit: A·B=1 and A·B_modified=0.
    a = np.array([1.0, 0.0]); b = np.array([1.0, 0.0]); modified = np.array([0.0, 1.0])
    metrics = evaluate_verification_pair(a, b, modified, {"far_1e-2": 0.5, "eer": 0.5, "far_1e-3": 0.5})
    assert metrics["clean_pair_valid"] is True
    assert metrics["crossed_primary_boundary"] is True
    invalid = evaluate_verification_pair(np.array([1.0, 0.0]), np.array([0.0, 1.0]), modified,
                                         {"far_1e-2": 0.5, "eer": 0.5, "far_1e-3": 0.5})
    assert invalid["clean_pair_valid"] is False
    assert invalid["crossed_primary_boundary"] is False


def test_runner_cannot_execute_held_out_test(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_verification_robustness.py", "--split", "test"])
    try:
        parse_args()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("held-out split unexpectedly accepted")
