from __future__ import annotations

import json
import csv
from pathlib import Path
import numpy as np
import pytest

from advface.calibration import eer, far_support_status, labels_from_pairs, operating_metrics, select_far_threshold


def test_label_handling_and_direction():
    labels = labels_from_pairs([{"same_identity": "true"}, {"same_identity": "false"}])
    metrics = operating_metrics(np.array([0.9, 0.1]), labels, 0.5)
    assert labels.tolist() == [True, False]
    assert metrics["tar"] == 1.0 and metrics["far"] == 0.0


def test_far_tar_and_eer():
    scores = np.array([0.9, 0.4, 0.8, 0.2])
    labels = np.array([True, True, False, False])
    metrics = operating_metrics(scores, labels, 0.5)
    assert metrics["tar"] == 0.5 and metrics["far"] == 0.5
    assert eer(scores, labels)["eer"] == pytest.approx(0.5)


def test_target_far_selection_is_conservative():
    selected = select_far_threshold(np.array([0.9, 0.8, 0.7, 0.1]), 0.25)
    assert selected["dev_far"] <= 0.25
    assert selected["threshold"] > 0.8


def test_low_far_support_is_flagged():
    assert far_support_status(1_000, 1e-3) == "EXPLORATORY"
    assert far_support_status(1_000, 1e-2) == "CALIBRATED"


def test_summary_serialization_and_official_split_separation(tmp_path):
    summary = {"train": ["a"], "test": ["b"], "operating_points": {"eer": {"threshold": 0.5}}}
    path = tmp_path / "summary.json"; path.write_text(json.dumps(summary), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["train"] != json.loads(path.read_text(encoding="utf-8"))["test"]
    root = Path(__file__).resolve().parents[1] / "data/datasets/lfw/pairs"
    with (root / "dev_train.csv").open(encoding="utf-8", newline="") as handle:
        train = list(csv.DictReader(handle))
    with (root / "dev_test.csv").open(encoding="utf-8", newline="") as handle:
        test = list(csv.DictReader(handle))
    train_pairs = {(row["image_a"], row["image_b"], row["same_identity"]) for row in train}
    test_pairs = {(row["image_a"], row["image_b"], row["same_identity"]) for row in test}
    assert train_pairs.isdisjoint(test_pairs)
    assert {row["source_split"] for row in train} == {"dev_train"}
    assert {row["source_split"] for row in test} == {"dev_test"}
