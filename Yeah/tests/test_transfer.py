"""Transfer evaluation helpers（不載入重型模型）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from advface.evaluation.transfer import (
    TransferEvalResult,
    build_conclusion,
    evaluate_on_model,
)


def test_calibrated_transfer_runner_is_development_only_and_keeps_models_separate():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/run_transfer_verification.py").read_text(encoding="utf-8")
    assert 'choices=["development"]' in script
    assert 'load_thresholds(victim_source)' in script
    assert 'load_thresholds(root / "results/calibration/arcface_lfw_v1")' in script
    assert 'victim_gradient_participation": False' in script
    assert 'ATTACKS["pgd_full"].apply(b, app=app' in script
    assert "victim.get_embedding" in script
    assert "transfer_verification_success=source_crossed and victim_crossed" in script
    assert "victim_self_B_Badv" in script and "victim_adv_A_Badv" in script


def test_facenet_calibration_reuses_canonical_embedder_and_protocol():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/calibrate_threshold.py").read_text(encoding="utf-8")
    assert '"facenet_vggface2"' in script
    assert "load_embedder(model_name)" in script
    assert 'data / "pairs/dev_train.csv"' in script
    assert 'data / "pairs/dev_test.csv"' in script
    assert 'FaceNetEmbedder: MTCNN detect/crop' in script


class _DummyModel:
    name = "dummy"
    threshold = 0.4

    def __init__(self, emb_o: np.ndarray, emb_a: np.ndarray) -> None:
        self._o = emb_o
        self._a = emb_a

    def get_embedding(self, image_bgr, label: str = "image") -> np.ndarray:
        return self._o if label == "original" else self._a


class _DummyNoThreshold:
    name = "dummy_no_thr"
    threshold = None

    def __init__(self, emb_o: np.ndarray, emb_a: np.ndarray) -> None:
        self._o = emb_o
        self._a = emb_a

    def get_embedding(self, image_bgr, label: str = "image") -> np.ndarray:
        return self._o if label == "original" else self._a


def test_evaluate_on_model_success_below_threshold():
    emb_o = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_a = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # cosine = 0
    model = _DummyModel(emb_o, emb_a)
    r = evaluate_on_model(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8), model)
    assert r.cosine_before == 1.0
    assert abs(r.cosine_after - 0.0) < 1e-6
    assert r.success is True
    assert r.delta == 1.0
    assert r.transfer_observed is True


def test_build_conclusion_transfer_not_observed():
    sur = TransferEvalResult("s", 1.0, 0.2, 0.8, True, threshold=0.4, transfer_observed=True)
    vic = TransferEvalResult("v", 1.0, 0.9, 0.1, False, threshold=None, transfer_observed=False)
    c = build_conclusion(sur, vic)
    assert c["whitebox"] == "Successful"
    assert c["transfer"] == "Not Observed"
    assert c["transfer_label"] == "Not Observed"
    assert "not yet transferable" in c["body"]
    assert "first transfer evaluation baseline" in c["closing"]


def test_build_conclusion_transfer_observed_via_delta():
    sur = TransferEvalResult("s", 1.0, 0.2, 0.8, True, threshold=0.4, transfer_observed=True)
    # 無正式 threshold：以 delta >= 0.2 觀測
    vic = TransferEvalResult("v", 1.0, 0.5, 0.5, False, threshold=None, transfer_observed=True)
    c = build_conclusion(sur, vic)
    assert c["transfer"] == "Observed"


def test_no_threshold_does_not_hardcode_arcface():
    emb_o = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_a = np.array([0.95, 0.3122, 0.0], dtype=np.float32)  # high similarity
    model = _DummyNoThreshold(emb_o, emb_a)
    r = evaluate_on_model(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8), model)
    assert r.threshold is None
    assert r.success is False
    assert r.transfer_observed is False


def test_make_run_dir_transfer(tmp_path, monkeypatch):
    monkeypatch.setenv("ADVFACE_ROOT", str(tmp_path))
    from advface.paths import make_run_dir

    out = make_run_dir(attack_type="transfer", run_name="unit_transfer")
    assert out == (tmp_path / "results" / "transfer" / "unit_transfer").resolve()


def test_report_html_from_metrics(tmp_path):
    from advface.experiments.transfer_output import write_report_html, write_transfer_metrics_json

    metrics = {
        "experiment": {
            "name": "Transfer Evaluation Experiment v1",
            "run_id": "unit_html",
            "date": "2026-08-05 00:00 UTC",
            "image": "data/raw/sun.png",
            "image_stem": "sun",
        },
        "attack": {"type": "pgd_full", "eps": 0.04, "steps": 200},
        "surrogate": {
            "model": "insightface_buffalo_l",
            "cosine_before": 1.0,
            "cosine_after": 0.1,
            "delta": 0.9,
            "success": True,
            "threshold": 0.4,
            "euclidean_distance": 1.2,
            "error": None,
        },
        "victim": {
            "model": "facenet_vggface2",
            "cosine_before": 1.0,
            "cosine_after": 0.99,
            "delta": 0.01,
            "success": False,
            "threshold": None,
            "euclidean_distance": 0.05,
            "transfer_observed": False,
            "error": None,
        },
        "images": {
            "original": "original.png",
            "adversarial": "adversarial.png",
            "perturbation": "perturbation.png",
        },
        "config": {"pipeline": "transfer_evaluation_experiment_v1"},
        "conclusion": {
            "whitebox": "Successful",
            "transfer": "Not Observed",
            "body": "The generated adversarial example successfully fools the surrogate model. However, the perturbation is not yet transferable to the victim model.",
            "closing": "This experiment establishes the first transfer evaluation baseline.",
            "transfer_label": "Not Observed",
        },
    }
    write_transfer_metrics_json(tmp_path, metrics)
    path = write_report_html(tmp_path, metrics)
    html = path.read_text(encoding="utf-8")
    assert "Experiment Information" in html
    assert "White-box Result" in html
    assert "Victim Result" in html
    assert "Threshold" in html and "N/A" in html
    assert "Not Observed" in html
    assert "metrics.json" in html
    # HTML 不應自行計算 cosine（數值來自 metrics）
    assert "0.1000" in html or "0.1" in html
