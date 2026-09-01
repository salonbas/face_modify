"""Architecture closure tests：不載入人臉模型、不寫入正式 results/。"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np

from advface.attacks.registry import (
    AttackOutput,
    get_attack_spec,
    normalize_attack_name,
)
from advface.evaluation.transfer import evaluate_on_model
from advface.experiments.runner import run_experiment


class _DummyModel:
    name = "dummy"
    threshold = 0.4
    app = object()

    def get_embedding(self, image_bgr, label: str = "image") -> np.ndarray:
        if label == "original":
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def test_registry_exposes_fgsm_and_pgd():
    fgsm = get_attack_spec("fgsm")
    pgd = get_attack_spec("pgd_full")
    assert fgsm.name == "fgsm" and not fgsm.requires_steps
    assert pgd.name == "pgd_full" and pgd.requires_steps
    assert normalize_attack_name("pgd-full") == "pgd_full"


def test_fgsm_and_pgd_through_canonical_runner(monkeypatch):
    from advface.experiments import runner as runner_mod

    seen: list[str] = []

    def fake_apply(img, *, attack, app, config, device=None):
        seen.append(normalize_attack_name(attack))
        return AttackOutput(
            adversarial_bgr=np.array(img, copy=True),
            name=normalize_attack_name(attack),
            parameters={"eps": float(config.eps), "steps": config.steps},
        )

    monkeypatch.setattr(runner_mod, "apply_attack", fake_apply)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    sur = _DummyModel()
    vic = _DummyModel()
    vic.name = "dummy_victim"

    r_fgsm = run_experiment(img, attack="fgsm", surrogate=sur, victims=vic, eps=0.01, steps=1)
    r_pgd = run_experiment(img, attack="pgd_full", surrogate=sur, victims={"victim": vic}, eps=0.02, steps=2)

    assert seen == ["fgsm", "pgd_full"]
    assert r_fgsm.attack_name == "fgsm"
    assert r_pgd.attack_name == "pgd_full"
    assert r_fgsm.surrogate.success is True
    assert r_pgd.primary_victim is not None
    assert "victim" in r_pgd.victims
    assert "victims" in r_fgsm.to_metrics_dict()


def test_victim_uses_embedding_model_interface():
    model = _DummyModel()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    r = evaluate_on_model(img, img, model)
    assert hasattr(model, "get_embedding")
    assert r.model_name == "dummy"
    assert r.cosine_after == 0.0


def test_single_and_batch_share_canonical_runner():
    from advface.benchmark.unit import run_experiment_unit
    from advface.experiments.transfer_unit import attack_and_evaluate

    assert "run_experiment(" in inspect.getsource(attack_and_evaluate)
    assert "run_experiment(" in inspect.getsource(run_experiment_unit)


def test_reporting_modules_do_not_import_attack_or_models():
    root = Path(__file__).resolve().parents[1]
    for rel in ("advface/benchmark/report.py",):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        mods: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
            elif isinstance(node, ast.Import):
                mods.extend(alias.name for alias in node.names)
        assert not any(m.startswith("advface.attacks") for m in mods)
        assert not any(m.startswith("advface.models") for m in mods)
        assert not any(m.startswith("advface.experiments.runner") for m in mods)


def test_write_report_html_does_not_call_models(tmp_path, monkeypatch):
    from advface.experiments.transfer_output import write_report_html, write_transfer_metrics_json

    def boom(*_a, **_k):
        raise AssertionError("reporting must not load models")

    monkeypatch.setattr("advface.models.insightface_app.create_face_app", boom)
    metrics = {
        "experiment": {"name": "t", "run_id": "r", "date": "d", "image": "x", "image_stem": "x"},
        "attack": {"type": "fgsm", "eps": 0.01, "steps": 1},
        "surrogate": {
            "model": "s",
            "cosine_before": 1.0,
            "cosine_after": 0.1,
            "delta": 0.9,
            "success": True,
            "threshold": 0.4,
            "euclidean_distance": 1.0,
            "error": None,
        },
        "victim": {
            "model": "v",
            "cosine_before": 1.0,
            "cosine_after": 0.9,
            "delta": 0.1,
            "success": False,
            "threshold": None,
            "euclidean_distance": 0.1,
            "transfer_observed": False,
            "error": None,
        },
        "images": {"original": "o.png", "adversarial": "a.png", "perturbation": "p.png"},
        "config": {},
        "conclusion": {
            "whitebox": "Successful",
            "transfer": "Not Observed",
            "body": "x",
            "closing": "y",
            "transfer_label": "Not Observed",
        },
    }
    write_transfer_metrics_json(tmp_path, metrics)
    html = write_report_html(tmp_path, metrics).read_text(encoding="utf-8")
    assert "fgsm" in html
    assert "0.1000" in html or "0.1" in html


def test_test_artifacts_do_not_pollute_repo_results(tmp_path, monkeypatch):
    monkeypatch.setenv("ADVFACE_ROOT", str(tmp_path))
    from advface.config import project_root
    from advface.paths import make_run_dir

    assert project_root() == tmp_path.resolve()
    out = make_run_dir(attack_type="benchmark", run_name="arch_isolation")
    assert out == (tmp_path / "results" / "benchmarks" / "arch_isolation").resolve()
    repo_results = Path(__file__).resolve().parents[1] / "results" / "benchmarks" / "arch_isolation"
    assert not repo_results.exists()
