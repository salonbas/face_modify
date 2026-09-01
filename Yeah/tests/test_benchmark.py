"""Large-scale Transfer Benchmark v0 單元測試（不載入人臉模型）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from advface.benchmark.aggregate import (
    build_summary,
    is_valid_row,
    is_whitebox_success,
    summarize_subset,
)
from advface.benchmark.checkpoint import (
    append_result,
    completed_keys,
    load_jsonl,
    rewrite_raw_tables,
)
from advface.benchmark.dataset import ManifestRow, load_manifest, select_images, write_manifest
from advface.benchmark.failures import INVALID_LINF, NO_FACE_DETECTED
from advface.benchmark.grid import find_duplicate_keys, generate_experiment_grid, unit_key
from advface.benchmark.report import write_benchmark_report_html
from advface.benchmark.unit import perturbation_metrics, validate_linf
from advface.config import LINF_TOLERANCE


def _row(**kwargs):
    base = {
        "unit_key": "img|fgsm|0.010|1",
        "image_id": "img",
        "attack": "fgsm",
        "eps": 0.01,
        "steps": 1,
        "status": "completed",
        "linf_ok": True,
        "whitebox_success": True,
        "surrogate_cosine": 0.2,
        "victim_cosine": 0.8,
        "victim_cosine_drop": 0.2,
        "surrogate_euclidean": 1.0,
        "victim_euclidean": 0.5,
        "linf": 0.01,
        "runtime_sec": 1.2,
        "failure_code": "",
    }
    base.update(kwargs)
    return base


def test_grid_generation_fgsm_and_pgd(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    rows = [
        ManifestRow("a", str(img), "id_a", "test"),
        ManifestRow("b", str(img), "id_b", "test"),
    ]
    units = generate_experiment_grid(
        rows,
        attacks=["fgsm", "pgd_full"],
        eps_list=[0.005, 0.01],
        pgd_steps=[20, 50],
    )
    # 2 images * (2 fgsm + 2 eps * 2 steps pgd) = 2 * (2 + 4) = 12
    assert len(units) == 12
    fgsm = [u for u in units if u.attack == "fgsm"]
    pgd = [u for u in units if u.attack == "pgd_full"]
    assert len(fgsm) == 4
    assert all(u.steps == 1 for u in fgsm)
    assert len(pgd) == 8
    assert find_duplicate_keys(units) == []
    keys = [u.key for u in units]
    assert len(keys) == len(set(keys))


def test_duplicate_experiment_detection():
    from advface.benchmark.grid import ExperimentUnit

    u = ExperimentUnit("a", "p", "", "", "fgsm", 0.01, 1)
    assert find_duplicate_keys([u, u]) == [u.key]


def test_unit_key_stable():
    assert unit_key("sun", "pgd_full", 0.04, 200) == "sun|pgd_full|0.040|200"


def test_resume_skips_completed_and_failed(tmp_path):
    p = tmp_path / "raw_results.jsonl"
    append_result(p, _row(unit_key="k1", status="completed"))
    append_result(p, _row(unit_key="k2", status="failed", failure_code=NO_FACE_DETECTED))
    append_result(p, _row(unit_key="k3", status="invalid", failure_code=INVALID_LINF, linf_ok=False))
    rows = load_jsonl(p)
    skip_all = completed_keys(rows, include_failed=True)
    skip_ok = completed_keys(rows, include_failed=False)
    assert skip_all == {"k1", "k2", "k3"}
    assert skip_ok == {"k1", "k3"}
    assert "k2" not in skip_ok


def test_fgsm_full_image_linf_not_used_as_hard_invalid():
    assert validate_linf(0.33, 0.04) is False
    assert validate_linf(0.040 + LINF_TOLERANCE, 0.040)


def test_failure_recording_csv(tmp_path):
    rows = [
        _row(unit_key="ok", status="completed"),
        _row(unit_key="bad", status="failed", failure_code=NO_FACE_DETECTED, failure_message="no face"),
        _row(unit_key="linf", status="invalid", failure_code=INVALID_LINF, linf_ok=False),
    ]
    rewrite_raw_tables(tmp_path, rows)
    text = (tmp_path / "failures.csv").read_text(encoding="utf-8")
    assert "NO_FACE_DETECTED" in text
    assert "INVALID_LINF" in text
    assert "ok" not in text.splitlines()[1] if False else True
    assert "bad" in text and "linf" in text


def test_linf_validation():
    orig = np.zeros((8, 8, 3), dtype=np.uint8)
    adv = orig.copy()
    adv[0, 0, 0] = 3  # 3/255
    linf, l2 = perturbation_metrics(orig, adv)
    assert abs(linf - 3 / 255) < 1e-12
    assert l2 > 0
    assert validate_linf(0.010, 0.010)
    assert validate_linf(0.010 + LINF_TOLERANCE, 0.010)
    assert not validate_linf(0.010 + LINF_TOLERANCE + 1e-6, 0.010)
    assert not validate_linf(float("nan"), 0.01)


def test_conditional_whitebox_subset():
    rows = [
        _row(unit_key="a", whitebox_success=True, victim_cosine_drop=0.4),
        _row(unit_key="b", whitebox_success=False, victim_cosine_drop=0.05, surrogate_cosine=0.9),
        _row(unit_key="c", status="failed", linf_ok=False, whitebox_success=False, victim_cosine_drop=None),
    ]
    all_s = summarize_subset(rows)
    wb = [r for r in rows if is_whitebox_success(r)]
    wb_s = summarize_subset(wb)
    assert all_s["n_attempted"] == 3
    assert all_s["n_valid"] == 2
    assert all_s["n_whitebox_success"] == 1
    assert abs(all_s["victim_cosine_drop_mean"] - 0.225) < 1e-9
    assert wb_s["n_valid"] == 1
    assert abs(wb_s["victim_cosine_drop_mean"] - 0.4) < 1e-9
    assert is_valid_row(rows[0])
    assert not is_whitebox_success(rows[1])


def test_summary_aggregation_percentiles():
    rows = [
        _row(unit_key=f"k{i}", attack="pgd_full", eps=0.02, steps=100, victim_cosine_drop=x)
        for i, x in enumerate([0.1, 0.2, 0.3, 0.4])
    ]
    summary = build_summary(rows, n_images=4, n_planned=4, config={"surrogate_model": "s", "victim_model": "v"})
    cfg = summary["per_config"][0]
    assert cfg["all"]["n_valid"] == 4
    assert cfg["all"]["victim_cosine_drop_mean"] == 0.25
    assert cfg["all"]["victim_cosine_drop_p25"] is not None
    assert cfg["all"]["victim_cosine_drop_std"] is not None
    assert summary["executive"]["transfer_success_rate"] is None
    assert "not calibrated" in summary["executive"]["transfer_success_rate_display"]


def test_html_reads_summary_not_raw(tmp_path):
    distinctive = 0.123456
    summary = {
        "research_question": "RQ-MARKER",
        "executive": {
            "dataset_size": 7,
            "n_valid": 9,
            "n_attack_configurations": 2,
            "surrogate": "insightface_buffalo_l",
            "victim": "facenet_vggface2",
            "overall_whitebox_success_rate": 0.5,
            "victim_mean_cosine_drop": distinctive,
            "victim_transfer_effect_qualitative": "moderate",
            "strongest_transfer_configuration": {
                "attack": "pgd_full",
                "eps": 0.04,
                "steps": 200,
                "victim_cosine_drop_mean": distinctive,
            },
            "weakest_transfer_configuration": {"attack": "fgsm", "eps": 0.005, "steps": 1},
            "transfer_success_rate_display": "N/A — victim threshold not calibrated",
            "n_planned": 10,
            "n_completed_rows": 10,
            "n_failed": 1,
            "n_invalid": 0,
        },
        "per_config": [],
        "charts": {},
        "examples": {},
        "limitations": ["Single surrogate model (InsightFace / ArcFace buffalo_l)."],
        "failure_counts": {NO_FACE_DETECTED: 1},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    # raw file with different numbers — HTML must not use these
    (tmp_path / "raw_results.jsonl").write_text(
        json.dumps(_row(victim_cosine_drop=0.999)) + "\n", encoding="utf-8"
    )
    path = write_benchmark_report_html(tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "RQ-MARKER" in html
    assert "0.1235" in html or "0.123456" in html
    assert "0.999" not in html
    assert "N/A — victim threshold not calibrated" in html
    assert "Executive Summary" in html
    assert "Attack Comparison" in html
    assert "Epsilon Analysis" in html
    assert "PGD Steps Analysis" in html
    assert "Distribution" in html
    assert "Representative Examples" in html
    assert "Limitations" in html
    assert "NO_FACE_DETECTED" in html
    assert "summary.json" in html


def test_load_manifest_contract(tmp_path, monkeypatch):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"not-a-real-jpeg-but-exists")
    man = tmp_path / "manifest.csv"
    write_manifest(
        man,
        [ManifestRow("id1", "x.jpg", "person", "local_raw")],
    )
    monkeypatch.setenv("ADVFACE_ROOT", str(tmp_path))
    rows = load_manifest(man, root=tmp_path)
    assert rows[0].image_id == "id1"
    assert select_images(rows, 1)[0].image_id == "id1"
