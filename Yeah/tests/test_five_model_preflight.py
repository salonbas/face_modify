"""Safety properties of the proposed five-model entry point."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_lfw_five_model_benchmark.py"


def load_script():
    spec = importlib.util.spec_from_file_location("five_model_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preflight_fails_closed_without_complete_timing_and_executor():
    report = load_script().preflight()
    assert report["decision"] == "NOT READY"
    assert any("end-to-end estimate" in p for p in report["problems"])
    assert any("execute/resume" in p for p in report["problems"])
    assert report["evidence"]["attack_contract"]["steps"] == 200
    assert report["evidence"]["attack_contract"]["alpha"] == 0.0002


def test_calibration_isolation_uses_identity_fields():
    module = load_script()
    # The invariant applies to all attack identities, including the OOS
    # manifest, without opening any OOS image or using it for model selection.
    dev = module.rows(module.SPLITS / "development.csv")
    oos = module.rows(module.SPLITS / "oos.csv")
    attack_ids = {r["identity_id"] for r in dev + oos}
    for part in ("dev_train", "dev_test"):
        pairs = module.rows(module.SPLITS / "calibration_pairs" / f"{part}.csv")
        assert pairs
        assert not any(r["identity_a"] in attack_ids or r["identity_b"] in attack_ids for r in pairs)


def test_execute_refuses_before_creating_formal_artifacts():
    formal = SCRIPT.parents[1] / "results/transfer/lfw_dev_100_five_model_v1/formal"
    before = set(formal.rglob("*")) if formal.exists() else set()
    result = subprocess.run([sys.executable, str(SCRIPT), "--execute"],
                            capture_output=True, text=True, check=False)
    after = set(formal.rglob("*")) if formal.exists() else set()
    assert result.returncode != 0
    assert "Formal execute refused" in result.stdout
    assert after == before
