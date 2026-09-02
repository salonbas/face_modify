"""tests for project_root and make_run_dir."""
from __future__ import annotations

import os
from pathlib import Path

from advface.config import project_root
from advface.paths import get_dataset_path, make_run_dir


def test_project_root_from_package_location():
    root = project_root()
    assert (root / "advface" / "config.py").is_file()
    assert (root / "scripts" / "run_attack.py").is_file()


def test_project_root_ignores_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ADVFACE_ROOT", raising=False)
    root = project_root()
    assert root == Path(__file__).resolve().parents[1]
    assert (root / "advface").is_dir()


def test_project_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ADVFACE_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()


def test_make_run_dir_under_project_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ADVFACE_ROOT", str(tmp_path))
    out = make_run_dir(attack_type="fgsm", run_name="unit_test_run")
    assert out == (tmp_path / "results" / "fgsm" / "unit_test_run").resolve()
    assert out.is_dir()


def test_get_dataset_path_rejects_nested_names(tmp_path, monkeypatch):
    monkeypatch.setenv("ADVFACE_ROOT", str(tmp_path))
    (tmp_path / "data" / "datasets" / "lfw").mkdir(parents=True)
    assert get_dataset_path("lfw") == tmp_path / "data" / "datasets" / "lfw"
