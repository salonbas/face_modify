"""Integration smoke：2 images × 1 eps × PGD 2 steps。不得寫入正式 run 目錄。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from advface.config import project_root

pytestmark = pytest.mark.integration


def test_benchmark_pipeline_smoke(tmp_path, monkeypatch):
    if os.environ.get("ADVFACE_SKIP_SMOKE", "").strip() in ("1", "true", "yes"):
        pytest.skip("ADVFACE_SKIP_SMOKE set")

    root = project_root()
    img_a = root / "data" / "raw" / "musk1.jpg"
    img_b = root / "data" / "raw" / "face2.jpg"
    if not img_a.is_file() or not img_b.is_file():
        pytest.skip("local smoke images missing")

    from advface.benchmark.dataset import ManifestRow, write_manifest
    from advface.benchmark.runner import run_benchmark

    man = tmp_path / "manifest.csv"
    write_manifest(
        man,
        [
            ManifestRow("smoke_a", str(img_a), "musk", "local_raw"),
            ManifestRow("smoke_b", str(img_b), "face2", "local_raw"),
        ],
    )
    monkeypatch.setenv("ADVFACE_ROOT", str(tmp_path))
    # 輸出寫到 tmp_path/results/benchmarks/...，不覆蓋正式 benchmark
    out = run_benchmark(
        manifest_path=man,
        max_images=2,
        attacks=["pgd_full"],
        eps_list=[0.01],
        pgd_steps=[2],
        victim="facenet",
        device="cpu",
        run_name="smoke_transfer_benchmark_v0",
        skip_examples=False,
    )
    assert out == (tmp_path / "results" / "benchmarks" / "smoke_transfer_benchmark_v0").resolve()
    assert (out / "raw_results.jsonl").is_file()
    assert (out / "summary.json").is_file()
    assert (out / "report.html").is_file()
    assert (out / "config.json").is_file()
    assert (out / "manifest_snapshot.csv").is_file()
    lines = [ln for ln in (out / "raw_results.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2

    # resume：不得重跑
    out2 = run_benchmark(
        manifest_path=man,
        max_images=2,
        attacks=["pgd_full"],
        eps_list=[0.01],
        pgd_steps=[2],
        victim="facenet",
        device="cpu",
        run_name="smoke_transfer_benchmark_v0",
        skip_examples=True,
    )
    assert out2 == out
    lines2 = [ln for ln in (out / "raw_results.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines2) == 2
