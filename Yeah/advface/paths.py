from datetime import datetime
from pathlib import Path

from advface.config import project_root


def get_dataset_path(name: str) -> Path:
    """Return the canonical local path for a repository dataset.

    Dataset binaries remain outside the repository.  The repository entry is
    a lightweight metadata directory and, when available, a symlink to the
    external cache.
    """
    if not name or Path(name).name != name:
        raise ValueError("dataset name must be a single directory name")
    path = project_root() / "data" / "datasets" / name
    if not path.is_dir():
        raise FileNotFoundError(f"dataset entry not found: {path}")
    return path.resolve()


def get_lfw_path() -> Path:
    """Return the canonical LFW entry, resolving its external image cache."""
    return get_dataset_path("lfw")

_ATTACK_BASE = {
    "fgsm": "results/fgsm",
    "pgd": "results/pgd",
    "pgd_full": "results/pgd",
    "gaussian": "results/gaussian",
    "compare": "results/compare",
    "transfer": "results/transfer",
    "benchmark": "results/benchmarks",
}


def make_run_dir(
    attack_type: str = "fgsm",
    run_name: str | None = None,
) -> Path:
    """
    建立並回傳本次執行的輸出目錄。

    目錄結構：<project_root>/results/<attack_type>/<run_name>/
      attack_type：fgsm / pgd / gaussian / compare
      run_name   ：未指定則自動用時間戳，避免覆蓋舊實驗。
    """
    rel = _ATTACK_BASE.get(attack_type, f"results/{attack_type}")
    name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = project_root() / rel / name
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()
