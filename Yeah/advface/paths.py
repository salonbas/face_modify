from datetime import datetime
from pathlib import Path

_ATTACK_BASE = {
    "fgsm": "results/fgsm",
    "pgd": "results/pgd",
    "pgd_full": "results/pgd",
    "gaussian": "results/gaussian",
    "compare": "results/compare",
}


def make_run_dir(
    attack_type: str = "fgsm",
    run_name: str | None = None,
) -> Path:
    """
    建立並回傳本次執行的輸出目錄。

    目錄結構：results/<attack_type>/<run_name>/
      attack_type：fgsm / pgd / gaussian / compare
      run_name   ：未指定則自動用時間戳，避免覆蓋舊實驗。
    """
    base = Path(_ATTACK_BASE.get(attack_type, f"results/{attack_type}"))
    name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()
