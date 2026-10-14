"""Checkpoint / resume：每個 unit 完成後立即 append JSONL。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional

RAW_FIELDS = [
    "unit_key",
    "image_id",
    "image_path",
    "identity_id",
    "source",
    "attack",
    "eps",
    "steps",
    "alpha_px",
    "random_start",
    "seed",
    "surrogate_model",
    "victim_model",
    "run_id",
    "status",
    "failure_code",
    "failure_message",
    "runtime_sec",
    "linf",
    "l2_pixel",
    "linf_ok",
    "surrogate_cosine",
    "surrogate_euclidean",
    "surrogate_threshold",
    "threshold_status_surrogate",
    "whitebox_success",
    "victim_cosine",
    "victim_euclidean",
    "victim_cosine_drop",
    "victim_threshold",
    "threshold_status_victim",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def completed_keys(rows: Iterable[dict[str, Any]], *, include_failed: bool = True) -> set[str]:
    out: set[str] = set()
    for r in rows:
        key = r.get("unit_key")
        if not key:
            continue
        status = str(r.get("status", ""))
        if status in ("completed", "invalid"):
            out.add(key)
        elif include_failed and status == "failed":
            out.add(key)
    return out


def rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
        f.flush()


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")
        f.flush()
        try:
            import os

            os.fsync(f.fileno())
        except OSError:
            pass


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Optional[list[str]] = None) -> Path:
    rows = list(rows)
    fields = fieldnames or list(RAW_FIELDS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return path


def rewrite_raw_tables(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(out_dir / "raw_results.csv", rows, RAW_FIELDS)
    failed = [r for r in rows if str(r.get("status")) in ("failed", "invalid")]
    fail_fields = [
        "unit_key",
        "image_id",
        "attack",
        "eps",
        "steps",
        "status",
        "failure_code",
        "failure_message",
        "runtime_sec",
    ]
    write_csv(out_dir / "failures.csv", failed, fail_fields)
