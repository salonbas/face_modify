#!/usr/bin/env python3
"""Create the append-only 20-identity four-method formal comparison report."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/transfer/lfw_attack_dev_20_four_method_v3"
SPLIT = ROOT / "data/datasets/lfw/evaluation_splits/attack_dev.csv"
METHODS = {
    "PGD": ROOT / "results/transfer/lfw_attack_dev_20_baseline_v2/pgd_full",
    "MI-FGSM": ROOT / "results/transfer/lfw_attack_dev_20_baseline_v2/mi_fgsm",
    "PGD+Mask": OUT / "pgd_landmark_superpixel_mask",
    "MI-FGSM+Mask": OUT / "mi_fgsm_landmark_superpixel_mask",
}


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_rows(name: str, directory: Path, pairs: dict[str, dict[str, str]]) -> list[dict]:
    with (directory / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20 or {r["identity_id"] for r in rows} != set(pairs):
        raise RuntimeError(f"{name}: results.csv is not the complete frozen 20-identity set")
    output = []
    for row in rows:
        ident = row["identity_id"]
        if row["valid"] != "True":
            raise RuntimeError(f"{name}/{ident}: invalid worker record: {row['failure_reason']}")
        if float(row["configured_epsilon"]) != 0.04 or int(row["steps"]) != 200 or float(row["tv_weight"]) != 0:
            raise RuntimeError(f"{name}/{ident}: attack contract mismatch")
        artifact = ROOT / row["adversarial_image"]
        probe = ROOT / pairs[ident]["probe_image"]
        if not artifact.is_file():
            raise RuntimeError(f"{name}/{ident}: missing B_adv: {artifact}")
        clean, saved = load_rgb(probe), load_rgb(artifact)
        if clean.shape != saved.shape:
            raise RuntimeError(f"{name}/{ident}: serialized B_adv dimensions differ")
        actual255 = int(np.abs(clean.astype(np.int16) - saved.astype(np.int16)).max())
        if actual255 > 11:
            raise RuntimeError(f"{name}/{ident}: serialized L_inf {actual255}/255 exceeds epsilon plus quantization")
        row = dict(row)
        row.update(method=name, actual_linf_serialized_rgb_01=actual255 / 255.0,
                   actual_linf_serialized_rgb_255=actual255,
                   arcface_asr=row["arcface_threshold_crossing"],
                   facenet_transfer_asr=row["facenet_transfer_crossing"])
        if name.endswith("+Mask"):
            # The per-worker readback check is authoritative; require its stored
            # result here as a second formal aggregation gate.
            if float(row["mask_outside_delta_max"]) != 0:
                raise RuntimeError(f"{name}/{ident}: non-zero serialized mask-outside delta")
        output.append(row)
    return output


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(r[key]) for r in rows]))


def aggregate(name: str, rows: list[dict]) -> dict:
    return {
        "method": name, "sample_count": len(rows), "configured_epsilon": 0.04,
        "actual_linf_serialized_rgb_01": mean(rows, "actual_linf_serialized_rgb_01"),
        "actual_linf_serialized_rgb_255": mean(rows, "actual_linf_serialized_rgb_255"),
        "arcface_clean_cosine": mean(rows, "arcface_clean_cosine"),
        "arcface_adv_cosine": mean(rows, "arcface_adv_cosine"),
        "arcface_cosine_drop": mean(rows, "arcface_cosine_drop"),
        "arcface_asr": sum(r["arcface_threshold_crossing"] == "True" for r in rows) / len(rows),
        "facenet_clean_cosine": mean(rows, "facenet_clean_cosine"),
        "facenet_adv_cosine": mean(rows, "facenet_adv_cosine"),
        "facenet_cosine_drop": mean(rows, "facenet_cosine_drop"),
        "facenet_transfer_asr": sum(r["facenet_transfer_crossing"] == "True" for r in rows) / len(rows),
        "ssim": mean(rows, "ssim"), "lpips": mean(rows, "lpips"), "dists": mean(rows, "dists"),
    }


def html(rows: list[dict], summaries: list[dict]) -> str:
    cols = ("method", "arcface_cosine_drop", "arcface_asr", "facenet_cosine_drop", "facenet_transfer_asr", "ssim", "lpips", "dists", "actual_linf_serialized_rgb_01")
    summary_html = "".join("<tr>" + "".join(f"<td>{escape(str(r[c]))}</td>" for c in cols) + "</tr>" for r in summaries)
    per_cols = ("identity_id", "method", "arcface_clean_cosine", "arcface_adv_cosine", "arcface_cosine_drop", "arcface_threshold_crossing", "facenet_clean_cosine", "facenet_adv_cosine", "facenet_cosine_drop", "facenet_transfer_crossing", "configured_epsilon", "actual_linf_serialized_rgb_01", "actual_linf_serialized_rgb_255", "ssim", "lpips", "dists", "mask_outside_delta_max", "adversarial_image")
    per_html = "".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in per_cols) + "</tr>" for r in rows)
    return f"<!doctype html><meta charset=utf-8><title>LFW four-method comparison v3</title><style>body{{font-family:system-ui;margin:2rem}}table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #bbb;padding:4px;white-space:nowrap}}th{{background:#eee}}</style><h1>LFW frozen attack_dev: four-method comparison v3</h1><p>n=20 per method. All metrics are read from each method's decoded serialized B_adv PNG; no test or reserve identities.</p><h2>Unified summary</h2><table><tr>{''.join(f'<th>{c}</th>' for c in cols)}</tr>{summary_html}</table><h2>Per-identity results</h2><table><tr>{''.join(f'<th>{c}</th>' for c in per_cols)}</tr>{per_html}</table>"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with SPLIT.open(encoding="utf-8", newline="") as handle:
        pairs_list = list(csv.DictReader(handle))
    if len(pairs_list) != 20 or any(p["split"] != "attack_dev" for p in pairs_list):
        raise RuntimeError("frozen split contract failed")
    pairs = {p["identity_id"]: p for p in pairs_list}
    all_rows = [row for name, directory in METHODS.items() for row in read_rows(name, directory, pairs)]
    summaries = [aggregate(name, [r for r in all_rows if r["method"] == name]) for name in METHODS]
    fields = list(all_rows[0])
    with (OUT / "per_identity_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    fields = list(summaries[0])
    with (OUT / "unified_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summaries)
    sanity = {"passed": True, "frozen_split": "attack_dev.csv", "no_test_or_reserve": True, "methods": {name: {"records": 20, "valid": 20, "serialized_B_adv_present": True, "serialized_linf_within_epsilon_plus_uint8_quantization": True, "mask_outside_serialized_delta_zero": name.endswith("+Mask")} for name in METHODS}, "all_metrics_from_serialized_B_adv": True}
    metadata = {"title": "LFW frozen attack_dev 20-identity four-method comparison v3", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "formal": True, "methods": {name: str(directory.relative_to(ROOT)) for name, directory in METHODS.items()}, "attack_contract": {"configured_epsilon": 0.04, "steps": 200, "tv_weight": 0, "mask": "68-point landmarks (eyebrows/eyes/nose/mouth), SLIC expansion, gradient *= mask"}, "provenance": {"baseline_methods": "Existing formal v2 serialized B_adv artifacts are retained without mutation.", "mask_methods": "Formal v3 per-identity subprocess outputs."}, "sanity_check": sanity, "aggregate": summaries}
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "report.html").write_text(html(all_rows, summaries), encoding="utf-8")
    print(json.dumps({"sanity_check": sanity, "aggregate": summaries}, indent=2))


if __name__ == "__main__":
    main()
