#!/usr/bin/env python3
"""Rebuild the formal LFW PGD vs MI-FGSM perceptual evaluation.

This evaluator deliberately has no result-directory discovery or globbing.
Its immutable allowlist is the five LFW development identities and the two
named formal attack authorities below.  It never generates adversarial images,
and it reads calibration thresholds only as recorded by the formal runs.
"""
from __future__ import annotations

import csv
import html
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import lpips
import numpy as np
import torch
from DISTS_pytorch import DISTS
from PIL import Image
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/perceptual/lfw_development_transfer_v2"
IDENTITIES = (
    "Michelle_Collins", "Meryl_Streep", "Jolanta_Kwasniewski",
    "Dino_de_Laurentis", "Cyndi_Thompson",
)
PGD_ROBUSTNESS = ROOT / "results/robustness/lfw_verification_pgd_v1"
PGD_TRANSFER = ROOT / "results/transfer/pgd_arcface_to_facenet_verification_v1"
MI_TRANSFER = ROOT / "results/transfer/mi_fgsm_arcface_to_facenet_verification_v1"
MAX_PERCEPTUAL_SIDE = 256

FIELDS = [
    "identity_id", "attack", "configured_epsilon", "configured_epsilon_255",
    "actual_linf_serialized_rgb_01", "actual_linf_serialized_rgb_255",
    "actual_linf_tensor_pre_serialization", "reference_image_a", "probe_image_b",
    "adversarial_image_b_adv", "image_width", "image_height", "ssim_rgb_native",
    "lpips_alex_resize256", "dists_vgg16_resize256", "arcface_clean_cosine",
    "arcface_adversarial_cosine", "arcface_cosine_drop", "arcface_threshold",
    "arcface_threshold_crossing", "facenet_clean_cosine",
    "facenet_adversarial_cosine", "facenet_cosine_drop", "facenet_threshold",
    "facenet_threshold_crossing", "source_authority_metrics",
    "perceptual_image_artifact_provenance",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def by_identity(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    found = {row["identity_id"]: row for row in rows if row.get("valid", "").lower() == "true"}
    if set(found) != set(IDENTITIES):
        raise ValueError(f"{rel(path)} must contain exactly the formal identities; got {sorted(found)}")
    return found


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in (None, ""):
        raise ValueError(f"missing {field} for {row.get('identity_id')}")
    return float(value)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def resize_for_deep_metric(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(1.0, MAX_PERCEPTUAL_SIDE / max(height, width))
    if scale == 1.0:
        return rgb
    return np.asarray(Image.fromarray(rgb).resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS), dtype=np.uint8)


def crossing(clean: float, adversarial: float, threshold: float) -> bool:
    return clean >= threshold and adversarial < threshold


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values), "mean": mean(values), "median": median(values),
        "std_sample": stdev(values) if len(values) > 1 else None,
    }


def validate_pair(identity: str, attack: str, source: dict[str, str], adversarial: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_a = ROOT / f"data/datasets/lfw/images/{identity}/{identity}_0001.jpg"
    expected_b = ROOT / f"data/datasets/lfw/images/{identity}/{identity}_0002.jpg"
    expected_adv = ROOT / "results/transfer" / (
        f"pgd_arcface_to_facenet_verification_v1/adversarial_probes/{identity}_B_adv.png"
        if attack == "PGD" else
        f"mi_fgsm_arcface_to_facenet_verification_v1/adversarial_probes/{identity}_B_adv_MI.png"
    )
    for label, actual, expected in (
        ("reference A", ROOT / source["reference_image"], expected_a),
        ("probe B", ROOT / source["probe_image"], expected_b),
        ("B_adv", adversarial, expected_adv),
    ):
        if actual != expected or not actual.is_file():
            raise ValueError(f"{identity} {attack}: invalid {label}: {actual}; expected {expected}")
    a, b, b_adv = load_rgb(expected_a), load_rgb(expected_b), load_rgb(expected_adv)
    if a.shape != b.shape or b.shape != b_adv.shape:
        raise ValueError(f"{identity} {attack}: A/B/B_adv shape mismatch: {a.shape}, {b.shape}, {b_adv.shape}")
    return a, b, b_adv


def source_values(attack: str, identity: str, pgd_robust: dict[str, dict[str, str]], pgd_transfer: dict[str, dict[str, str]], mi_transfer: dict[str, dict[str, str]]) -> tuple[dict[str, str], dict[str, float], str, str]:
    if attack == "PGD":
        robust, transfer = pgd_robust[identity], pgd_transfer[identity]
        params = json.loads(robust["perturbation_configuration"])
        values = {
            "configured_epsilon": float(params["eps"]), "configured_epsilon_255": float(params["eps_255"]),
            "actual_linf_tensor_pre_serialization": number(robust, "linf_tensor"),
            "arcface_clean_cosine": number(robust, "clean_cosine_A_B"),
            "arcface_adversarial_cosine": number(robust, "modified_cosine_A_Bmodified"),
            "arcface_threshold": number(robust, "primary_threshold"),
            "facenet_clean_cosine": number(transfer, "victim_clean_A_B"),
            "facenet_adversarial_cosine": number(transfer, "victim_adv_A_Badv"),
            "facenet_threshold": number(transfer, "victim_threshold"),
        }
        return transfer, values, rel(PGD_ROBUSTNESS / "results.csv"), "regenerated_after_robustness_run_same_configuration; transfer artifact is not claimed to be an original robustness-run artifact"
    row = mi_transfer[identity]
    params = json.loads(row["attack_parameters"])
    values = {
        "configured_epsilon": float(params["eps"]), "configured_epsilon_255": float(params["eps_255"]),
        "actual_linf_tensor_pre_serialization": number(row, "linf_tensor"),
        "arcface_clean_cosine": number(row, "source_clean_cosine"),
        "arcface_adversarial_cosine": number(row, "source_verification_adv_cosine"),
        "arcface_threshold": number(row, "source_threshold"),
        "facenet_clean_cosine": number(row, "victim_clean_cosine"),
        "facenet_adversarial_cosine": number(row, "victim_verification_adv_cosine"),
        "facenet_threshold": number(row, "victim_threshold"),
    }
    return row, values, rel(MI_TRANSFER / "results.csv"), "formal MI-FGSM transfer run persisted B_adv artifact"


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["attack"]].append(row)
    records = []
    metrics = (
        "configured_epsilon", "configured_epsilon_255", "actual_linf_serialized_rgb_01",
        "actual_linf_serialized_rgb_255", "ssim_rgb_native", "lpips_alex_resize256",
        "dists_vgg16_resize256", "arcface_cosine_drop", "facenet_cosine_drop",
    )
    for attack in ("PGD", "MI-FGSM"):
        items = grouped[attack]
        record: dict[str, Any] = {"attack": attack, "sample_count": len(items)}
        for metric in metrics:
            record[metric] = stats([float(item[metric]) for item in items])
        record["arcface_asr"] = sum(bool(item["arcface_threshold_crossing"]) for item in items) / len(items)
        record["facenet_transfer_asr"] = sum(bool(item["facenet_threshold_crossing"]) for item in items) / len(items)
        records.append(record)
    return records


def render_html(rows: list[dict[str, Any]], summary: list[dict[str, Any]], sanity: dict[str, Any]) -> str:
    def f(value: Any) -> str:
        return html.escape(str(value))
    columns = ("attack", "sample_count", "configured_epsilon", "actual_linf_serialized_rgb_01", "ssim_rgb_native", "lpips_alex_resize256", "dists_vgg16_resize256", "arcface_cosine_drop", "facenet_cosine_drop", "arcface_asr", "facenet_transfer_asr")
    def aggregate_value(item: dict[str, Any], key: str) -> Any:
        value = item.get(key, "")
        return f"{value['mean']:.6g} / {value['median']:.6g}" if isinstance(value, dict) else value
    aggregate_rows = "".join("<tr>" + "".join(f"<td>{f(aggregate_value(item, key))}</td>" for key in columns) + "</tr>" for item in summary)
    pair_columns = ("identity_id", "attack", "configured_epsilon", "actual_linf_serialized_rgb_01", "ssim_rgb_native", "lpips_alex_resize256", "dists_vgg16_resize256", "arcface_cosine_drop", "facenet_cosine_drop", "arcface_threshold_crossing", "facenet_threshold_crossing")
    pair_rows = "".join("<tr>" + "".join(f"<td>{f(row[key])}</td>" for key in pair_columns) + "</tr>" for row in rows)
    return f"""<!doctype html><meta charset='utf-8'><title>LFW Perceptual Evaluation v2</title>
<style>body{{font:14px system-ui;margin:2rem;color:#19222d}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}td,th{{border:1px solid #cbd5e1;padding:.4rem;text-align:left}}th{{background:#eaf2f8}}code{{font-size:.9em}}</style>
<h1>LFW development transfer perceptual evaluation v2</h1>
<p>Exactly 10 allowlisted LFW pairs: PGD (n=5) and MI-FGSM (n=5). The report is restricted to those fixed pairs.</p>
<p><b>PGD provenance:</b> ArcFace authority metrics come from <code>results/robustness/lfw_verification_pgd_v1/results.csv</code>; its B_adv was not persisted there. Perceptual B_adv is the later regenerated same-configuration transfer artifact, and is never represented as an original robustness-run artifact.</p>
<h2>Aggregate (mean / median; ASR is a rate)</h2><table><tr>{''.join(f'<th>{f(x)}</th>' for x in columns)}</tr>{aggregate_rows}</table>
<h2>Ten fixed pairs</h2><table><tr>{''.join(f'<th>{f(x)}</th>' for x in pair_columns)}</tr>{pair_rows}</table>
<h2>Sanity check</h2><p>Pairing, identity, dimensions, and serialized perturbation-bound checks passed. Counts: PGD={sanity['counts']['PGD']}, MI-FGSM={sanity['counts']['MI-FGSM']}, total={sanity['counts']['total']}.</p>"""


def main() -> None:
    torch.set_num_threads(2)
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/perceptual_torch"))
    pgd_robust = by_identity(PGD_ROBUSTNESS / "results.csv")
    pgd_transfer = by_identity(PGD_TRANSFER / "results.csv")
    mi_transfer = by_identity(MI_TRANSFER / "results.csv")
    lpips_model, dists_model = lpips.LPIPS(net="alex", verbose=False).eval(), DISTS().eval()
    rows: list[dict[str, Any]] = []
    sanity = {"allowlist_identities": list(IDENTITIES), "pairing_correct": True, "dimensions_consistent": True, "no_smoke_or_legacy_samples": True, "serialized_linf_within_configured_epsilon_plus_uint8_quantization": True, "counts": {"PGD": 0, "MI-FGSM": 0, "total": 0}}
    for attack in ("PGD", "MI-FGSM"):
        for identity in IDENTITIES:
            source, values, authority, provenance = source_values(attack, identity, pgd_robust, pgd_transfer, mi_transfer)
            adversarial = ROOT / source["adversarial_image"]
            _, probe, b_adv = validate_pair(identity, attack, source, adversarial)
            serialized_255 = int(np.abs(probe.astype(np.int16) - b_adv.astype(np.int16)).max())
            serialized_01 = serialized_255 / 255.0
            # A source eps may serialize one uint8 level above eps*255, hence one level tolerance.
            if serialized_255 > values["configured_epsilon_255"] + 1.000001:
                sanity["serialized_linf_within_configured_epsilon_plus_uint8_quantization"] = False
                raise ValueError(f"{identity} {attack}: serialized L∞ exceeds configured epsilon plus quantization")
            reduced_probe, reduced_adv = resize_for_deep_metric(probe), resize_for_deep_metric(b_adv)
            tensor_probe = torch.from_numpy(np.ascontiguousarray(reduced_probe.transpose(2, 0, 1))).unsqueeze(0).float() / 255.0
            tensor_adv = torch.from_numpy(np.ascontiguousarray(reduced_adv.transpose(2, 0, 1))).unsqueeze(0).float() / 255.0
            with torch.inference_mode():
                lpips_value = float(lpips_model(tensor_probe * 2 - 1, tensor_adv * 2 - 1).item())
                dists_value = float(dists_model(tensor_probe, tensor_adv).item())
            arc_drop = values["arcface_clean_cosine"] - values["arcface_adversarial_cosine"]
            face_drop = values["facenet_clean_cosine"] - values["facenet_adversarial_cosine"]
            rows.append({
                "identity_id": identity, "attack": attack, **values,
                "actual_linf_serialized_rgb_01": serialized_01, "actual_linf_serialized_rgb_255": serialized_255,
                "reference_image_a": source["reference_image"], "probe_image_b": source["probe_image"],
                "adversarial_image_b_adv": rel(adversarial), "image_width": probe.shape[1], "image_height": probe.shape[0],
                "ssim_rgb_native": float(structural_similarity(probe.astype(np.float32) / 255, b_adv.astype(np.float32) / 255, channel_axis=2, data_range=1.0)),
                "lpips_alex_resize256": lpips_value, "dists_vgg16_resize256": dists_value,
                "arcface_cosine_drop": arc_drop, "arcface_threshold_crossing": crossing(values["arcface_clean_cosine"], values["arcface_adversarial_cosine"], values["arcface_threshold"]),
                "facenet_cosine_drop": face_drop, "facenet_threshold_crossing": crossing(values["facenet_clean_cosine"], values["facenet_adversarial_cosine"], values["facenet_threshold"]),
                "source_authority_metrics": authority, "perceptual_image_artifact_provenance": provenance,
            })
            sanity["counts"][attack] += 1
            sanity["counts"]["total"] += 1
    if sanity["counts"] != {"PGD": 5, "MI-FGSM": 5, "total": 10}:
        raise AssertionError(f"unexpected sample counts: {sanity['counts']}")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "perceptual_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    summary = aggregate(rows)
    summary_rows = []
    for item in summary:
        for metric, value in item.items():
            if isinstance(value, dict):
                summary_rows.append({"attack": item["attack"], "sample_count": item["sample_count"], "metric": metric, **value})
            elif metric.endswith("asr"):
                summary_rows.append({"attack": item["attack"], "sample_count": item["sample_count"], "metric": metric, "n": item["sample_count"], "mean": value, "median": value, "std_sample": ""})
    with (OUT / "aggregate_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attack", "sample_count", "metric", "n", "mean", "median", "std_sample"]); writer.writeheader(); writer.writerows(summary_rows)
    metadata = {"title": "LFW development transfer perceptual evaluation v2", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "fixed 5 identity LFW development comparison; not a general transfer-rate estimate", "allowlist": {"identities": list(IDENTITIES), "attacks": ["PGD", "MI-FGSM"], "pair_count": 10}, "provenance": {"PGD": "ArcFace authority metrics: robustness run; perceptual artifact: later regenerated transfer B_adv under same configuration", "MI-FGSM": "metrics and persisted B_adv: formal MI-FGSM transfer run"}, "metric_protocol": {"actual_linf_serialized_rgb_01": "decoded RGB uint8 maximum absolute delta / 255", "actual_linf_serialized_rgb_255": "decoded RGB uint8 maximum absolute delta", "ssim": "RGB native geometry, data_range=1.0", "lpips": "AlexNet RGB [-1,1], jointly LANCZOS resized to max 256 px", "dists": "VGG16 RGB [0,1], jointly LANCZOS resized to max 256 px"}, "sanity_check": sanity, "aggregate": summary}
    (OUT / "aggregate_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "report_v2.html").write_text(render_html(rows, summary, sanity), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
