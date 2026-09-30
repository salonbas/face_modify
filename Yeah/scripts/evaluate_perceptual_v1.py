#!/usr/bin/env python3
"""Append-only perceptual evaluation for serialized adversarial-image artifacts.

This never generates attacks and never edits legacy result files.  It evaluates
the exact decoded original/adversarial RGB pairs that still exist on disk.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np
import torch
from DISTS_pytorch import DISTS
from PIL import Image
from skimage.metrics import structural_similarity
import lpips

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "perceptual_v1"
MAX_PERCEPTUAL_SIDE = 256
EPS_RE = re.compile(r"_eps_(\d+\.\d+)\.png$")

FIELDS = [
    "experiment", "attack", "epsilon", "pair_id", "pair_semantics",
    "original_image", "adversarial_image", "original_width", "original_height",
    "linf_serialized_rgb_01", "linf_serialized_rgb_255", "ssim_rgb_native",
    "lpips_alex_resize256", "dists_vgg16_resize256",
    "arcface_clean_cosine", "arcface_adv_cosine", "arcface_threshold",
    "arcface_threshold_crossing", "facenet_clean_cosine", "facenet_adv_cosine",
    "facenet_threshold", "facenet_threshold_crossing", "arcface_cosine_drop",
    "facenet_cosine_drop", "legacy_arcface_self_original_adv_cosine",
    "source_result_csv", "notes",
]


@dataclass(frozen=True)
class Pair:
    experiment: str
    attack: str
    epsilon: float | None
    pair_id: str
    pair_semantics: str
    original: Path
    adversarial: Path
    source: Path
    extra: dict[str, Any]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_legacy_pairs() -> list[Pair]:
    pairs: list[Pair] = []
    for attack_dir, prefix, label in [
        (ROOT / "results" / "fgsm", "fgsm", "fgsm"),
        (ROOT / "results" / "pgd", "pgd", "pgd"),
    ]:
        for experiment_dir in sorted(p for p in attack_dir.iterdir() if p.is_dir()):
            bases = sorted(experiment_dir.glob("base_*.png"))
            if len(bases) != 1:
                continue
            base = bases[0]
            for adv in sorted(experiment_dir.glob("*_eps_*.png")):
                match = EPS_RE.search(adv.name)
                if not match:
                    continue
                pairs.append(Pair(
                    experiment=rel(experiment_dir), attack=label,
                    epsilon=float(match.group(1)), pair_id=adv.stem,
                    pair_semantics="legacy_self_pair_original_vs_adversarial; no independent A/B verification pair",
                    original=base, adversarial=adv,
                    source=experiment_dir / ("fgsm_cosine_metrics.csv" if label == "fgsm" else "pgd_cosine_metrics.csv"),
                    extra={},
                ))
                if not pairs[-1].source.exists():
                    pairs[-1] = Pair(**{**pairs[-1].__dict__, "source": experiment_dir / "pgd_full_cosine_metrics.csv"})
    return pairs


def find_transfer_pairs() -> list[Pair]:
    pairs: list[Pair] = []
    for experiment_dir in sorted((ROOT / "results" / "transfer").glob("*verification*")):
        source = experiment_dir / "results.csv"
        if not source.exists():
            continue
        for row in read_csv(source):
            if row.get("valid", "").lower() != "true" or not row.get("adversarial_image"):
                continue
            attack = "mi_fgsm" if experiment_dir.name.startswith("mi_fgsm") else "pgd_full"
            params = json.loads(row.get("attack_parameters") or "{}")
            epsilon = float(params["eps"]) if "eps" in params else 0.04
            pairs.append(Pair(
                experiment=rel(experiment_dir), attack=attack, epsilon=epsilon,
                pair_id=row["identity_id"], pair_semantics="LFW genuine verification pair: reference A, probe B, adversarial probe B_adv",
                original=ROOT / row["probe_image"], adversarial=ROOT / row["adversarial_image"], source=source,
                extra=row,
            ))
    return pairs


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def resize_for_deep_metric(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(1.0, MAX_PERCEPTUAL_SIDE / max(h, w))
    if scale == 1.0:
        return rgb
    size = (round(w * scale), round(h * scale))
    return np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)


def value(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        raw = row.get(name, "")
        if raw not in ("", None):
            return float(raw)
    return None


def crossing(clean: float | None, adv: float | None, threshold: float | None) -> bool | None:
    return None if None in (clean, adv, threshold) else clean >= threshold and adv < threshold


def fmt(x: Any) -> Any:
    if isinstance(x, float):
        # Keep enough precision for an audit to reproduce a uint8 L∞ / 255.
        return f"{x:.17g}"
    return x


def stats(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "mean": mean(values), "median": median(values), "std_sample": stdev(values) if len(values) > 1 else None}


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["attack"], row["epsilon"])] .append(row)
    by_attack_epsilon = []
    metric_names = ["linf_serialized_rgb_01", "ssim_rgb_native", "lpips_alex_resize256", "dists_vgg16_resize256", "arcface_cosine_drop", "facenet_cosine_drop"]
    for (attack, epsilon), items in sorted(grouped.items()):
        record: dict[str, Any] = {"attack": attack, "epsilon": epsilon, "n": len(items)}
        for name in metric_names:
            values = [float(item[name]) for item in items if item[name] != ""]
            record[name] = stats(values) if values else None
        by_attack_epsilon.append(record)
    correlations = []
    for victim, drop in [("arcface", "arcface_cosine_drop"), ("facenet", "facenet_cosine_drop")]:
        eligible = [r for r in rows if r[drop] != ""]
        for metric in ["ssim_rgb_native", "lpips_alex_resize256", "dists_vgg16_resize256"]:
            correlations.append({"victim": victim, "perceptual_metric": metric, "n": len(eligible), "pearson_r": pearson([float(r[metric]) for r in eligible], [float(r[drop]) for r in eligible])})
    return {"by_attack_epsilon": by_attack_epsilon, "perceptual_to_victim_cosine_drop": correlations}


def report_html(rows: list[dict[str, Any]], aggregate: dict[str, Any], missing: list[dict[str, str]]) -> str:
    def h(x: Any) -> str:
        import html
        return html.escape(str(x))
    aggregate_rows = "".join(
        f"<tr><td>{h(item['attack'])}</td><td>{h(item['epsilon'])}</td><td>{item['n']}</td>"
        + "".join(f"<td>{'' if item[key] is None else h(item[key]['mean'])}</td>" for key in ["ssim_rgb_native", "lpips_alex_resize256", "dists_vgg16_resize256", "arcface_cosine_drop", "facenet_cosine_drop"])
        + "</tr>" for item in aggregate["by_attack_epsilon"])
    pair_rows = "".join("<tr>" + "".join(f"<td>{h(row.get(key, ''))}</td>" for key in ["experiment", "attack", "epsilon", "pair_id", "linf_serialized_rgb_01", "ssim_rgb_native", "lpips_alex_resize256", "dists_vgg16_resize256", "arcface_clean_cosine", "arcface_adv_cosine", "facenet_clean_cosine", "facenet_adv_cosine"]) + "</tr>" for row in rows)
    missing_rows = "".join(f"<li><code>{h(item['experiment'])}</code>: {h(item['reason'])}</li>" for item in missing) or "<li>None</li>"
    return f"""<!doctype html><meta charset='utf-8'><title>Perceptual Evaluation v1</title>
<style>body{{font:14px system-ui;margin:2rem;color:#19222d}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}td,th{{border:1px solid #cbd5e1;padding:.4rem;text-align:left}}th{{background:#eaf2f8}}code{{font-size:.9em}}.note{{color:#475569}}</style>
<h1>Perceptual Evaluation v1</h1><p>{len(rows)} persisted image pairs; append-only evaluation. Generated {datetime.now(timezone.utc).isoformat()}.</p>
<h2>Protocol</h2><p>Images are decoded with Pillow as RGB uint8. L∞ is the maximum absolute decoded serialized-pixel delta / 255. SSIM uses native-resolution RGB arrays with <code>skimage.metrics.structural_similarity(channel_axis=2, data_range=1.0)</code>. For LPIPS and DISTS only, both images are jointly resized with LANCZOS preserving aspect ratio so their longest side is at most 256 px; LPIPS AlexNet receives RGB float [-1,1], and DISTS VGG16 receives RGB float [0,1] (the package applies ImageNet normalization internally).</p>
<h2>Attack × epsilon aggregate (mean; full mean/median/std in JSON)</h2><table><tr><th>attack</th><th>epsilon</th><th>n</th><th>SSIM</th><th>LPIPS</th><th>DISTS</th><th>ArcFace drop</th><th>FaceNet drop</th></tr>{aggregate_rows}</table>
<h2>Pair metrics</h2><p class='note'>Blank clean/adv verification cosine fields mean that the legacy experiment saved only a self-pair cosine and no independent A/B reference image.</p><table><tr><th>experiment</th><th>attack</th><th>eps</th><th>pair/id</th><th>L∞</th><th>SSIM</th><th>LPIPS</th><th>DISTS</th><th>Arc clean</th><th>Arc adv</th><th>Face clean</th><th>Face adv</th></tr>{pair_rows}</table>
<h2>Not retroactively computable</h2><ul>{missing_rows}</ul>"""


def main() -> None:
    torch.set_num_threads(2)
    cache = ROOT / ".cache" / "perceptual_torch"
    os.environ.setdefault("TORCH_HOME", str(cache))
    OUT.mkdir(parents=True, exist_ok=True)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()
    dists_model = DISTS().eval()
    pairs = find_legacy_pairs() + find_transfer_pairs()
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, 1):
        if not pair.original.exists() or not pair.adversarial.exists():
            raise FileNotFoundError(f"Persisted pair missing: {pair.original} / {pair.adversarial}")
        original, adversarial = load_rgb(pair.original), load_rgb(pair.adversarial)
        if original.shape != adversarial.shape:
            raise ValueError(f"Shape mismatch: {pair.original} {original.shape}; {pair.adversarial} {adversarial.shape}")
        unit_original, unit_adv = original.astype(np.float32) / 255.0, adversarial.astype(np.float32) / 255.0
        reduced_original, reduced_adv = resize_for_deep_metric(original), resize_for_deep_metric(adversarial)
        tensor_original = torch.from_numpy(np.ascontiguousarray(reduced_original.transpose(2, 0, 1))).unsqueeze(0).float() / 255.0
        tensor_adv = torch.from_numpy(np.ascontiguousarray(reduced_adv.transpose(2, 0, 1))).unsqueeze(0).float() / 255.0
        with torch.inference_mode():
            lpips_value = float(lpips_model(tensor_original * 2 - 1, tensor_adv * 2 - 1).item())
            dists_value = float(dists_model(tensor_original, tensor_adv).item())
        source = pair.extra
        arc_clean = value(source, "source_clean_cosine", "source_clean_A_B")
        arc_adv = value(source, "source_verification_adv_cosine", "source_adv_A_Badv")
        arc_threshold = value(source, "source_threshold")
        face_clean = value(source, "victim_clean_cosine", "victim_clean_A_B")
        face_adv = value(source, "victim_verification_adv_cosine", "victim_adv_A_Badv")
        face_threshold = value(source, "victim_threshold")
        row: dict[str, Any] = {
            "experiment": pair.experiment, "attack": pair.attack, "epsilon": "" if pair.epsilon is None else pair.epsilon,
            "pair_id": pair.pair_id, "pair_semantics": pair.pair_semantics, "original_image": rel(pair.original), "adversarial_image": rel(pair.adversarial),
            "original_width": original.shape[1], "original_height": original.shape[0],
            "linf_serialized_rgb_01": "", "linf_serialized_rgb_255": "",
            "ssim_rgb_native": float(structural_similarity(unit_original, unit_adv, channel_axis=2, data_range=1.0)), "lpips_alex_resize256": lpips_value, "dists_vgg16_resize256": dists_value,
            "arcface_clean_cosine": arc_clean if arc_clean is not None else "", "arcface_adv_cosine": arc_adv if arc_adv is not None else "", "arcface_threshold": arc_threshold if arc_threshold is not None else "",
            "arcface_threshold_crossing": crossing(arc_clean, arc_adv, arc_threshold), "facenet_clean_cosine": face_clean if face_clean is not None else "", "facenet_adv_cosine": face_adv if face_adv is not None else "", "facenet_threshold": face_threshold if face_threshold is not None else "",
            "facenet_threshold_crossing": crossing(face_clean, face_adv, face_threshold), "arcface_cosine_drop": (arc_clean - arc_adv) if None not in (arc_clean, arc_adv) else "", "facenet_cosine_drop": (face_clean - face_adv) if None not in (face_clean, face_adv) else "",
            "legacy_arcface_self_original_adv_cosine": "", "source_result_csv": rel(pair.source) if pair.source.exists() else "", "notes": "",
        }
        # Legacy reported values are intentionally preserved as legacy values; they
        # are not relabelled as A/B verification cosines.
        if not source:
            metric_file = pair.source
            if metric_file.exists():
                for legacy in read_csv(metric_file):
                    if math.isclose(float(legacy["eps"]), pair.epsilon or -1):
                        row["legacy_arcface_self_original_adv_cosine"] = legacy.get("cosine_similarity", "")
                        break
            row["notes"] = "No persisted independent A/B verification reference; legacy ArcFace self cosine retained separately."
        linf_255 = int(np.abs(original.astype(np.int16) - adversarial.astype(np.int16)).max())
        row["linf_serialized_rgb_255"] = linf_255
        row["linf_serialized_rgb_01"] = linf_255 / 255.0
        rows.append({key: fmt(row[key]) for key in FIELDS})
        print(f"[{index}/{len(pairs)}] {pair.experiment} {pair.pair_id}")
    missing = [
        {"experiment": "results/robustness/lfw_verification_pgd_smoke", "reason": "results.csv has metrics but no persisted B_adv image; cannot calculate image-pair perceptual metrics."},
        {"experiment": "results/robustness/lfw_verification_pgd_v1", "reason": "results.csv has metrics but no persisted B_adv image; transfer PGD artifacts are a separately regenerated run and are evaluated under their own experiment."},
    ]
    with (OUT / "perceptual_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    aggregate = summary(rows)
    aggregate_csv_rows = []
    for item in aggregate["by_attack_epsilon"]:
        for metric in ("linf_serialized_rgb_01", "ssim_rgb_native", "lpips_alex_resize256", "dists_vgg16_resize256", "arcface_cosine_drop", "facenet_cosine_drop"):
            values = item[metric]
            aggregate_csv_rows.append({"attack": item["attack"], "epsilon": item["epsilon"], "metric": metric, **(values or {"n": 0, "mean": "", "median": "", "std_sample": ""})})
    with (OUT / "aggregate_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attack", "epsilon", "metric", "n", "mean", "median", "std_sample"])
        writer.writeheader(); writer.writerows(aggregate_csv_rows)
    with (OUT / "perceptual_to_victim_cosine_drop.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["victim", "perceptual_metric", "n", "pearson_r"])
        writer.writeheader(); writer.writerows(aggregate["perceptual_to_victim_cosine_drop"])
    metadata = {"title": "Perceptual Evaluation v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "metric_implementations": {"linf": "NumPy decoded RGB uint8 absolute maximum / 255", "ssim": "scikit-image 0.26.0 structural_similarity, RGB native geometry, data_range=1.0", "lpips": "lpips 0.1.4, learned AlexNet, RGB [-1,1]", "dists": "DISTS-pytorch 0.1, learned VGG16, RGB [0,1], internal ImageNet normalization"}, "deep_metric_resize": {"method": "Pillow LANCZOS", "preserve_aspect_ratio": True, "max_long_side": MAX_PERCEPTUAL_SIDE}, "missing": missing, "aggregate": aggregate}
    (OUT / "aggregate_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "report_v2.html").write_text(report_html(rows, aggregate, missing), encoding="utf-8")
    print(f"Wrote {OUT / 'perceptual_metrics.csv'}")


if __name__ == "__main__":
    main()
