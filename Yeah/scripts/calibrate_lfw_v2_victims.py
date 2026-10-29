#!/usr/bin/env python3
"""Calibrate an author-verified victim on isolated LFW v2 pairs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from advface.calibration import operating_metrics, select_far_threshold
from advface.evaluation.similarity import cosine_similarity
from advface.image_io import load_bgr
from advface.models import load_embedder

SPLITS = ROOT / "data/datasets/lfw/evaluation_splits/v2"
CHECKPOINTS = {
    "mobilefacenet": ROOT / "models/insightface/buffalo_sc/w600k_mbf.onnx",
    "adaface": ROOT / "models/adaface/adaface_ir50_ms1mv2.ckpt",
    "magface": ROOT / "models/magface/magface_iresnet100_ms1mv2.pth",
}
PREPROCESSING = {
    "mobilefacenet": "insightface-buffalo-sc-norm-crop-112-v1",
    "adaface": "adaface-author-mtcnn-bgr-112-v1",
    "magface": "magface-insightface-fivepoint-bgr-112-v1",
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), default="mobilefacenet")
    args = parser.parse_args()
    name = args.model
    out = ROOT / f"results/calibration/{name}_lfw_v2"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite calibration artifacts: {out}")
    attack_ids = {r["identity_id"] for part in ("development", "oos")
                  for r in read(SPLITS / f"{part}.csv")}
    pairs = {part: read(SPLITS / "calibration_pairs" / f"{part}.csv")
             for part in ("dev_train", "dev_test")}
    if any(r[k] in attack_ids for section in pairs.values() for r in section
           for k in ("identity_a", "identity_b")):
        raise RuntimeError("calibration identity isolation failed")
    expected = json.loads((ROOT / "models/checkpoint_manifest.json").read_text())[name]["sha256"]
    checksum = hashlib.sha256(CHECKPOINTS[name].read_bytes()).hexdigest()
    if checksum != expected:
        raise RuntimeError(f"{name} checkpoint checksum mismatch")
    model = load_embedder(name)
    cache: dict[str, np.ndarray | None] = {}
    failures: dict[str, str] = {}

    def embedding(identity: str, image_id: str):
        if image_id not in cache:
            path = ROOT / "data/datasets/lfw/images" / identity / f"{image_id}.jpg"
            try:
                cache[image_id] = model.get_embedding(load_bgr(path), image_id)
            except (OSError, ValueError, RuntimeError) as exc:
                cache[image_id] = None
                failures[image_id] = f"{type(exc).__name__}: {exc}"
        return cache[image_id]

    started = time.perf_counter()
    scored = []
    arrays = {}
    for part, section in pairs.items():
        for i, pair in enumerate(section, 1):
            a = embedding(pair["identity_a"], pair["image_a"])
            b = embedding(pair["identity_b"], pair["image_b"])
            score = float(cosine_similarity(a, b)) if a is not None and b is not None else None
            scored.append({"pair_id": f"{part}_{i:04d}", "split": part, **pair,
                           "cosine_similarity": score if score is not None else "", "valid": score is not None})
        valid = [r for r in scored if r["split"] == part and r["valid"]]
        arrays[part] = (np.array([r["cosine_similarity"] for r in valid], dtype=float),
                        np.array([r["same_identity"].lower() == "true" for r in valid], dtype=bool))
        print(f"{part}: {len(valid)}/{len(section)} valid; {len(cache)} image embeddings", flush=True)
    train_scores, train_genuine = arrays["dev_train"]
    threshold = select_far_threshold(train_scores[~train_genuine], 0.01)
    threshold["development"] = operating_metrics(*arrays["dev_train"], threshold["threshold"])
    threshold["held_out_test"] = operating_metrics(*arrays["dev_test"], threshold["threshold"])
    out.mkdir(parents=True)
    with (out / "pair_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
        writer.writeheader(); writer.writerows(scored)
    metadata = {"model": name, "checkpoint_sha256": checksum,
                "preprocessing_version": PREPROCESSING[name],
                "source": json.loads((ROOT / "models/checkpoint_manifest.json").read_text())[name]["source"],
                "selection": "FAR=1e-2 on isolated dev_train impostor pairs only",
                "test_usage": "evaluation only", "attack_identity_overlap": 0,
                "pair_counts": {part: {"defined": len(pairs[part]), "valid": len(arrays[part][0]),
                                       "genuine": int(arrays[part][1].sum()),
                                       "impostor": int((~arrays[part][1]).sum())} for part in pairs},
                "embedding_failures": failures, "unique_images_processed": len(cache),
                "runtime_seconds": time.perf_counter() - started}
    (out / "thresholds.json").write_text(json.dumps({"far_1e-2": threshold}, indent=2) + "\n")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"threshold": threshold, "metadata": {k: v for k, v in metadata.items() if k != "embedding_failures"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
