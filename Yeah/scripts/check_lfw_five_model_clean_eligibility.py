#!/usr/bin/env python3
"""Score Development 100 clean A/B only; never generates an attack."""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from advface.evaluation.multi_victim import MODELS, load_v2_thresholds, summarize
from advface.evaluation.similarity import cosine_similarity
from advface.image_io import load_bgr
from advface.models import load_embedder

MANIFEST = ROOT / "data/datasets/lfw/evaluation_splits/v2/development.csv"
OUTPUT = ROOT / "results/transfer/lfw_dev_100_five_model_v1/clean_eligibility.json"


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUTPUT}")
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    if len(pairs) != 100 or len({r["identity_id"] for r in pairs}) != 100:
        raise RuntimeError("Development 100 manifest is invalid")
    thresholds = load_v2_thresholds(ROOT)
    rows = [{"identity_id": pair["identity_id"], "reference_image": pair["reference_image"],
             "probe_image": pair["probe_image"], "models": {}} for pair in pairs]
    timings = {}
    for name in MODELS:
        started = time.perf_counter()
        embedder = load_embedder(name)
        for pair, record in zip(pairs, rows):
            ident = pair["identity_id"]
            for side in ("reference", "probe"):
                path = ROOT / pair[f"{side}_image"]
                if not path.is_file() or path.parent.name != ident:
                    raise RuntimeError(f"invalid A/B image for {ident}: {path}")
            try:
                a = embedder.get_embedding(load_bgr(ROOT / pair["reference_image"]), f"{ident}:A")
                b = embedder.get_embedding(load_bgr(ROOT / pair["probe_image"]), f"{ident}:B")
                score = float(cosine_similarity(a, b))
                record["models"][name] = {"clean_cosine": score, "eligible": score >= thresholds[name]}
            except (OSError, ValueError, RuntimeError) as exc:
                record["models"][name] = {"clean_cosine": None, "eligible": False,
                                          "error": f"{type(exc).__name__}: {exc}"}
        timings[name] = time.perf_counter() - started
        print(f"{name}: {timings[name]:.2f} s for 200 clean images", flush=True)
        del embedder
        gc.collect()
    report = {"manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              "thresholds": thresholds, "timings_seconds": timings,
              "summary": summarize(rows, thresholds), "rows": rows}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(OUTPUT)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
