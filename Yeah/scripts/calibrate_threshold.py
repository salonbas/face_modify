#!/usr/bin/env python3
"""Calibrate a canonical embedding model's verification threshold on LFW pairs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from advface.calibration import distribution, eer, labels_from_pairs, operating_metrics, roc_rows, select_far_threshold
from advface.config import ARCFACE_ONNX_FILENAME, EMBEDDING_DIM, FACE_MODEL_NAME, project_root
from advface.image_io import load_bgr
from advface.models import load_embedder
from advface.models.facenet import FACENET_WEIGHT_FILENAME, facenet_pretrained_weight_path
from advface.models.insightface_app import create_face_app, get_embedding_from_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="lfw", choices=["lfw"])
    parser.add_argument("--model", default="insightface_buffalo_l", choices=["insightface_buffalo_l", "facenet_vggface2"])
    parser.add_argument("--run-name", default="arcface_lfw_v1")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def load_protocol(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, dict[str, str]]]:
    data = root / "data/datasets/lfw"
    images = {row["image_id"]: row for row in read_csv(data / "manifest.csv")}
    train = read_csv(data / "pairs/dev_train.csv")
    test = read_csv(data / "pairs/dev_test.csv")
    if not train or not test:
        raise RuntimeError("official LFW development pair CSVs are empty")
    return train, test, images


def cache_metadata(model: str = "insightface_buffalo_l") -> dict[str, Any]:
    if model == "facenet_vggface2":
        return {"model_id": "facenet_vggface2", "weights": FACENET_WEIGHT_FILENAME,
                "weights_path": str(facenet_pretrained_weight_path()), "embedding_dimension": 512,
                "embedding_normalized": True,
                "preprocessing_alignment": "FaceNetEmbedder: MTCNN detect/crop (160x160, post_process=True) -> InceptionResnetV1 VGGFace2 -> L2 normalization"}
    return {"model_id": FACE_MODEL_NAME, "recognition_model": ARCFACE_ONNX_FILENAME, "embedding_dimension": EMBEDDING_DIM,
            "embedding_normalized": True, "preprocessing_alignment": "InsightFace FaceAnalysis detection -> alignment -> face.normed_embedding"}


def extract_embeddings(root: Path, rows: list[dict[str, str]], images: dict[str, dict[str, str]], out: Path, failures: list[dict[str, str]], model_name: str = "insightface_buffalo_l") -> dict[str, np.ndarray]:
    required = sorted({row[key] for row in rows for key in ("image_a", "image_b")})
    cache = out / "embeddings.npz"
    meta = out / "embeddings_metadata.json"
    if cache.exists() and meta.exists():
        try:
            saved_meta = json.loads(meta.read_text(encoding="utf-8"))
            loaded = np.load(cache, allow_pickle=False)
            if saved_meta == cache_metadata(model_name) and set(loaded.files) == set(required):
                print(f"[2/4] Reusing embedding cache: {len(required)} images")
                return {key: loaded[key] for key in loaded.files}
        except Exception as exc:  # cache is optional; extract cleanly instead
            print(f"[2/4] Ignoring unusable cache: {exc}")
    print(f"[2/4] Extracting embeddings: 0/{len(required)}")
    embedder = load_embedder(model_name) if model_name == "facenet_vggface2" else None
    app = create_face_app() if model_name == "insightface_buffalo_l" else None
    embeddings: dict[str, np.ndarray] = {}
    for index, image_id in enumerate(required, 1):
        entry = images.get(image_id)
        path = root / entry["image_path"] if entry else None
        try:
            if entry is None or path is None or not path.exists():
                raise FileNotFoundError(f"manifest image missing: {image_id}")
            embedding = (embedder.get_embedding(load_bgr(path), label=image_id) if embedder is not None
                         else get_embedding_from_path(app, path))
            if embedding.shape != (512,) or not np.isfinite(embedding).all():
                raise ValueError(f"invalid embedding shape/values: {embedding.shape}")
            embeddings[image_id] = embedding
        except Exception as exc:
            failures.append({"image_id": image_id, "path": str(path or ""), "pair_id": "", "failure_type": type(exc).__name__, "message": str(exc)})
        if index == len(required) or index % 50 == 0:
            print(f"[2/4] Extracting embeddings: {index}/{len(required)}")
    np.savez_compressed(cache, **embeddings)
    meta.write_text(json.dumps(cache_metadata(model_name), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return embeddings


def score_pairs(split: str, rows: list[dict[str, str]], images: dict[str, dict[str, str]], embeddings: dict[str, np.ndarray], failures: list[dict[str, str]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for number, pair in enumerate(rows, 1):
        pair_id = f"{split}_{number:04d}"
        a, b = pair["image_a"], pair["image_b"]
        valid = a in embeddings and b in embeddings
        row: dict[str, Any] = {"pair_id": pair_id, "split": split, "fold": "", "image_a": a, "image_b": b,
            "identity_a": pair["identity_a"], "identity_b": pair["identity_b"], "same_identity": pair["same_identity"],
            "cosine_similarity": "", "euclidean_distance": "", "valid": valid}
        if valid:
            va, vb = embeddings[a], embeddings[b]
            row["cosine_similarity"] = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
            row["euclidean_distance"] = float(np.linalg.norm(va - vb))
        else:
            for image_id in (a, b):
                if image_id not in embeddings:
                    failures.append({"image_id": image_id, "path": images.get(image_id, {}).get("image_path", ""), "pair_id": pair_id,
                                     "failure_type": "pair_invalid_missing_embedding", "message": "pair omitted; image has no valid embedding"})
        scored.append(row)
    return scored


def valid_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = [row for row in rows if row["valid"]]
    return (np.asarray([float(row["cosine_similarity"]) for row in valid]), labels_from_pairs(valid),
            np.asarray([float(row["euclidean_distance"]) for row in valid]))


def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    return operating_metrics(scores, labels, threshold)


def print_distribution(name: str, values: np.ndarray) -> None:
    stats = distribution(values)
    print(f"=== {name} ===")
    print(" ".join(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in stats.items()))


def main() -> int:
    args = parse_args(); root = project_root()
    out = args.output_dir or root / "results/calibration" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    print("[1/4] Loading pair protocol...")
    train_pairs, test_pairs, images = load_protocol(root)
    all_pairs = train_pairs + test_pairs
    # Exactly 10 genuine and 10 impostor pairs: extraction/cache smoke only.
    smoke = [p for p in train_pairs if p["same_identity"] == "true"][:10] + [p for p in train_pairs if p["same_identity"] == "false"][:10]
    failures: list[dict[str, str]] = []
    embeddings = extract_embeddings(root, smoke if args.smoke_only else all_pairs, images, out, failures, args.model)
    smoke_scored = score_pairs("smoke", smoke, images, embeddings, failures)
    smoke_scores, smoke_labels, smoke_distances = valid_arrays(smoke_scored)
    if len(smoke_scores) != 20 or not np.isfinite(smoke_scores).all() or not np.isfinite(smoke_distances).all() or not np.allclose(smoke_distances, np.sqrt(2 - 2 * smoke_scores), atol=2e-5):
        raise RuntimeError("20-pair smoke failed: decode, embeddings, labels, or normalized cosine/distance consistency invalid")
    print("Smoke PASS: 10 genuine + 10 impostor; finite 512-D normalized embeddings and cosine/distance consistency confirmed.")
    if args.smoke_only:
        return 0
    train_scored = score_pairs("dev_train", train_pairs, images, embeddings, failures)
    test_scored = score_pairs("dev_test", test_pairs, images, embeddings, failures)
    all_scored = train_scored + test_scored
    fields = ["pair_id", "split", "fold", "image_a", "image_b", "identity_a", "identity_b", "same_identity", "cosine_similarity", "euclidean_distance", "valid"]
    write_csv(out / "pair_scores.csv", all_scored, fields)
    write_csv(out / "failures.csv", failures, ["image_id", "path", "pair_id", "failure_type", "message"])
    train_scores, train_labels, train_distances = valid_arrays(train_scored)
    test_scores, test_labels, test_distances = valid_arrays(test_scored)
    print("[3/4] Computing calibration...")
    print_distribution("Genuine Cosine", train_scores[train_labels]); print_distribution("Impostor Cosine", train_scores[~train_labels])
    print_distribution("Genuine Euclidean", train_distances[train_labels]); print_distribution("Impostor Euclidean", train_distances[~train_labels])
    roc = roc_rows(train_scores, train_labels)
    write_csv(out / "roc.csv", roc, ["threshold", "far", "fpr", "tar", "tpr", "frr", "fnr", "accuracy", "tp", "fp", "tn", "fn"])
    e = eer(train_scores, train_labels)
    far01 = select_far_threshold(train_scores[~train_labels], 1e-2)
    far001 = select_far_threshold(train_scores[~train_labels], 1e-3)
    points: dict[str, dict[str, Any]] = {"eer": {**e, "support_status": "CALIBRATED", "selection_method": "discrete ROC point minimizing |FAR-FRR|"},
                                           "far_1e-2": far01, "far_1e-3": far001,
                                           "legacy_0.4": {"threshold": 0.4, "support_status": "REFERENCE_ONLY", "selection_method": "legacy fixed threshold; not selected by this calibration"}}
    print("[4/4] Held-out evaluation...")
    for point in points.values():
        threshold = float(point["threshold"])
        point["development"] = evaluate(train_scores, train_labels, threshold)
        point["held_out_test"] = evaluate(test_scores, test_labels, threshold)
    thresholds = {key: value for key, value in points.items() if key != "legacy_0.4"}
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    model_label = "FaceNet VGGFace2" if args.model == "facenet_vggface2" else "ArcFace"
    summary = {"title": f"{model_label} × LFW Threshold Calibration v1", "protocol": "official LFW Development Train -> Development Test; thresholds selected on train only", "model": cache_metadata(args.model), "metric": "cosine_similarity", "acceptance_rule": "cosine_similarity >= threshold => same identity", "selection_method": far01["selection_method"],
               "pair_counts": {"train": {"requested": len(train_scored), "valid": int(len(train_scores)), "genuine": int(train_labels.sum()), "impostor": int((~train_labels).sum()), "failed": len(train_scored)-len(train_scores)}, "test": {"requested": len(test_scored), "valid": int(len(test_scores)), "genuine": int(test_labels.sum()), "impostor": int((~test_labels).sum()), "failed": len(test_scored)-len(test_scores)}},
               "distributions": {"train_genuine_cosine": distribution(train_scores[train_labels]), "train_impostor_cosine": distribution(train_scores[~train_labels]), "train_genuine_euclidean": distribution(train_distances[train_labels]), "train_impostor_euclidean": distribution(train_distances[~train_labels])}, "operating_points": points,
               "euclidean_cosine_consistency": {"formula": "sqrt(2 - 2*cosine_similarity)", "max_abs_error": float(np.max(np.abs(train_distances - np.sqrt(2 - 2 * train_scores))) )}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    config = {"dataset": "lfw", "model": cache_metadata(args.model), "protocol": summary["protocol"], "targets": [1e-2, 1e-3], "far_support_rule": "CALIBRATED when target FAR * N_impostor >= 5; otherwise EXPLORATORY", "threshold_selection": far01["selection_method"]}
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=" * 58); print(f"{model_label} × LFW Threshold Calibration v1"); print("=" * 58)
    print(f"Model: {args.model}; protocol: Development Train -> Development Test")
    for name, point in points.items():
        dev, test = point["development"], point["held_out_test"]
        print(f"{name}: threshold={point['threshold']:.6f} status={point['support_status']} | dev FAR={dev['far']:.6f} TAR={dev['tar']:.6f} | test FAR={test['far']:.6f} TAR={test['tar']:.6f}")
    print(f"Artifacts: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
