"""Large-scale Transfer Benchmark runner。"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from advface.benchmark.aggregate import SUMMARY_CSV_FIELDS, build_summary
from advface.benchmark.checkpoint import (
    append_result,
    completed_keys,
    load_jsonl,
    rewrite_jsonl,
    rewrite_raw_tables,
    write_csv,
)
from advface.benchmark.dataset import ManifestRow, load_manifest, select_images, snapshot_manifest
from advface.benchmark.grid import ExperimentUnit, generate_experiment_grid
from advface.benchmark.report import write_benchmark_report_html, write_charts
from advface.benchmark.unit import run_experiment_unit
from advface.config import (
    DEFAULT_ATTACK_SEED,
    DEFAULT_BENCHMARK_ATTACKS,
    DEFAULT_BENCHMARK_EPS,
    DEFAULT_BENCHMARK_PGD_STEPS,
    DEFAULT_DET_SIZE,
    DEFAULT_VICTIM_MODEL,
    FACE_MODEL_NAME,
    LINF_TOLERANCE,
    SIMILARITY_THRESHOLD,
    SURROGATE_THRESHOLD_STATUS,
    VICTIM_THRESHOLD_STATUS,
    ensure_project_dirs,
    insightface_providers,
    project_root,
)
from advface.experiments.output import write_config_json
from advface.experiments.transfer_output import save_transfer_images
from advface.image_io import load_bgr
from advface.models.insightface_app import InsightFaceEmbedder
from advface.paths import make_run_dir


def _git_commit(root: Path) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except Exception:
        return None
    return None


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in (
        "numpy",
        "cv2",
        "torch",
        "insightface",
        "onnxruntime",
        "onnx2torch",
        "facenet_pytorch",
        "matplotlib",
    ):
        try:
            if name == "cv2":
                import cv2

                out["opencv-python"] = getattr(cv2, "__version__", "unknown")
            else:
                mod = __import__(name)
                out[name] = getattr(mod, "__version__", "unknown")
        except Exception as e:
            out[name] = f"unavailable: {e}"
    return out


def _resolve_device(requested: Optional[str]) -> str:
    try:
        import torch

        if requested:
            key = requested.strip().lower()
            if key.startswith("cuda") and not torch.cuda.is_available():
                return "cpu"
            return key
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return requested or "cpu"


def _load_victim(name: str, device: str):
    from advface.models import load_embedder

    return load_embedder(name, device=device)


def build_run_config_payload(
    *,
    run_id: str,
    manifest_path: str,
    n_images: int,
    attacks: list[str],
    eps_list: list[float],
    pgd_steps: list[int],
    victim: str,
    device: str,
    seed: int,
    det_size: tuple[int, int],
    save_all_images: bool,
) -> dict[str, Any]:
    root = project_root()
    return {
        "pipeline": "large_scale_transfer_benchmark_v0",
        "experiment": "Large-scale Transfer Benchmark v0",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_path,
        "n_images": n_images,
        "attacks": attacks,
        "eps": eps_list,
        "pgd_steps": pgd_steps,
        "alpha_rule": "eps_px / steps (canonical PGD Full)",
        "random_start": False,
        "seed": seed,
        "device": device,
        "detector_size": list(det_size),
        "provider": insightface_providers(),
        "surrogate_model": f"insightface_{FACE_MODEL_NAME}",
        "surrogate_weight": "w600k_r50.onnx",
        "victim_model": "facenet_vggface2" if "facenet" in victim.lower() else victim,
        "victim_weight": "InceptionResnetV1 pretrained=vggface2",
        "surrogate_threshold": SIMILARITY_THRESHOLD,
        "threshold_status_surrogate": SURROGATE_THRESHOLD_STATUS,
        "victim_threshold": None,
        "threshold_status_victim": VICTIM_THRESHOLD_STATUS,
        "linf_tolerance": LINF_TOLERANCE,
        "save_all_images": save_all_images,
        "git_commit": _git_commit(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "deterministic": False,
        "deterministic_note": (
            "Full bitwise determinism is not guaranteed "
            "(ONNX Runtime, PyTorch, MTCNN, uint8 quantization)."
        ),
    }


def _save_examples(
    *,
    out_dir: Path,
    example_keys: dict[str, Any],
    rows_by_key: dict[str, dict[str, Any]],
    units_by_key: dict[str, ExperimentUnit],
    surrogate: InsightFaceEmbedder,
    victim,
    device: str,
    seed: int,
    run_id: str,
) -> dict[str, Any]:
    examples_root = out_dir / "examples"
    examples_root.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Any] = {}
    for role, key in example_keys.items():
        if not key or key not in units_by_key:
            continue
        unit = units_by_key[key]
        dest = examples_root / role
        dest.mkdir(parents=True, exist_ok=True)
        orig = load_bgr(unit.image_path)
        row, orig_bgr, adv_bgr = run_experiment_unit(
            unit,
            original_bgr=orig,
            surrogate=surrogate,
            victim=victim,
            device=device,
            seed=seed,
            run_id=run_id,
            save_image=True,
        )
        if orig_bgr is not None and adv_bgr is not None:
            save_transfer_images(dest, orig_bgr, adv_bgr)
        meta = rows_by_key.get(key) or row
        saved[role] = {
            "dir": f"examples/{role}",
            "unit_key": key,
            "meta": {
                "attack": meta.get("attack"),
                "eps": meta.get("eps"),
                "steps": meta.get("steps"),
                "surrogate_cosine": meta.get("surrogate_cosine"),
                "victim_cosine": meta.get("victim_cosine"),
                "status": meta.get("status"),
                "failure_code": meta.get("failure_code"),
            },
        }
    (examples_root / "index.json").write_text(
        json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return saved


def finalize_outputs(
    out_dir: Path,
    *,
    rows: list[dict[str, Any]],
    n_images: int,
    n_planned: int,
    config: dict[str, Any],
    example_assets: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    summary = build_summary(rows, n_images=n_images, n_planned=n_planned, config=config)
    if example_assets:
        summary["examples"] = example_assets
    charts = write_charts(out_dir, summary)
    summary["charts"] = charts
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    flat = []
    for c in summary["per_config"]:
        flat.append({k: c.get(k, "") for k in SUMMARY_CSV_FIELDS})
    write_csv(out_dir / "summary.csv", flat, SUMMARY_CSV_FIELDS)
    write_benchmark_report_html(out_dir, summary)
    return summary


def run_benchmark(
    *,
    manifest_path: str | Path,
    max_images: Optional[int],
    attacks: list[str] | None,
    eps_list: list[float] | None,
    pgd_steps: list[int] | None,
    victim: str = DEFAULT_VICTIM_MODEL,
    device: Optional[str] = None,
    run_name: str = "transfer_benchmark_v0",
    seed: int = DEFAULT_ATTACK_SEED,
    det_size: tuple[int, int] = DEFAULT_DET_SIZE,
    save_all_images: bool = False,
    retry_failures: bool = False,
    retry_invalid: bool = False,
    skip_examples: bool = False,
) -> Path:
    ensure_project_dirs()
    root = project_root()
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        cand = root / manifest_path
        if cand.is_file():
            manifest_path = cand
    images = select_images(load_manifest(manifest_path), max_images)
    attacks_n = list(attacks or DEFAULT_BENCHMARK_ATTACKS)
    eps_n = list(eps_list or DEFAULT_BENCHMARK_EPS)
    steps_n = list(pgd_steps or DEFAULT_BENCHMARK_PGD_STEPS)
    units = generate_experiment_grid(
        images, attacks=attacks_n, eps_list=eps_n, pgd_steps=steps_n
    )

    out_dir = make_run_dir(attack_type="benchmark", run_name=run_name)
    run_id = out_dir.name
    device_n = _resolve_device(device)
    jsonl_path = out_dir / "raw_results.jsonl"

    config = build_run_config_payload(
        run_id=run_id,
        manifest_path=str(manifest_path),
        n_images=len(images),
        attacks=attacks_n,
        eps_list=eps_n,
        pgd_steps=steps_n,
        victim=victim,
        device=device_n,
        seed=seed,
        det_size=det_size,
        save_all_images=save_all_images,
    )
    write_config_json(out_dir, config)
    snapshot_manifest(images, out_dir / "manifest_snapshot.csv")

    existing = load_jsonl(jsonl_path)
    skip = completed_keys(existing, include_failed=not retry_failures)
    drop_keys: set[str] = set()
    if retry_invalid:
        drop_keys |= {r["unit_key"] for r in existing if str(r.get("status")) == "invalid" and r.get("unit_key")}
    if retry_failures:
        drop_keys |= {r["unit_key"] for r in existing if str(r.get("status")) == "failed" and r.get("unit_key")}
    if drop_keys:
        existing = [r for r in existing if r.get("unit_key") not in drop_keys]
        rewrite_jsonl(jsonl_path, existing)
        skip -= drop_keys
    remaining = [u for u in units if u.key not in skip]

    print("=" * 60)
    print("Large-scale Transfer Benchmark v0")
    print(f"  images     : {len(images)}")
    print(f"  planned    : {len(units)}")
    print(f"  skip/resume: {len(skip)}")
    print(f"  remaining  : {len(remaining)}")
    print(f"  device     : {device_n}")
    print(f"  out        : {out_dir}")
    print("=" * 60)

    surrogate = InsightFaceEmbedder(det_size=det_size)
    victim_model = _load_victim(victim, device_n)

    image_cache: dict[str, Any] = {}
    all_images_dir = out_dir / "images"
    done = 0
    for unit in remaining:
        if unit.image_id not in image_cache:
            image_cache[unit.image_id] = load_bgr(unit.image_path)
        orig = image_cache[unit.image_id]
        row, orig_bgr, adv_bgr = run_experiment_unit(
            unit,
            original_bgr=orig,
            surrogate=surrogate,
            victim=victim_model,
            device=device_n,
            seed=seed,
            run_id=run_id,
            save_image=save_all_images,
        )
        append_result(jsonl_path, row)
        existing.append(row)
        if save_all_images and orig_bgr is not None and adv_bgr is not None:
            dest = all_images_dir / unit.key.replace("|", "_")
            dest.mkdir(parents=True, exist_ok=True)
            save_transfer_images(dest, orig_bgr, adv_bgr)
        done += 1
        if done % 10 == 0 or done == len(remaining):
            rewrite_raw_tables(out_dir, existing)
            print(
                f"  [{done}/{len(remaining)}] {unit.key} status={row.get('status')} "
                f"code={row.get('failure_code') or '-'}"
            )

    rewrite_raw_tables(out_dir, existing)
    rows_by_key = {r["unit_key"]: r for r in existing if r.get("unit_key")}
    units_by_key = {u.key: u for u in units}

    summary = build_summary(
        existing, n_images=len(images), n_planned=len(units), config=config
    )
    example_assets = {}
    if not skip_examples:
        print("Saving representative examples…")
        example_assets = _save_examples(
            out_dir=out_dir,
            example_keys=summary["example_keys"],
            rows_by_key=rows_by_key,
            units_by_key=units_by_key,
            surrogate=surrogate,
            victim=victim_model,
            device=device_n,
            seed=seed,
            run_id=run_id,
        )
    finalize_outputs(
        out_dir,
        rows=existing,
        n_images=len(images),
        n_planned=len(units),
        config=config,
        example_assets=example_assets,
    )
    print(f"Done. report: {out_dir / 'report.html'}")
    return out_dir
