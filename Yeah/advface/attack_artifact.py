"""Fail-closed, portable storage for serialized attack outputs.

Evaluation summaries are deliberately outside this record: changing a victim
threshold must never invalidate or regenerate an otherwise identical attack.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

SCHEMA = 1
REQUIRED = ("identity_id", "reference_sha256", "probe_sha256", "method",
            "mask", "arcface_checkpoint_sha256", "attack_config_sha256",
            "preprocessing_version", "python_version", "torch_version",
            "dependency_versions", "git_commit")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def validate(png: Path, metadata: Path, expected: dict, probe: np.ndarray | None = None) -> dict:
    if not png.is_file() or not metadata.is_file():
        raise ValueError("attack PNG or metadata missing")
    try:
        record = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("attack metadata unreadable") from exc
    if record.get("schema") != SCHEMA or record.get("status") != "complete":
        raise ValueError("attack metadata incomplete")
    if any(not record.get(key) for key in REQUIRED):
        raise ValueError("attack provenance incomplete")
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("attack provenance mismatch")
    if sha256(png) != record.get("output_png_sha256"):
        raise ValueError("attack PNG checksum mismatch")
    image = cv2.imread(str(png), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("attack PNG cannot be decoded")
    if probe is not None:
        if image.shape != probe.shape:
            raise ValueError("attack PNG shape mismatch")
        linf = float(np.abs(image.astype(np.int16) - probe.astype(np.int16)).max() / 255)
        if abs(linf - float(record["serialized_linf"])) > 1e-12:
            raise ValueError("serialized L∞ mismatch")
    return record


def save(png: Path, metadata: Path, image: np.ndarray, probe: np.ndarray,
         provenance: dict) -> dict:
    if png.exists() or metadata.exists():
        raise FileExistsError("refusing to overwrite an attack artifact")
    if any(not provenance.get(key) for key in REQUIRED):
        raise ValueError("attack provenance incomplete")
    if image.shape != probe.shape or image.dtype != np.uint8 or probe.dtype != np.uint8:
        raise ValueError("attack/probe must be same-shape uint8 images")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed")
    payload = encoded.tobytes()
    record = dict(provenance, schema=SCHEMA, status="complete",
                  output_png_sha256=hashlib.sha256(payload).hexdigest(),
                  serialized_linf=float(np.abs(image.astype(np.int16) - probe.astype(np.int16)).max() / 255))
    # Metadata is the completion marker. A crash between writes leaves an
    # orphan PNG that validation rejects and the caller must quarantine.
    _atomic_bytes(png, payload)
    _atomic_bytes(metadata, (json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
    validate(png, metadata, provenance, probe)
    return record
