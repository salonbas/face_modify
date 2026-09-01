"""Benchmark dataset manifest contract。"""
from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from advface.config import project_root

MANIFEST_FIELDS = ("image_id", "image_path", "identity_id", "source")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    image_path: str
    identity_id: str
    source: str

    def resolved_path(self, root: Path | None = None) -> Path:
        p = Path(self.image_path)
        if p.is_file():
            return p.resolve()
        base = root or project_root()
        cand = (base / self.image_path).resolve()
        return cand


def _cell(row: dict, key: str, default: str = "") -> str:
    v = row.get(key, default)
    if v is None:
        return default
    return str(v).strip()


def load_manifest(path: str | Path, *, root: Path | None = None) -> list[ManifestRow]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 manifest：{path}")
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("manifest.csv 沒有表頭")
        missing = [c for c in ("image_id", "image_path") if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"manifest.csv 缺少欄位：{missing}")
        for i, raw in enumerate(reader, start=2):
            image_id = _cell(raw, "image_id")
            image_path = _cell(raw, "image_path")
            if not image_id or not image_path:
                raise ValueError(f"manifest 第 {i} 列缺少 image_id 或 image_path")
            if image_id in seen:
                raise ValueError(f"manifest 重複 image_id：{image_id}")
            seen.add(image_id)
            rows.append(
                ManifestRow(
                    image_id=image_id,
                    image_path=image_path,
                    identity_id=_cell(raw, "identity_id"),
                    source=_cell(raw, "source"),
                )
            )
    if root is None:
        root = project_root()
    for r in rows:
        p = r.resolved_path(root)
        if not p.is_file():
            raise FileNotFoundError(f"manifest 指向的圖片不存在：{r.image_id} -> {r.image_path}")
    return rows


def select_images(rows: Iterable[ManifestRow], max_images: Optional[int]) -> list[ManifestRow]:
    out = list(rows)
    if max_images is not None:
        if max_images < 1:
            raise ValueError("--max-images 必須 >= 1")
        out = out[: int(max_images)]
    return out


def write_manifest(path: str | Path, rows: Iterable[ManifestRow]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(MANIFEST_FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "image_id": r.image_id,
                    "image_path": r.image_path,
                    "identity_id": r.identity_id,
                    "source": r.source,
                }
            )
    return path


def snapshot_manifest(rows: Iterable[ManifestRow], dest: Path) -> Path:
    return write_manifest(dest, rows)


def copy_image_into_benchmark(src: Path, dest_dir: Path, image_id: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower() if src.suffix.lower() in IMAGE_EXTS else ".png"
    dest = dest_dir / f"{image_id}{ext}"
    if not dest.is_file() or dest.stat().st_size == 0:
        shutil.copy2(src, dest)
    return dest
