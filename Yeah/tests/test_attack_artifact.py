import json

import numpy as np
import pytest

from advface.attack_artifact import config_hash, save, validate


def provenance():
    return dict(identity_id="Example", reference_sha256="a" * 64,
                probe_sha256="b" * 64, method="pgd_full", mask="none",
                arcface_checkpoint_sha256="c" * 64,
                attack_config_sha256=config_hash({"epsilon": .04, "steps": 200}),
                preprocessing_version="v1", python_version="3.12",
                torch_version="2.2.2", dependency_versions={"cv2": "test"},
                git_commit="d" * 40)


def test_resume_integrity_and_threshold_independence(tmp_path):
    png, meta = tmp_path / "adv.png", tmp_path / "adv.json"
    clean = np.zeros((8, 8, 3), dtype=np.uint8)
    adv = clean.copy(); adv[0, 0, 0] = 11
    expected = provenance()
    save(png, meta, adv, clean, expected)
    assert validate(png, meta, expected, clean)["serialized_linf"] == 11 / 255
    # Threshold is evaluation state, absent from generation provenance.
    assert validate(png, meta, expected, clean)["status"] == "complete"
    changed = dict(expected, attack_config_sha256=config_hash({"epsilon": .03}))
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate(png, meta, changed, clean)
    png.write_bytes(png.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate(png, meta, expected, clean)


def test_incomplete_and_interrupted_are_rejected(tmp_path):
    png, meta = tmp_path / "adv.png", tmp_path / "adv.json"
    clean = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="missing"):
        validate(png, meta, provenance(), clean)
    save(png, meta, clean, clean, provenance())
    row = json.loads(meta.read_text()); del row["probe_sha256"]
    meta.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="incomplete"):
        validate(png, meta, provenance(), clean)
    meta.unlink()
    with pytest.raises(ValueError, match="missing"):
        validate(png, meta, provenance(), clean)
