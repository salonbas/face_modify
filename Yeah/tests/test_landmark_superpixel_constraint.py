from __future__ import annotations

import numpy as np

from advface.constraints.landmark_superpixel import LANDMARK_REGIONS, LandmarkSuperpixelMask, build_constraint


def test_constraint_public_api_and_binary_same_geometry(monkeypatch):
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    # A deterministic standard-layout-like set of points; the test exercises
    # polygons -> SLIC intersection -> whole-superpixel selection without a
    # model download/inference.
    points = np.zeros((68, 2), dtype=np.float32)
    points[:, 0] = np.linspace(12, 52, 68)
    points[:, 1] = 24
    for offset, indices in enumerate(LANDMARK_REGIONS.values()):
        for i, point in enumerate(indices):
            angle = 2 * np.pi * i / len(indices)
            points[point] = (20 + offset * 5 + 3 * np.cos(angle), 24 + 3 * np.sin(angle))
    constraint = LandmarkSuperpixelMask(n_segments=12, compactness=5)
    monkeypatch.setattr(LandmarkSuperpixelMask, "_landmarks", lambda _self, _image: points)
    final, initial, got = constraint.build(image)
    assert final.shape == image.shape[:2] == initial.shape
    assert got.shape == (68, 2)
    assert set(np.unique(final)) <= {0.0, 1.0}
    assert np.all(final[initial.astype(bool)] == 1.0)
    assert build_constraint(None) is None
    assert build_constraint("landmark_superpixel_mask").name == "landmark_superpixel_mask"


def test_overlay_preserves_shape(monkeypatch):
    image = np.full((20, 20, 3), 100, dtype=np.uint8)
    constraint = LandmarkSuperpixelMask(n_segments=4)
    mask = np.zeros((20, 20), dtype=np.float32); mask[2:12, 2:12] = 1
    overlay = constraint.overlay(image, mask, mask, np.zeros((68, 2), dtype=np.float32))
    assert overlay.shape == image.shape
    assert not np.array_equal(overlay, image)
