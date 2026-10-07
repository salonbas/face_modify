"""68-landmark + SLIC spatial mask used by the Invisible Mask ablation.

The mask construction is deliberately separate from attacks: a constraint only
turns an input BGR image into an HxW float mask.  PGD, MI-FGSM, and future
attacks can apply it to their gradient without creating attack-specific forks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from skimage.segmentation import slic

from advface.config import DEFAULT_DET_SIZE, insightface_providers
from advface.models.insightface_app import insightface_model_root, pick_best_face

# Standard 68-point layout, zero based.  The paper-level feature selection is
# eyebrows, eyes, nose and mouth; we intentionally do not add face contour,
# saliency, or a JND heuristic.
LANDMARK_REGIONS: dict[str, tuple[int, ...]] = {
    "left_eyebrow": tuple(range(17, 22)),
    "right_eyebrow": tuple(range(22, 27)),
    "nose": tuple(range(27, 36)),
    "left_eye": tuple(range(36, 42)),
    "right_eye": tuple(range(42, 48)),
    "outer_mouth": tuple(range(48, 60)),
    "inner_mouth": tuple(range(60, 68)),
}


@dataclass(frozen=True)
class LandmarkSuperpixelMask:
    """Build a binary full-image mask from 68 landmarks and SLIC labels.

    ``n_segments`` and ``compactness`` are recorded implementation choices:
    the available paper description does not provide a reproducible numeric
    SLIC setting.  They are parameters, rather than hidden constants, so an
    ablation can state them exactly.
    """

    n_segments: int = 100
    compactness: float = 10.0
    sigma: float = 0.0
    name: str = "landmark_superpixel_mask"

    def _landmarks(self, image_bgr: np.ndarray) -> np.ndarray:
        # buffalo_l ships 1k3d68 locally.  FaceAnalysis calls the landmark
        # model after detection and attaches landmark_3d_68 to the best face.
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_l", root=str(insightface_model_root()), providers=insightface_providers(),
            allowed_modules=["detection", "landmark_3d_68"],
        )
        app.prepare(ctx_id=0, det_size=DEFAULT_DET_SIZE)
        face = pick_best_face(app.get(image_bgr), "landmark mask input")
        points = np.asarray(face.landmark_3d_68, dtype=np.float32)
        if points.shape != (68, 3):
            raise RuntimeError(f"expected InsightFace 68x3 landmarks, got {points.shape}")
        return points[:, :2]

    def build(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(final_mask, initial_landmark_mask, landmarks_xy)``.

        A superpixel is selected iff it has at least one pixel in the initial
        polygon mask.  The result is HxW float32 with values exactly 0 or 1.
        """
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("landmark superpixel mask expects HxWx3 BGR image")
        landmarks = self._landmarks(image_bgr)
        height, width = image_bgr.shape[:2]
        initial = np.zeros((height, width), dtype=np.uint8)
        for indices in LANDMARK_REGIONS.values():
            polygon = np.rint(landmarks[list(indices)]).astype(np.int32)
            cv2.fillPoly(initial, [polygon], 1)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        labels = slic(
            rgb, n_segments=self.n_segments, compactness=self.compactness,
            sigma=self.sigma, start_label=0, channel_axis=-1,
        )
        chosen = np.unique(labels[initial.astype(bool)])
        final = np.isin(labels, chosen).astype(np.float32)
        return final, initial.astype(np.float32), landmarks

    def segmentation(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return deterministic SLIC labels for an inspection visualization."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return slic(
            rgb, n_segments=self.n_segments, compactness=self.compactness,
            sigma=self.sigma, start_label=0, channel_axis=-1,
        )

    @staticmethod
    def colorize_segments(labels: np.ndarray) -> np.ndarray:
        """Make label boundaries visible without implying a semantic region."""
        labels = np.asarray(labels, dtype=np.int32)
        palette = np.stack(((labels * 37) % 255, (labels * 73) % 255, (labels * 109) % 255), axis=-1).astype(np.uint8)
        boundaries = np.zeros(labels.shape, dtype=bool)
        boundaries[1:, :] |= labels[1:, :] != labels[:-1, :]
        boundaries[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        palette[boundaries] = (255, 255, 255)
        return palette

    @staticmethod
    def landmark_region_overlay(image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Inspection image with every landmark index and the assumed regions."""
        output = image_bgr.copy()
        colors = {
            "left_eyebrow": (0, 255, 255), "right_eyebrow": (0, 200, 255),
            "nose": (255, 0, 255), "left_eye": (255, 255, 0),
            "right_eye": (255, 180, 0), "outer_mouth": (0, 0, 255),
            "inner_mouth": (0, 100, 255),
        }
        for name, indices in LANDMARK_REGIONS.items():
            color = colors[name]
            for index in indices:
                x, y = np.rint(landmarks[index]).astype(int)
                if 0 <= x < output.shape[1] and 0 <= y < output.shape[0]:
                    cv2.circle(output, (x, y), 2, color, -1, lineType=cv2.LINE_AA)
                    cv2.putText(output, str(index), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
        return output

    def overlay(self, image_bgr: np.ndarray, final_mask: np.ndarray, initial_mask: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Visual inspection image: final green region, initial red, cyan points."""
        output = image_bgr.copy()
        green = np.zeros_like(output); green[:, :, 1] = 255
        red = np.zeros_like(output); red[:, :, 2] = 255
        # Blend explicitly to retain unselected pixels byte-for-byte.
        selected = final_mask.astype(bool)
        output[selected] = (0.72 * output[selected] + 0.28 * green[selected]).astype(np.uint8)
        seeded = initial_mask.astype(bool)
        output[seeded] = (0.55 * output[seeded] + 0.45 * red[seeded]).astype(np.uint8)
        for x, y in np.rint(landmarks).astype(int):
            if 0 <= x < output.shape[1] and 0 <= y < output.shape[0]:
                cv2.circle(output, (x, y), 1, (255, 255, 0), -1, lineType=cv2.LINE_AA)
        return output


def build_constraint(name: str | None, **kwargs: Any) -> LandmarkSuperpixelMask | None:
    """Resolve the public ``constraint=None|landmark_superpixel_mask`` API."""
    if name is None or str(name).lower() in {"", "none"}:
        return None
    if str(name).lower() != "landmark_superpixel_mask":
        raise ValueError(f"unknown constraint: {name}")
    allowed = {"n_segments", "compactness", "sigma"}
    unexpected = set(kwargs) - allowed
    if unexpected:
        raise ValueError(f"unknown landmark_superpixel_mask options: {sorted(unexpected)}")
    return LandmarkSuperpixelMask(**kwargs)
