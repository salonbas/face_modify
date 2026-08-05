"""tests for similarity metrics."""
from __future__ import annotations

import numpy as np
import pytest

from advface.evaluation.similarity import (
    cosine_similarity,
    cosine_similarity_or_nan,
    euclidean_distance,
)


def test_identical_vectors_cosine_one():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_opposite_vectors_cosine_minus_one():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_similarity(v, -v) == pytest.approx(-1.0)


def test_euclidean_distance():
    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    assert euclidean_distance(a, b) == pytest.approx(5.0)


def test_zero_vector_raises():
    z = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError):
        cosine_similarity(z, z)


def test_zero_vector_or_nan():
    z = np.zeros(3, dtype=np.float32)
    assert np.isnan(cosine_similarity_or_nan(z, z))
