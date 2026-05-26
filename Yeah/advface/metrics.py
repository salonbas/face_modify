import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0:
        raise ValueError("向量範數為 0，無法計算 Cosine Similarity。")
    return float(np.dot(vec1, vec2) / denom)


def cosine_similarity_or_nan(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0:
        return float("nan")
    return float(np.dot(vec1, vec2) / denom)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.linalg.norm(vec1 - vec2))
