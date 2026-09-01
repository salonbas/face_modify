"""Benchmark 失敗分類。"""
from __future__ import annotations

from typing import Optional

NO_FACE_DETECTED = "NO_FACE_DETECTED"
MULTIPLE_FACE_AMBIGUOUS = "MULTIPLE_FACE_AMBIGUOUS"
ATTACK_EXCEPTION = "ATTACK_EXCEPTION"
VICTIM_EXCEPTION = "VICTIM_EXCEPTION"
OOM = "OOM"
INVALID_LINF = "INVALID_LINF"
INVALID_EMBEDDING = "INVALID_EMBEDDING"
OTHER = "OTHER"

FAILURE_CODES = (
    NO_FACE_DETECTED,
    MULTIPLE_FACE_AMBIGUOUS,
    ATTACK_EXCEPTION,
    VICTIM_EXCEPTION,
    OOM,
    INVALID_LINF,
    INVALID_EMBEDDING,
    OTHER,
)


class BenchmarkFailure(Exception):
    def __init__(self, code: str, message: str = "") -> None:
        self.code = code if code in FAILURE_CODES else OTHER
        self.message = message or code
        super().__init__(self.message)


def classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, BenchmarkFailure):
        return exc.code
    if "out of memory" in text or "cuda" in text and "memory" in text:
        return OOM
    if "沒有偵測到人臉" in str(exc) or "未偵測到人臉" in str(exc) or "no face" in text:
        return NO_FACE_DETECTED
    if "multiple" in text and "face" in text:
        return MULTIPLE_FACE_AMBIGUOUS
    if "embedding" in text:
        return INVALID_EMBEDDING
    return OTHER


def classify_attack_exception(exc: BaseException) -> str:
    code = classify_exception(exc)
    if code in (NO_FACE_DETECTED, MULTIPLE_FACE_AMBIGUOUS, OOM, INVALID_EMBEDDING):
        return code
    return ATTACK_EXCEPTION
