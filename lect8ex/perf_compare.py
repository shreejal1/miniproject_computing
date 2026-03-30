"""Performance comparison helpers for pure Python vs NumPy implementations."""

from __future__ import annotations

import numpy as np


def mean_slow(data: list[float]) -> float:
    """Compute mean using a Python loop."""
    total = 0.0
    count = 0
    for value in data:
        total += value
        count += 1
    if count == 0:
        raise ValueError("data must not be empty")
    return total / count


def mean_fast(data: np.ndarray) -> float:
    """Compute mean using NumPy vectorized operations."""
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    return float(np.mean(arr))


def variance_slow(data: list[float], ddof: int = 0) -> float:
    """Compute variance in pure Python."""
    n = len(data)
    if n <= ddof:
        raise ValueError("len(data) must be greater than ddof")

    mu = mean_slow(data)
    total = 0.0
    for value in data:
        diff = value - mu
        total += diff * diff
    return total / (n - ddof)


def variance_fast(data: np.ndarray, ddof: int = 0) -> float:
    """Compute variance using NumPy."""
    arr = np.asarray(data, dtype=float)
    if arr.size <= ddof:
        raise ValueError("len(data) must be greater than ddof")
    return float(np.var(arr, ddof=ddof))