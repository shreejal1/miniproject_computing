"""Pytest suite for stats.py."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from stats import mean, normalize, std, variance


@pytest.mark.parametrize(
    "data,expected",
    [
        ([1, 2, 3, 4], 2.5),
        ([-5, -1, 2], -4 / 3),
        ([10, 10, 10, 10], 10.0),
        ([0.25, 0.75], 0.5),
    ],
)
def test_mean_parametrized(data, expected):
    assert mean(data) == pytest.approx(expected)


def test_mean_raises_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        mean([])


def test_mean_raises_nan():
    with pytest.raises(ValueError, match="must not contain NaN"):
        mean([1.0, np.nan, 3.0])


def test_variance_population_and_sample():
    assert variance([1, 2, 3, 4], ddof=0) == pytest.approx(1.25)
    assert variance([1, 2, 3, 4], ddof=1) == pytest.approx(1.6666666666666667)


def test_variance_raises_ddof_too_large():
    with pytest.raises(ValueError, match="greater than ddof"):
        variance([1, 2], ddof=2)


def test_variance_raises_nan():
    with pytest.raises(ValueError, match="must not contain NaN"):
        variance([1.0, np.nan, 3.0])


def test_std_squared_matches_variance():
    data = [2, 4, 6, 8]
    assert std(data) ** 2 == pytest.approx(variance(data), rel=1e-12, abs=1e-12)


def test_std_raises_ddof_too_large():
    with pytest.raises(ValueError, match="greater than ddof"):
        std([5], ddof=1)


def test_normalize_mean_and_std():
    normalized = normalize([3, 6, 9, 12])
    assert mean(normalized) == pytest.approx(0.0, abs=1e-12)
    assert std(normalized) == pytest.approx(1.0, abs=1e-12)


def test_normalize_raises_identical_values():
    with pytest.raises(ValueError, match="non-zero"):
        normalize([7, 7, 7, 7])


def test_normalize_raises_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        normalize([])


def test_normalize_raises_nan():
    with pytest.raises(ValueError, match="must not contain NaN"):
        normalize([1.0, np.nan, 3.0])


def test_tolerance_mean_exact_target():
    assert_allclose(mean([1, 2, 3, 4, 5]), 3.0, rtol=1e-12)


def test_tolerance_bessel_correction_large_sample():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 2, 10000)
    assert_allclose(variance(data, ddof=1), 4.0, rtol=0.05)


def test_tolerance_normalize_moments():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = normalize(data)
    assert abs(mean(normalized)) < 1e-12
    assert abs(std(normalized) - 1.0) < 1e-12