"""Property-based tests for stats.py using Hypothesis."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stats import mean, normalize, std, variance


def _finite_float_strategy(min_value: float = -1e6, max_value: float = 1e6):
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )


@settings(deadline=None, max_examples=200)
@given(
    data=st.lists(_finite_float_strategy(-1e3, 1e3), min_size=2, max_size=50),
    c=_finite_float_strategy(-1e3, 1e3),
)
def test_shift_invariance_of_mean(data, c):
    shifted = [x + c for x in data]
    assert mean(shifted) == pytest.approx(mean(data) + c, rel=1e-10, abs=1e-10)


@settings(deadline=None, max_examples=200)
@given(
    data=st.lists(_finite_float_strategy(-1e3, 1e3), min_size=2, max_size=50),
    c=_finite_float_strategy(-20, 20),
)
def test_scale_rule_of_variance(data, c):
    scaled = [c * x for x in data]
    assert variance(scaled) == pytest.approx((c**2) * variance(data), rel=1e-8, abs=1e-8)


@settings(deadline=None, max_examples=200)
@given(
    data=st.lists(_finite_float_strategy(-1e3, 1e3), min_size=2, max_size=50).filter(
        lambda xs: not np.allclose(np.asarray(xs), xs[0])
    )
)
def test_normalization_moments(data):
    normalized = normalize(data)
    assert mean(normalized) == pytest.approx(0.0, abs=1e-10)
    assert std(normalized) == pytest.approx(1.0, abs=1e-10)