"""Small equation-solver utilities.

This module intentionally started as a low-quality example in the exercise,
then was refactored to satisfy static-analysis checks.
"""

from __future__ import annotations

import math


def solve_linear(coefficient: float, constant: float) -> float:
    """Solve ``coefficient * x + constant = 0`` for ``x``.

    Parameters
    ----------
    coefficient : float
        Non-zero linear coefficient.
    constant : float
        Constant term.

    Returns
    -------
    float
        The unique solution.

    Raises
    ------
    ValueError
        If ``coefficient`` is zero.
    """
    if coefficient == 0:
        raise ValueError("coefficient must be non-zero")
    return -constant / coefficient


def solve_quadratic(a: float, b: float, c: float) -> tuple[float, float]:
    """Solve ``a*x^2 + b*x + c = 0`` over the real numbers.

    Parameters
    ----------
    a : float
        Non-zero quadratic coefficient.
    b : float
        Linear coefficient.
    c : float
        Constant term.

    Returns
    -------
    tuple[float, float]
        Two real roots sorted in ascending order.

    Raises
    ------
    ValueError
        If ``a`` is zero.
    ValueError
        If the discriminant is negative (no real roots).
    """
    if a == 0:
        raise ValueError("a must be non-zero")

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        raise ValueError("equation has no real roots")

    sqrt_disc = math.sqrt(discriminant)
    root1 = (-b - sqrt_disc) / (2.0 * a)
    root2 = (-b + sqrt_disc) / (2.0 * a)
    return (root1, root2) if root1 <= root2 else (root2, root1)
