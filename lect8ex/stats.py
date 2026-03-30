"""Basic statistical helpers with defensive validation.

This module provides small, dependency-light wrappers around common
descriptive statistics used throughout the exercises: mean, variance,
standard deviation, and z-score normalization.
"""

from __future__ import annotations

import numpy as np


def mean(data: np.ndarray | list[float]) -> float:
    """Compute the arithmetic mean of a 1D numeric dataset.

    Parameters
    ----------
    data : array-like of float
        Input values. Any 1D array-like object accepted by
        :func:`numpy.asarray` is supported.

    Returns
    -------
    float
        Arithmetic mean of the input values.

    Raises
    ------
    ValueError
        If ``data`` is empty.
    ValueError
        If ``data`` contains at least one ``NaN`` value.

    Examples
    --------
    >>> mean([1, 2, 3, 4])
    2.5
    >>> mean(np.array([-2.0, 0.0, 2.0]))
    0.0
    >>> mean([])
    Traceback (most recent call last):
    ...
    ValueError: data must not be empty
    """
    arr = np.asarray(data, dtype=float)

    if arr.size == 0:
        raise ValueError("data must not be empty")
    if np.isnan(arr).any():
        raise ValueError("data must not contain NaN")

    return float(np.mean(arr))


def variance(data: np.ndarray | list[float], ddof: int = 0) -> float:
    """Compute variance with configurable degrees-of-freedom correction.

    Parameters
    ----------
    data : array-like of float
        Input values.
    ddof : int, default=0
        Delta degrees of freedom. The divisor used in the calculation is
        ``N - ddof``, where ``N`` is the number of elements in ``data``.

    Returns
    -------
    float
        Variance of the input values under the selected ``ddof``.

    Raises
    ------
    ValueError
        If ``len(data) <= ddof``.
    ValueError
        If ``data`` contains at least one ``NaN`` value.

    Examples
    --------
    >>> variance([1, 2, 3, 4], ddof=0)
    1.25
    >>> variance([1, 2, 3, 4], ddof=1)
    1.6666666666666667
    >>> variance([10, 10], ddof=2)
    Traceback (most recent call last):
    ...
    ValueError: len(data) must be greater than ddof
    """
    arr = np.asarray(data, dtype=float)

    if np.isnan(arr).any():
        raise ValueError("data must not contain NaN")
    if arr.size <= ddof:
        raise ValueError("len(data) must be greater than ddof")

    return float(np.var(arr, ddof=ddof))


def std(data: np.ndarray | list[float], ddof: int = 0) -> float:
    """Compute standard deviation with configurable ``ddof``.

    Parameters
    ----------
    data : array-like of float
        Input values.
    ddof : int, default=0
        Delta degrees of freedom forwarded to :func:`variance`.

    Returns
    -------
    float
        Standard deviation of the input values.

    Raises
    ------
    ValueError
        If ``len(data) <= ddof``.
    ValueError
        If ``data`` contains at least one ``NaN`` value.

    Examples
    --------
    >>> std([1, 2, 3, 4])
    1.118033988749895
    >>> std([2, 2, 2])
    0.0
    >>> std([5], ddof=1)
    Traceback (most recent call last):
    ...
    ValueError: len(data) must be greater than ddof
    """
    return float(np.sqrt(variance(data, ddof=ddof)))


def normalize(data: np.ndarray | list[float]) -> np.ndarray:
    """Return a z-score normalized copy of the input values.

    Parameters
    ----------
    data : array-like of float
        Input values.

    Returns
    -------
    numpy.ndarray
        Normalized values computed as ``(x - mean(data)) / std(data)``.

    Raises
    ------
    ValueError
        If ``data`` is empty.
    ValueError
        If ``data`` contains at least one ``NaN`` value.
    ValueError
        If standard deviation is zero (all values identical).

    Examples
    --------
    >>> normalize([1, 2, 3]).tolist()
    [-1.224744871391589, 0.0, 1.224744871391589]
    >>> normalize([10, 20]).tolist()
    [-1.0, 1.0]
    >>> normalize([7, 7, 7])
    Traceback (most recent call last):
    ...
    ValueError: standard deviation must be non-zero
    """
    arr = np.asarray(data, dtype=float)
    mu = mean(arr)
    sigma = std(arr)

    if sigma == 0.0:
        raise ValueError("standard deviation must be non-zero")

    return (arr - mu) / sigma
