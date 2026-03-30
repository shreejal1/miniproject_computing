"""Unittest suite for stats.py."""

import unittest

import numpy as np

from stats import mean, normalize, std, variance


class TestMean(unittest.TestCase):
    def test_basic_mean(self) -> None:
        self.assertEqual(mean([1, 2, 3, 4]), 2.5)

    def test_identical_values(self) -> None:
        self.assertEqual(mean([5, 5, 5, 5]), 5.0)

    def test_negative_numbers(self) -> None:
        self.assertEqual(mean([-5, -1, 2]), -4 / 3)

    def test_raises_on_empty(self) -> None:
        with self.assertRaises(ValueError):
            mean([])

    def test_raises_on_nan(self) -> None:
        with self.assertRaises(ValueError):
            mean([1.0, np.nan, 3.0])


class TestVariance(unittest.TestCase):
    def test_population_variance(self) -> None:
        self.assertAlmostEqual(variance([1, 2, 3, 4], ddof=0), 1.25)

    def test_sample_variance(self) -> None:
        self.assertAlmostEqual(variance([1, 2, 3, 4], ddof=1), 1.6666666666666667)

    def test_error_when_ddof_gte_len(self) -> None:
        with self.assertRaises(ValueError):
            variance([1, 2], ddof=2)

    def test_raises_on_nan(self) -> None:
        with self.assertRaises(ValueError):
            variance([1.0, np.nan, 3.0], ddof=0)


class TestStd(unittest.TestCase):
    def test_square_relationship(self) -> None:
        data = [2, 4, 6, 8]
        self.assertAlmostEqual(std(data) ** 2, variance(data), places=12)


class TestNormalize(unittest.TestCase):
    def test_mean_zero_std_one_after_normalization(self) -> None:
        data = [3, 6, 9, 12]
        normalized = normalize(data)
        self.assertAlmostEqual(mean(normalized), 0.0, places=12)
        self.assertAlmostEqual(std(normalized), 1.0, places=12)

    def test_raises_when_identical(self) -> None:
        with self.assertRaises(ValueError):
            normalize([7, 7, 7, 7])

    def test_raises_on_empty(self) -> None:
        with self.assertRaises(ValueError):
            normalize([])

    def test_raises_on_nan(self) -> None:
        with self.assertRaises(ValueError):
            normalize([1.0, np.nan, 3.0])


if __name__ == "__main__":
    unittest.main()