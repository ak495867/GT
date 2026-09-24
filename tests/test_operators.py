"""
Unit tests for sublinear operators on Gabriel Horn representations.
"""

import pytest
import numpy as np
from gabriel_transform.core import GabrielTransform
from gabriel_transform.operators import (
    gabriel_point_query,
    gabriel_range_integrate,
    gabriel_inner_product,
    gabriel_fast_convolve,
)


class TestGabrielOperators:
    @pytest.fixture
    def signal_pair(self):
        t = np.linspace(0, 1, 64)
        x1 = np.sin(2 * np.pi * t)
        x2 = np.cos(2 * np.pi * t)
        gt = GabrielTransform(q=0.5)
        rep1 = gt.forward(x1)
        rep2 = gt.forward(x2)
        return x1, x2, rep1, rep2

    def test_point_query(self, signal_pair):
        x1, _, rep1, _ = signal_pair
        val, accessed = gabriel_point_query(rep1, index=16)
        assert accessed > 0
        assert val == pytest.approx(x1[16], abs=0.2)

    def test_range_integrate(self, signal_pair):
        x1, _, rep1, _ = signal_pair
        start, end = 10, 40
        exact_sum = float(np.sum(x1[start:end]))
        val, accessed = gabriel_range_integrate(rep1, start=start, end=end)
        assert accessed > 0
        assert val == pytest.approx(exact_sum, abs=1.0)

    def test_inner_product(self, signal_pair):
        x1, x2, rep1, rep2 = signal_pair
        exact_dot = float(np.dot(x1, x2))
        val, accessed = gabriel_inner_product(rep1, rep2)
        assert accessed > 0
        # sin and cos are orthogonal over period [0, 1] => dot is ~0
        assert val == pytest.approx(exact_dot, abs=1.5)

    def test_fast_convolve(self, signal_pair):
        x1, x2, rep1, rep2 = signal_pair
        exact_conv = np.convolve(x1, x2, mode="same")
        fast_conv = gabriel_fast_convolve(rep1, rep2, epsilon=0.01)
        assert len(fast_conv) == len(exact_conv)
        # Check shapes and correlation
        corr = np.corrcoef(exact_conv, fast_conv)[0, 1]
        assert corr > 0.95
