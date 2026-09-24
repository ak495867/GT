"""
Unit tests for Spectral Gabriel Transform (Chebyshev and Legendre Horns).
"""

import pytest
import numpy as np
from gabriel_transform.spectral import (
    SpectralChebyshevHornTransform,
    SpectralLegendreHornTransform,
)


class TestSpectralGabrielTransform:
    def test_chebyshev_fit_and_evaluate(self):
        # Analytic function: f(x) = exp(x) on [-1, 1]
        func = lambda x: np.exp(x)
        sgt = SpectralChebyshevHornTransform(func=func, domain=(-1.0, 1.0), q=0.5)
        sgt.fit(max_degree=16, num_sample_points=64)

        for x_val in [-0.75, 0.0, 0.5, 0.9]:
            exact = func(x_val)
            approx, deg = sgt.evaluate(x_val, epsilon=1e-5)
            assert approx == pytest.approx(exact, abs=1e-4)
            assert deg <= 17

    def test_chebyshev_definite_integral(self):
        # f(x) = x^2, integral from -1 to 1 is 2/3 = 0.666666...
        func = lambda x: x ** 2
        sgt = SpectralChebyshevHornTransform(func=func, domain=(-1.0, 1.0), q=0.5)
        sgt.fit(max_degree=8, num_sample_points=32)

        integral_val = sgt.definite_integral(-1.0, 1.0)
        assert integral_val == pytest.approx(2.0 / 3.0, abs=1e-5)

    def test_legendre_fit_and_evaluate(self):
        func = lambda x: np.cos(np.pi * x)
        leg_gt = SpectralLegendreHornTransform(func=func, domain=(-1.0, 1.0), q=0.5)
        leg_gt.fit(max_degree=12, num_sample_points=32)

        val, deg = leg_gt.evaluate(0.0)
        assert val == pytest.approx(1.0, abs=0.05)
