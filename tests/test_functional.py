"""
Unit tests for continuous and functional Gabriel Transform.
"""

import pytest
import numpy as np
from gabriel_transform.functional import FunctionalGabrielTransform, ContinuousHornBasis


class TestFunctionalGabrielTransform:
    def test_continuous_horn_basis_spline(self):
        basis = ContinuousHornBasis(basis_type="spline")
        k = 2
        m = 1
        scale = 2 ** k
        t_center = m / scale
        val = basis.evaluate(k, m, t_center)
        assert val == pytest.approx(1.0)

        # Outside support: |t - center| >= half_width
        val_outside = basis.evaluate(k, m, 0.99)
        assert val_outside == 0.0

    def test_functional_fit_and_evaluate(self):
        # f(t) = sin(2*pi*t)
        func = lambda t: np.sin(2 * np.pi * t)
        fgt = FunctionalGabrielTransform(func=func, domain=(0.0, 1.0), q=0.5)
        fgt.fit(max_levels=5, samples=512)

        # Evaluate at test points
        for t_test in [0.25, 0.5, 0.75]:
            exact = func(t_test)
            approx = fgt.evaluate(t_test)
            assert approx == pytest.approx(exact, abs=0.15)

    def test_functional_definite_integral(self):
        # f(t) = cos(2*pi*t), integral from 0 to 0.5 is 0
        func = lambda t: np.cos(2 * np.pi * t)
        fgt = FunctionalGabrielTransform(func=func, domain=(0.0, 1.0), q=0.5)
        fgt.fit(max_levels=5, samples=256)

        quad_res = fgt.definite_integral(0.0, 0.5)
        assert quad_res == pytest.approx(0.0, abs=0.1)
