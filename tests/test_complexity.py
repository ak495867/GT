"""
Empirical verification of theoretical complexity targets:
- Depth K(eps) = O(log(1/eps))
- Point query T(n) = O(log n)
- Combined target T(n, eps) = O(log n + log(1/eps))
- Double-logarithmic scale jumping O(log log n)
- Recurrence solver T(n) = T(n/r) + O(1) => O(log_r n)
"""

import pytest
import numpy as np
from gabriel_transform.core import GeometricHornProfile
from gabriel_transform.tree import AdaptiveGabrielHornTree
from gabriel_transform.theory import (
    GabrielComplexityModel,
    fit_log_scaling,
    solve_recurrence_complexity,
)


class TestComplexityTargets:
    def test_depth_log_epsilon_scaling(self):
        """Verify K(eps) is proportional to log(1/eps) with R^2 > 0.99."""
        q = 0.5
        c = 1.0
        prof = GeometricHornProfile(q=q, c=c)

        # Range of epsilon from 1e-1 down to 1e-8
        epsilons = np.logspace(-1, -8, 8)
        depths = [prof.required_depth(eps) for eps in epsilons]

        # Fit depth against log(1/eps)
        inv_eps = 1.0 / epsilons
        slope, intercept, r_squared = fit_log_scaling(inv_eps, np.array(depths))

        # Theoretical slope: 1 / log(1/q) = 1 / log(2) ~ 1.442695
        expected_slope = 1.0 / np.log(1.0 / q)
        assert r_squared > 0.99, f"Expected linear fit against log(1/eps), got R^2 = {r_squared}"
        assert slope == pytest.approx(expected_slope, rel=0.1)

    def test_tree_point_query_log_n_scaling(self):
        """Verify point query steps scale as O(log n) across sizes n = 32, 64, ..., 1024."""
        ns = [32, 64, 128, 256, 512, 1024]
        max_steps = []

        for n in ns:
            arr = np.sin(np.linspace(0, 4 * np.pi, n))
            tree = AdaptiveGabrielHornTree.build(arr, q=0.5)
            # Evaluate at arbitrary index
            _, steps = tree.point_query(n // 3, epsilon=0.0)
            max_steps.append(steps)

        # Check that steps are strictly bounded by ceil(log2(n)) + 1
        for n, s in zip(ns, max_steps):
            theoretical_max = int(np.ceil(np.log2(n))) + 1
            assert s <= theoretical_max

        # Fit steps against log(n)
        slope, _, r_squared = fit_log_scaling(np.array(ns), np.array(max_steps))
        assert r_squared > 0.98

    def test_scale_jump_log_log_scaling(self):
        """Verify scale jumping takes <= O(log log n) steps."""
        ns = [64, 256, 1024, 4096]
        jump_steps = []

        for n in ns:
            arr = np.zeros(n)
            # Add single high-frequency burst at fine level
            arr[n // 2] = 10.0
            tree = AdaptiveGabrielHornTree.build(arr, q=0.5)
            _, steps = tree.jump_to_scale(epsilon=0.5)
            jump_steps.append(steps)

        # For n = 4096, log2(n) = 12, log2(log2(n)) ~ 3.5. Steps should be <= 8.
        for n, s in zip(ns, jump_steps):
            log_log_bound = int(np.ceil(np.log2(np.log2(n)))) + 4
            assert s <= log_log_bound

    def test_recurrence_solver(self):
        rec_res = solve_recurrence_complexity(a=1, r=2, work_per_level=1)
        assert "O(log n)" in rec_res
        assert "T(n/2)" in rec_res
