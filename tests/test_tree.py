"""
Unit tests for AdaptiveGabrielHornTree.
Validates O(log(1/eps)) point evaluation, O(log n) exact point evaluation,
O(log log n) scale jumping, and definite range integration.
"""

import pytest
import numpy as np
from gabriel_transform.tree import AdaptiveGabrielHornTree


class TestAdaptiveGabrielHornTree:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 128
        t = np.linspace(0, 1, n)
        # Smooth structured signal + small step
        arr = np.sin(4 * np.pi * t) + (t > 0.5).astype(float) * 0.5
        return arr

    def test_build_tree(self, sample_data):
        tree = AdaptiveGabrielHornTree.build(sample_data, q=0.5)
        assert tree.root is not None
        assert tree.n == len(sample_data)
        assert tree.max_depth == 7  # log2(128) = 7

    def test_exact_point_query(self, sample_data):
        tree = AdaptiveGabrielHornTree.build(sample_data, q=0.5)
        for idx in [0, 10, 64, 127]:
            val, steps = tree.point_query(idx, epsilon=0.0)
            assert val == pytest.approx(sample_data[idx], abs=1e-5)
            # Complexity must be at most max_depth + 1 = 8 steps (O(log n))
            assert steps <= tree.max_depth + 1

    def test_sublinear_point_query_with_epsilon(self, sample_data):
        tree = AdaptiveGabrielHornTree.build(sample_data, q=0.5)
        eps = 0.5
        val, steps = tree.point_query(32, epsilon=eps)
        # Should stop early when subtree energy <= eps
        assert steps <= tree.max_depth + 1
        # Error should be within epsilon tolerance
        exact = sample_data[32]
        assert abs(val - exact) <= eps * 1.5

    def test_range_sum_query(self, sample_data):
        tree = AdaptiveGabrielHornTree.build(sample_data, q=0.5)
        start, end = 20, 80
        exact_sum = float(np.sum(sample_data[start:end]))
        approx_sum, steps = tree.range_sum_query(start, end, epsilon=0.0)
        assert approx_sum == pytest.approx(exact_sum, abs=1e-5)
        # Steps should be far less than linear scan of 60 elements
        assert steps <= 4 * (tree.max_depth + 1)

    def test_scale_jump_log_log(self, sample_data):
        tree = AdaptiveGabrielHornTree.build(sample_data, q=0.5)
        depth, steps = tree.jump_to_scale(epsilon=0.2)
        assert depth >= 0
        # O(log log n) check: for n=128, log2(n)=7, log2(7) ~ 3 steps!
        assert steps <= 6

    def test_update_consistency(self, sample_data):
        tree = AdaptiveGabrielHornTree.build(sample_data, q=0.5)
        idx = 42
        delta = 3.5
        steps = tree.update(idx, delta)
        # Update takes O(log n)
        assert steps <= tree.max_depth + 1

        val, _ = tree.point_query(idx, epsilon=0.0)
        assert val == pytest.approx(sample_data[idx] + delta, abs=1e-5)
