"""
Unit tests for EytzingerGabrielHornTree (Flat contiguous memory layout).
"""

import pytest
import numpy as np
from gabriel_transform.eytzinger import EytzingerGabrielHornTree
from gabriel_transform.tree import AdaptiveGabrielHornTree


class TestEytzingerGabrielHornTree:
    @pytest.fixture
    def test_array(self):
        np.random.seed(42)
        n = 128
        t = np.linspace(0, 1, n)
        return np.sin(4 * np.pi * t) + 0.3 * np.cos(10 * np.pi * t)

    def test_build_and_shapes(self, test_array):
        tree = EytzingerGabrielHornTree.build(test_array, q=0.5)
        assert tree.n == len(test_array)
        assert tree.tree_size == (2 * 128) - 1
        assert len(tree.values) == tree.tree_size
        assert len(tree.subtree_energies) == tree.tree_size

    def test_exact_point_queries(self, test_array):
        tree = EytzingerGabrielHornTree.build(test_array, q=0.5)
        for idx in [0, 15, 63, 127]:
            val, steps = tree.point_query(idx, epsilon=0.0)
            assert val == pytest.approx(test_array[idx], abs=1e-5)
            # Must be strictly bounded by max_depth + 1
            assert steps <= tree.max_depth + 1

    def test_range_sum_query(self, test_array):
        tree = EytzingerGabrielHornTree.build(test_array, q=0.5)
        start, end = 25, 75
        exact = float(np.sum(test_array[start:end]))
        approx, steps = tree.range_sum_query(start, end, epsilon=0.0)
        assert approx == pytest.approx(exact, abs=1e-4)

    def test_batch_query(self, test_array):
        tree = EytzingerGabrielHornTree.build(test_array, q=0.5)
        indices = np.array([5, 10, 20, 50, 100])
        batch_vals = tree.batch_point_query(indices, epsilon=0.0)
        np.testing.assert_allclose(batch_vals, test_array[indices], atol=1e-5)

    def test_equivalence_with_object_tree(self, test_array):
        eytz = EytzingerGabrielHornTree.build(test_array, q=0.5)
        obj_tree = AdaptiveGabrielHornTree.build(test_array, q=0.5)

        for idx in [7, 33, 77, 111]:
            val_eytz, _ = eytz.point_query(idx, epsilon=1e-3)
            val_obj, _ = obj_tree.point_query(idx, epsilon=1e-3)
            assert val_eytz == pytest.approx(val_obj, abs=0.1)
