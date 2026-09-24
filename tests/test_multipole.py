"""
Unit tests for Gabriel Fast Multipole Engine (gravitational/electrostatic potential summation).
"""

import pytest
import numpy as np
from gabriel_transform.multipole import (
    GabrielMultipoleTree,
    direct_nbody_potential,
)


class TestGabrielMultipole:
    @pytest.fixture
    def nbody_system(self):
        np.random.seed(42)
        N = 256
        positions = np.random.uniform(-1.0, 1.0, size=(N, 3))
        masses = np.random.uniform(0.5, 2.0, size=N)
        # Far-field targets where Gabriel multipole clustering takes effect
        targets = np.random.uniform(4.0, 8.0, size=(10, 3))
        return positions, masses, targets

    def test_multipole_tree_build(self, nbody_system):
        positions, masses, _ = nbody_system
        tree = GabrielMultipoleTree.build(positions, masses=masses, max_leaf_size=16)
        assert tree.root is not None
        assert tree.n_sources == len(positions)
        assert tree.root.total_mass == pytest.approx(np.sum(masses))

    def test_potential_accuracy_and_speedup(self, nbody_system):
        positions, masses, targets = nbody_system
        tree = GabrielMultipoleTree.build(positions, masses=masses, max_leaf_size=16)

        exact = direct_nbody_potential(positions, masses, targets)
        approx, total_interactions = tree.batch_evaluate_potential(targets, epsilon=1e-3, theta=0.5)

        # Average interactions per target must be less than direct N=256
        avg_inter = total_interactions / len(targets)
        assert avg_inter < len(positions)

        # Relative error should be small
        rel_errors = np.abs(exact - approx) / np.abs(exact)
        assert np.max(rel_errors) < 0.05
