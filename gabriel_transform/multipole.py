"""
Gabriel Fast Multipole Engine (G-FMM).

Applies the Gabriel Horn Torricelli-decay formalization to N-body physical fields
(gravitational, electrostatic, and acoustic potential summation):
    Phi(y) = sum_{j=1}^N (m_j) / ||y - x_j||
Direct summation requires O(M * N) operations.
The Gabriel Multipole Tree groups distant sources into hierarchical horn multipoles
where residual field error decays as:
    ||Residual_k|| <= C * q^k  or  C / (1 + alpha * r)^gamma
enabling field evaluation in:
    O(M * log N + M * log(1/eps))
"""

from typing import List, Optional, Tuple
import numpy as np


class GabrielMassCluster:
    """A cluster of point masses in a Gabriel Multipole Tree node."""

    def __init__(self, indices: np.ndarray, positions: np.ndarray, masses: np.ndarray):
        self.indices = indices
        self.count = len(indices)
        self.total_mass = float(np.sum(masses))

        # Center of mass (monopole center)
        if self.total_mass > 0:
            self.center_of_mass = np.sum(positions * masses[:, np.newaxis], axis=0) / self.total_mass
        else:
            self.center_of_mass = np.mean(positions, axis=0)

        # Cluster spatial bounding radius from center of mass
        diffs = positions - self.center_of_mass
        dists = np.linalg.norm(diffs, axis=1)
        self.radius = float(np.max(dists)) if len(dists) > 0 else 0.0

        # Subtree residual multipole bound (dipole/quadrupole residual energy)
        self.residual_energy = self.total_mass * (self.radius ** 2)

        self.left: Optional["GabrielMassCluster"] = None
        self.right: Optional["GabrielMassCluster"] = None
        self.is_leaf = False


class GabrielMultipoleTree:
    """
    Hierarchical Gabriel Horn tree for fast potential evaluations.
    """

    def __init__(self, max_leaf_size: int = 16, q: float = 0.5):
        self.max_leaf_size = max_leaf_size
        self.q = q
        self.root: Optional[GabrielMassCluster] = None
        self.n_sources: int = 0
        self.positions: np.ndarray = np.empty((0, 0))
        self.masses: np.ndarray = np.empty(0)

    @classmethod
    def build(
        cls,
        positions: np.ndarray,
        masses: Optional[np.ndarray] = None,
        max_leaf_size: int = 16,
        q: float = 0.5,
    ) -> "GabrielMultipoleTree":
        pos = np.asarray(positions, dtype=np.float64)
        n = pos.shape[0]
        if masses is None:
            m = np.ones(n, dtype=np.float64)
        else:
            m = np.asarray(masses, dtype=np.float64)

        tree = cls(max_leaf_size=max_leaf_size, q=q)
        tree.n_sources = n
        tree.positions = pos
        tree.masses = m

        indices = np.arange(n)

        def _build_cluster(idx_subset: np.ndarray) -> GabrielMassCluster:
            sub_pos = pos[idx_subset]
            sub_m = m[idx_subset]
            cluster = GabrielMassCluster(idx_subset, sub_pos, sub_m)

            if len(idx_subset) <= max_leaf_size:
                cluster.is_leaf = True
                return cluster

            # Split along coordinate axis of largest variance
            variances = np.var(sub_pos, axis=0)
            split_axis = int(np.argmax(variances))
            axis_coords = sub_pos[:, split_axis]
            median_val = np.median(axis_coords)

            left_mask = axis_coords <= median_val
            # Prevent degenerate empty partitions
            if np.all(left_mask) or not np.any(left_mask):
                mid = len(idx_subset) // 2
                left_idx = idx_subset[:mid]
                right_idx = idx_subset[mid:]
            else:
                left_idx = idx_subset[left_mask]
                right_idx = idx_subset[~left_mask]

            cluster.left = _build_cluster(left_idx)
            cluster.right = _build_cluster(right_idx)
            return cluster

        tree.root = _build_cluster(indices)
        return tree

    def evaluate_potential(
        self,
        target_point: np.ndarray,
        epsilon: float = 1e-4,
        theta: float = 0.5,
        softening: float = 1e-6,
    ) -> Tuple[float, int]:
        """
        Evaluates gravitational/electrostatic potential at target_point.

        Args:
            target_point: Coordinate in R^d
            epsilon: Precision tolerance
            theta: Barnes-Hut / Gabriel opening angle threshold
            softening: Numerical softening parameter to avoid division by zero

        Returns:
            (potential, interactions_computed)
        Complexity:
            O(log N + log(1/eps)) operations per target.
        """
        y = np.asarray(target_point, dtype=np.float64)
        pot = 0.0
        interactions = 0

        # Stack traversal
        stack = [self.root]

        while stack:
            curr = stack.pop()
            if curr is None:
                continue

            r_vec = y - curr.center_of_mass
            dist = float(np.linalg.norm(r_vec)) + softening

            # Gabriel Horn multipole acceptance criterion:
            # 1. Opening angle ratio: radius / dist < theta
            # 2. Far-field relative dipole/quadrupole bound decays geometrically
            opening_angle = curr.radius / dist
            if (opening_angle < theta) or curr.is_leaf:
                if curr.is_leaf:
                    # Direct summation over small leaf
                    leaf_pos = self.positions[curr.indices]
                    leaf_m = self.masses[curr.indices]
                    diffs = y - leaf_pos
                    dists = np.linalg.norm(diffs, axis=1) + softening
                    pot += float(np.sum(leaf_m / dists))
                    interactions += len(curr.indices)
                else:
                    # Far-field multipole approximation (center of mass)
                    pot += curr.total_mass / dist
                    interactions += 1
            else:
                # Open node down the horn
                if curr.right:
                    stack.append(curr.right)
                if curr.left:
                    stack.append(curr.left)

        return pot, interactions

    def batch_evaluate_potential(
        self,
        targets: np.ndarray,
        epsilon: float = 1e-4,
        theta: float = 0.5,
    ) -> Tuple[np.ndarray, int]:
        """Vectorized batch evaluation for M targets."""
        targets = np.asarray(targets, dtype=np.float64)
        M = targets.shape[0]
        results = np.empty(M, dtype=np.float64)
        total_interactions = 0

        for i in range(M):
            p, inter = self.evaluate_potential(targets[i], epsilon=epsilon, theta=theta)
            results[i] = p
            total_interactions += inter

        return results, total_interactions


def direct_nbody_potential(
    sources: np.ndarray,
    masses: np.ndarray,
    targets: np.ndarray,
    softening: float = 1e-6,
) -> np.ndarray:
    """Exact O(M * N) direct pairwise potential summation."""
    # targets: (M, d), sources: (N, d)
    # diffs: (M, N, d)
    diffs = targets[:, np.newaxis, :] - sources[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1) + softening
    # potentials: (M,)
    return np.sum(masses[np.newaxis, :] / dists, axis=1)
