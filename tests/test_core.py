"""
Unit tests for core Gabriel Transform and Horn Profiles.
"""

import pytest
import numpy as np
from gabriel_transform.core import (
    GeometricHornProfile,
    TorricelliHornProfile,
    ExponentialHornProfile,
    GabrielTransform,
    discrete_gabriel_transform,
    inverse_gabriel_transform,
)
from gabriel_transform.theory import verify_geometric_decay, verify_epsilon_bound


class TestHornProfiles:
    def test_geometric_profile_envelope_and_tail(self):
        q = 0.5
        c = 4.0
        prof = GeometricHornProfile(q=q, c=c)
        assert prof.envelope(0) == pytest.approx(4.0)
        assert prof.envelope(1) == pytest.approx(2.0)
        assert prof.envelope(2) == pytest.approx(1.0)
        # Tail at 0: 4 / (1 - 0.5) = 8.0
        assert prof.cumulative_tail(0) == pytest.approx(8.0)
        # Tail at 1: 2 / (1 - 0.5) = 4.0
        assert prof.cumulative_tail(1) == pytest.approx(4.0)

    def test_geometric_profile_required_depth(self):
        prof = GeometricHornProfile(q=0.5, c=1.0)
        # Tail(k) = 1.0 * 0.5^k / 0.5 = 2.0 * 0.5^k
        # For eps = 0.25: 2 * 0.5^k <= 0.25 => 0.5^k <= 0.125 => k >= 3
        k = prof.required_depth(0.25)
        assert k == 3
        assert prof.cumulative_tail(k) <= 0.25

    def test_torricelli_profile(self):
        prof = TorricelliHornProfile(alpha=1.0, gamma=2.0, c=2.0)
        assert prof.envelope(0) == pytest.approx(2.0)
        assert prof.envelope(1) == pytest.approx(0.5)
        # Required depth for eps
        k = prof.required_depth(0.1)
        assert k > 0
        assert prof.cumulative_tail(k) <= 0.1

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            GeometricHornProfile(q=1.5)
        with pytest.raises(ValueError):
            GeometricHornProfile(q=-0.1)
        with pytest.raises(ValueError):
            TorricelliHornProfile(gamma=0.5)


class TestGabrielTransform:
    def test_forward_inverse_reconstruction(self):
        # Generate smooth test signal
        t = np.linspace(0, 1, 128)
        x = np.sin(2 * np.pi * t) + 0.5 * np.cos(4 * np.pi * t)

        gt = GabrielTransform(q=0.5)
        rep = gt.forward(x)

        assert rep.num_levels > 0
        assert rep.original_length == 128

        # Invert transform
        x_rec = gt.inverse(rep)
        assert len(x_rec) == len(x)

        # Residual should be small
        diff = np.linalg.norm(x - x_rec)
        assert diff < 0.2

    def test_geometric_decay_bound(self):
        t = np.linspace(0, 1, 64)
        x = np.exp(-t) * np.sin(3 * np.pi * t)

        gt = GabrielTransform(q=0.6)
        rep = gt.forward(x)

        valid, checks = verify_geometric_decay(rep.level_norms, q=0.6, c=rep.profile.c)
        assert valid, f"Decay check failed: {checks}"

    def test_pruning_to_tolerance(self):
        x = np.linspace(0, 10, 128) + np.sin(np.linspace(0, 10, 128))
        gt = GabrielTransform(q=0.5)
        rep = gt.forward(x)

        eps = 1.0
        pruned_rep = rep.prune_to_tolerance(eps)
        assert pruned_rep.num_levels <= rep.num_levels

        x_rec_pruned = gt.inverse(pruned_rep)
        err = np.linalg.norm(x - x_rec_pruned)
        # Should be controlled by epsilon order
        assert err <= eps * 1.5

    def test_constant_signal(self):
        x = np.full(64, 5.0)
        gt = GabrielTransform(q=0.5)
        rep = gt.forward(x)
        x_rec = gt.inverse(rep)
        np.testing.assert_allclose(x, x_rec, atol=1e-5)

    def test_convenience_functions(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        rep = discrete_gabriel_transform(x, q=0.5)
        rec = inverse_gabriel_transform(rep)
        assert len(rec) == len(x)
