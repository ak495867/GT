"""
Unit tests for 2D Gabriel Transform (spatial fields and images).
"""

import pytest
import numpy as np
from gabriel_transform.spatial_2d import GabrielTransform2D, Gabriel2DRepresentation


class TestGabrielTransform2D:
    @pytest.fixture
    def test_image(self):
        # 64x64 synthetic 2D surface: Gaussian bump + sinusoids
        r = np.linspace(-2, 2, 64)
        c = np.linspace(-2, 2, 64)
        R, C = np.meshgrid(r, c)
        img = np.exp(-(R**2 + C**2)) + 0.3 * np.sin(3 * R) * np.cos(3 * C)
        return img

    def test_forward_inverse_reconstruction(self, test_image):
        gt2d = GabrielTransform2D(q=0.55)
        rep = gt2d.forward(test_image)

        assert rep.shape == test_image.shape
        assert rep.num_levels > 0

        rec = gt2d.inverse(rep)
        assert rec.shape == test_image.shape

        # Residual norm should be small
        rel_err = np.linalg.norm(test_image - rec) / np.linalg.norm(test_image)
        assert rel_err < 0.2

    def test_sublinear_pixel_query(self, test_image):
        gt2d = GabrielTransform2D(q=0.5)
        rep = gt2d.forward(test_image)

        val, accessed = rep.query_pixel(32, 32, epsilon=1e-3)
        assert accessed > 0
        assert val == pytest.approx(test_image[32, 32], abs=0.25)

    def test_pruning_and_compression(self, test_image):
        gt2d = GabrielTransform2D(q=0.5)
        rep = gt2d.forward(test_image)

        eps = 1.0
        pruned = rep.prune_to_tolerance(eps)
        assert pruned.num_levels <= rep.num_levels
        rec_pruned = gt2d.inverse(pruned)
        diff = np.linalg.norm(test_image - rec_pruned)
        assert diff <= eps * 2.0
