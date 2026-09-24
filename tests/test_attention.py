"""
Unit tests for Gabriel Attention (hierarchical Transformer attention).
"""

import pytest
import numpy as np
from gabriel_transform.attention import gabriel_attention_numpy, HAS_TORCH

if HAS_TORCH:
    import torch
    from gabriel_transform.attention import GabrielAttention


class TestGabrielAttention:
    def test_numpy_attention_shape_and_stats(self):
        np.random.seed(42)
        N, D = 128, 32
        Q = np.random.randn(N, D)
        K = np.random.randn(N, D)
        V = np.random.randn(N, D)

        out, stats = gabriel_attention_numpy(
            Q, K, V,
            window_size=16,
            q_decay=0.5,
            epsilon=1e-3,
        )

        assert out.shape == (N, D)
        assert stats["sequence_length"] == N
        assert stats["sparsity_ratio"] > 0.0
        assert stats["gabriel_evaluations"] < stats["dense_evaluations"]

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_attention_module(self):
        B, N, D = 2, 64, 32
        x = torch.randn(B, N, D, requires_grad=True)
        attn = GabrielAttention(d_model=D, num_heads=4, window_size=16)

        out = attn(x)
        assert out.shape == (B, N, D)

        # Verify backpropagation
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
