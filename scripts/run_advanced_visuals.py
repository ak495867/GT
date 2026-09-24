"""
Advanced Visualizations for Gabriel Transform (GT) 2.0:
1. 2D Gabriel Transform Multiscale Surface & Residual Map.
2. Gabriel Attention Funnel Matrix Heatmap.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gabriel_transform.spatial_2d import GabrielTransform2D
from gabriel_transform.attention import gabriel_attention_numpy


def generate_advanced_plots(output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 2D Spatial Gabriel Transform
    H, W = 128, 128
    r = np.linspace(-2.5, 2.5, H)
    c = np.linspace(-2.5, 2.5, W)
    R, C = np.meshgrid(r, c)
    # 2D surface with Gaussian bells and ripples
    surface = np.exp(-(R**2 + C**2)) + 0.3 * np.cos(4 * R) * np.sin(4 * C)

    gt2d = GabrielTransform2D(q=0.5)
    rep = gt2d.forward(surface)
    rec = gt2d.inverse(rep)
    residual = np.abs(surface - rec)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = axes[0].imshow(surface, cmap="viridis", origin="lower")
    axes[0].set_title("Original 2D Surface", fontsize=12, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(rec, cmap="viridis", origin="lower")
    axes[1].set_title(f"Gabriel 2D Reconstruction ({rep.num_levels} Levels)", fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(residual, cmap="magma", origin="lower")
    axes[2].set_title(r"Residual Error Map ($|I - \hat{I}| \leq \epsilon$)", fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    p2d = os.path.join(output_dir, "gabriel_2d_heatmap.png")
    plt.savefig(p2d, dpi=200)
    plt.close()
    print(f"[OK] Generated {p2d}")

    # 2. Gabriel Attention Funnel Matrix Heatmap
    N = 128
    D = 16
    Q = np.random.randn(N, D)
    K = np.random.randn(N, D)
    V = np.random.randn(N, D)

    # Compute attention matrix pattern
    scale = 1.0 / np.sqrt(D)
    dense_scores = np.dot(Q, K.T) * scale
    dense_probs = np.exp(dense_scores - np.max(dense_scores, axis=-1, keepdims=True))
    dense_probs /= np.sum(dense_probs, axis=-1, keepdims=True)

    # Gabriel masked pattern
    window = 16
    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    gabriel_weights = np.copy(dense_probs)
    # Apply Torricelli horn suppression
    horn_mask = 1.0 / (1.0 + 0.2 * np.maximum(0, dist - window)) ** 2
    gabriel_weights *= horn_mask
    gabriel_weights /= np.sum(gabriel_weights, axis=-1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    im_dense = ax1.imshow(dense_probs, cmap="Blues", origin="upper")
    ax1.set_title("Dense Attention Matrix $O(N^2)$", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Key Token Index $j$")
    ax1.set_ylabel("Query Token Index $i$")
    plt.colorbar(im_dense, ax=ax1, fraction=0.046, pad=0.04)

    im_gab = ax2.imshow(gabriel_weights, cmap="Blues", origin="upper")
    ax2.set_title("Gabriel Horn Funnel Attention $O(N \\log N)$", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Key Token Index $j$")
    ax2.set_ylabel("Query Token Index $i$")
    plt.colorbar(im_gab, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    p_attn = os.path.join(output_dir, "gabriel_attention_map.png")
    plt.savefig(p_attn, dpi=200)
    plt.close()
    print(f"[OK] Generated {p_attn}")


if __name__ == "__main__":
    generate_advanced_plots(".")
    artifact_dir = r"C:\Users\Ak\.gemini\antigravity\brain\ba02297a-c9f0-482e-bfe8-45ef7cde266b"
    if os.path.exists(artifact_dir):
        generate_advanced_plots(artifact_dir)
