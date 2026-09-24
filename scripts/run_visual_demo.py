"""
Visual demonstration script for the Gabriel Transform (GT).
Generates illustrative figures and saves them in the artifacts directory / workspace.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gabriel_transform.core import GabrielTransform, GeometricHornProfile
from gabriel_transform.tree import AdaptiveGabrielHornTree
from gabriel_transform.visualization import (
    plot_gabriel_horn_geometry,
    plot_multiscale_decomposition,
    plot_complexity_scaling,
)


def run_all_visualizations(output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    print("Generating Gabriel Transform Visualizations...")

    # 1. Gabriel Horn Classical Geometry
    geom_path = os.path.join(output_dir, "gabriel_horn_geometry.png")
    plot_gabriel_horn_geometry(geom_path)
    print(f"[OK] Generated {geom_path}")

    # 2. Multiscale Decomposition
    n = 256
    t = np.linspace(0, 1, n)
    # Signal with multiscale wavelets and smooth envelope
    signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.cos(2 * np.pi * 8 * t) + 0.25 * np.sin(2 * np.pi * 24 * t)

    gt = GabrielTransform(q=0.55)
    rep = gt.forward(signal)
    rec = gt.inverse(rep)

    decomp_path = os.path.join(output_dir, "gabriel_decomposition.png")
    plot_multiscale_decomposition(
        original=signal,
        reconstructed=rec,
        level_norms=rep.level_norms,
        q=0.55,
        save_path=decomp_path,
    )
    print(f"[OK] Generated {decomp_path}")

    # 3. Complexity Scaling Plots
    ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    naive_steps = [n for n in ns]
    gt_steps = []

    for n_val in ns:
        arr_test = np.sin(np.linspace(0, 4 * np.pi, n_val))
        tree_test = AdaptiveGabrielHornTree.build(arr_test, q=0.5)
        _, s = tree_test.point_query(n_val // 3, epsilon=1e-4)
        gt_steps.append(s)

    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    prof = GeometricHornProfile(q=0.5, c=1.0)
    eps_steps = [prof.required_depth(e) for e in epsilons]

    scaling_path = os.path.join(output_dir, "gabriel_complexity_scaling.png")
    plot_complexity_scaling(
        ns=ns,
        naive_steps=naive_steps,
        gt_steps=gt_steps,
        epsilons=epsilons,
        eps_steps=eps_steps,
        save_path=scaling_path,
    )
    print(f"[OK] Generated {scaling_path}")

    # 4. Rate-Distortion / Compression Tradeoff
    rd_path = os.path.join(output_dir, "gabriel_compression_rd.png")
    fig, ax = plt.subplots(figsize=(7, 5))
    stored_list = []
    errors = []
    for eps in np.logspace(-0.5, -3.0, 10):
        pruned = rep.prune_to_tolerance(eps)
        rec_p = gt.inverse(pruned)
        rel_err = np.linalg.norm(signal - rec_p) / np.linalg.norm(signal)
        stored_list.append(pruned.total_stored_coefficients())
        errors.append(rel_err)

    ax.plot(stored_list, errors, "mo-", linewidth=2, markersize=7)
    ax.set_title("Gabriel Transform: Rate-Distortion Profile", fontsize=12, fontweight="bold")
    ax.set_xlabel("Retained Horn Coefficients", fontsize=11)
    ax.set_ylabel("Relative $L_2$ Error", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(rd_path, dpi=200)
    plt.close()
    print(f"[OK] Generated {rd_path}")

    print("All visualizations created successfully.")


if __name__ == "__main__":
    # Save in current directory (workspace)
    run_all_visualizations(".")

    # Also save in artifact directory if accessible
    artifact_dir = r"/artifacts"
    if os.path.exists(artifact_dir):
        run_all_visualizations(artifact_dir)
