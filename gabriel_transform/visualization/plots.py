from typing import List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_gabriel_horn_geometry(save_path: str = "gabriel_horn_geometry.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.linspace(1.0, 10.0, 500)
    y_upper = 1.0 / x
    y_lower = -1.0 / x

    ax1.plot(x, y_upper, "b-", linewidth=2, label="Horn Envelope $r(x) = 1/x$")
    ax1.plot(x, y_lower, "b-", linewidth=2)
    ax1.fill_between(x, y_lower, y_upper, color="cyan", alpha=0.25, label="Horn Cross-Section")
    ax1.set_title("Gabriel's Horn Geometry (Torricelli 1644)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Axial Scale Coordinate $x$", fontsize=11)
    ax1.set_ylabel("Radius $r(x)$", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    vol = np.pi * (1.0 - 1.0 / x)
    area = 2.0 * np.pi * np.log(x)

    ax2.plot(x, vol, "g-", linewidth=2.5, label="Cumulative Volume $\\pi(1 - 1/x) \\to \\pi$")
    ax2.plot(x, area, "r--", linewidth=2.5, label="Surface Area $2\\pi \\ln(x) \\to \\infty$")
    ax2.axhline(np.pi, color="darkgreen", linestyle=":", label="Volume Limit $\\pi$ (Finite)")
    ax2.set_title("The Gabriel Paradox: Finite Volume vs Infinite Detail", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Axial Depth $x$", fontsize=11)
    ax2.set_ylabel("Value", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="center right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_multiscale_decomposition(
    original: np.ndarray,
    reconstructed: np.ndarray,
    level_norms: List[float],
    q: float,
    save_path: str = "gabriel_decomposition.png",
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(original, "k-", alpha=0.5, linewidth=2, label="Original Signal $X$")
    ax1.plot(reconstructed, "r--", linewidth=1.5, label="Gabriel Reconstruction $\\hat{X}_K$")
    ax1.set_title("Signal Reconstruction", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Sample Index $i$", fontsize=11)
    ax1.set_ylabel("Amplitude", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    k_vals = np.arange(len(level_norms))
    c_bound = level_norms[0] if level_norms else 1.0
    geo_bound = [c_bound * (q ** k) for k in k_vals]

    ax2.semilogy(k_vals, level_norms, "bo-", linewidth=2, markersize=6, label="Empirical Level Norm $\\|X_k\\|$")
    ax2.semilogy(k_vals, geo_bound, "r--", linewidth=2, label=f"Gabriel Bound $C \\cdot q^k$ ($q={q}$)")
    ax2.set_title("Geometric Energy Funnel (||X_k|| <= C * q^k)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Horn Level $k$", fontsize=11)
    ax2.set_ylabel("Norm $\\|X_k\\|$ (Log Scale)", fontsize=11)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_complexity_scaling(
    ns: List[int],
    naive_steps: List[int],
    gt_steps: List[int],
    epsilons: List[float],
    eps_steps: List[int],
    save_path: str = "gabriel_complexity_scaling.png",
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ns, naive_steps, "r-o", linewidth=2, label="Naive Linear Scan $O(n)$")
    ax1.plot(ns, gt_steps, "b-s", linewidth=2, label="Gabriel Horn Query $O(\\log n)$")
    ax1.set_title("Query Scaling vs Input Size $n$", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Input Size $n$", fontsize=11)
    ax1.set_ylabel("Operation Steps", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    log_inv_eps = [np.log10(1.0 / e) for e in epsilons]
    ax2.plot(log_inv_eps, eps_steps, "g-^", linewidth=2, markersize=7, label="Empirical Levels Evaluated")
    ax2.set_title("Query Scaling vs Accuracy $\\epsilon$ ($O(\\log(1/\\epsilon))$)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("$\\log_{10}(1 / \\epsilon)$ (Digits of Precision)", fontsize=11)
    ax2.set_ylabel("Required Depth $K$", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
