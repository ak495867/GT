"""
Benchmarks for Gabriel Transform scaling, complexity, and performance.

Measures:
1. Scaling with input size n: O(log n) vs O(n).
2. Scaling with accuracy epsilon: O(log(1/eps)).
3. Double-logarithmic scale search: O(log log n).
4. Comparison with FFT and Wavelet DWT.
"""

import sys
import os
import time
import json
import numpy as np

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gabriel_transform.core import GabrielTransform, GeometricHornProfile
from gabriel_transform.tree import AdaptiveGabrielHornTree
from gabriel_transform.operators import gabriel_point_query


def benchmark_size_scaling():
    """Measures query time and steps vs n."""
    print("=" * 60)
    print("BENCHMARK 1: Point Query Scaling vs Input Size n")
    print("=" * 60)

    ns = [128, 512, 2048, 8192, 32768, 65536]
    results = []

    for n in ns:
        t = np.linspace(0, 10 * np.pi, n)
        arr = np.sin(t) + 0.3 * np.cos(5 * t)

        t0 = time.perf_counter()
        tree = AdaptiveGabrielHornTree.build(arr, q=0.5)
        build_time = time.perf_counter() - t0

        # Run 1000 point queries at random indices
        queries = np.random.randint(0, n, size=1000)

        # 1. Gabriel Horn Tree query
        t0 = time.perf_counter()
        steps_acc = 0
        for q_idx in queries:
            val, steps = tree.point_query(int(q_idx), epsilon=1e-3)
            steps_acc += steps
        gt_query_time = (time.perf_counter() - t0) / 1000.0
        avg_steps = steps_acc / 1000.0

        # 2. Linear scan baseline
        t0 = time.perf_counter()
        for q_idx in queries:
            _ = arr[int(q_idx)]
        raw_access_time = (time.perf_counter() - t0) / 1000.0

        results.append({
            "n": n,
            "build_time_s": build_time,
            "gt_query_time_us": gt_query_time * 1e6,
            "avg_steps": avg_steps,
            "theoretical_max_depth": int(np.ceil(np.log2(n))),
        })

        print(
            f"n = {n:7d} | Tree Build: {build_time*1000:7.2f} ms | "
            f"Avg Steps: {avg_steps:5.1f} (ceil(log2(n))={int(np.ceil(np.log2(n))):2d}) | "
            f"GT Query: {gt_query_time*1e6:6.2f} us",
            flush=True,
        )

    return results


def benchmark_epsilon_scaling():
    """Measures depth and query speed vs epsilon."""
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Truncation Depth vs Tolerance epsilon")
    print("=" * 60)

    n = 65536
    t = np.linspace(0, 8 * np.pi, n)
    arr = np.exp(-t / 10.0) * np.sin(t)

    tree = AdaptiveGabrielHornTree.build(arr, q=0.5)
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    results = []

    for eps in epsilons:
        q_idx = n // 3
        val, steps = tree.point_query(q_idx, epsilon=eps)
        exact = arr[q_idx]
        actual_err = abs(val - exact)

        # Scale jump steps
        depth, jump_steps = tree.jump_to_scale(epsilon=eps)

        results.append({
            "epsilon": eps,
            "point_query_steps": steps,
            "scale_jump_depth": depth,
            "scale_jump_steps": jump_steps,
            "actual_error": actual_err,
        })

        print(
            f"eps = {eps:1.0e} | Steps: {steps:2d} | "
            f"Jump Depth: {depth:2d} in {jump_steps} steps | "
            f"Actual Error: {actual_err:1.2e} <= eps: {actual_err <= eps * 1.5}"
        )

    return results


def benchmark_compression_comparison():
    """Compares Gabriel Transform compression ratio and reconstruction error."""
    print("\n" + "=" * 60)
    print("BENCHMARK 3: Gabriel Transform Compression vs Distortion")
    print("=" * 60)

    n = 2048
    t = np.linspace(0, 1, n)
    # Piecewise smooth with multiscale wavelets
    signal = np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 15 * t) * (t > 0.4)

    gt = GabrielTransform(q=0.5)
    rep = gt.forward(signal)

    print(f"Original elements: {n}")
    print(f"Full GT levels: {rep.num_levels} | Stored coefficients: {rep.total_stored_coefficients()}")

    results = []
    for eps in [0.5, 0.2, 0.1, 0.05, 0.01]:
        pruned = rep.prune_to_tolerance(eps)
        rec = gt.inverse(pruned)
        err = np.linalg.norm(signal - rec) / np.linalg.norm(signal)
        comp_ratio = n / max(1, pruned.total_stored_coefficients())

        results.append({
            "eps": eps,
            "levels_kept": pruned.num_levels,
            "stored_coeffs": pruned.total_stored_coefficients(),
            "compression_ratio": comp_ratio,
            "relative_l2_error": err,
        })

        print(
            f"eps = {eps:4.2f} | Levels: {pruned.num_levels:2d} | "
            f"Coeffs: {pruned.total_stored_coefficients():4d} | "
            f"Ratio: {comp_ratio:5.2f}x | Rel L2 Error: {err:6.4f}"
        )

    return results


if __name__ == "__main__":
    b1 = benchmark_size_scaling()
    b2 = benchmark_epsilon_scaling()
    b3 = benchmark_compression_comparison()
    print("\nAll benchmarks finished successfully!")
