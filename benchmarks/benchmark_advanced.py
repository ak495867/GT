"""
Advanced Benchmarks for Gabriel Transform (GT) Upgrades:
1. Eytzinger Flat Tree vs Pointer Object Tree Latency.
2. 2D Spatial Pixel Query vs Full Image Decoding.
3. Gabriel Attention vs Dense Self-Attention Sparsity & FLOPs.
4. Spectral Chebyshev Convergence vs Faber-Schauder Hats.
5. Streaming Ingestion Throughput (samples/sec).
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gabriel_transform.tree import AdaptiveGabrielHornTree
from gabriel_transform.eytzinger import EytzingerGabrielHornTree
from gabriel_transform.spatial_2d import GabrielTransform2D
from gabriel_transform.attention import gabriel_attention_numpy
from gabriel_transform.spectral import SpectralChebyshevHornTransform
from gabriel_transform.functional import FunctionalGabrielTransform
from gabriel_transform.streaming import StreamingGabrielTransform


def benchmark_eytzinger_vs_object():
    print("=" * 65)
    print("1. HARDWARE SPEEDUP: Eytzinger Flat Buffer vs Pointer Object Tree")
    print("=" * 65)
    n = 65536
    arr = np.sin(np.linspace(0, 16 * np.pi, n))

    # Build both
    t0 = time.perf_counter()
    obj_tree = AdaptiveGabrielHornTree.build(arr, q=0.5)
    t_obj_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    eytz_tree = EytzingerGabrielHornTree.build(arr, q=0.5)
    t_eytz_build = time.perf_counter() - t0

    # 10,000 random queries
    queries = np.random.randint(0, n, size=10000)

    # Object Tree
    t0 = time.perf_counter()
    for q in queries:
        _ = obj_tree.point_query(int(q), epsilon=1e-3)
    t_obj_query = (time.perf_counter() - t0) / 10000.0

    # Eytzinger Tree
    t0 = time.perf_counter()
    for q in queries:
        _ = eytz_tree.point_query(int(q), epsilon=1e-3)
    t_eytz_query = (time.perf_counter() - t0) / 10000.0

    speedup = t_obj_query / max(1e-9, t_eytz_query)
    print(f"Array size n: {n:,}")
    print(f"Object Tree Build:    {t_obj_build*1000:7.2f} ms | Query: {t_obj_query*1e6:6.2f} us")
    print(f"Eytzinger Tree Build: {t_eytz_build*1000:7.2f} ms | Query: {t_eytz_query*1e6:6.2f} us")
    print(f"-> Eytzinger Query Speedup: {speedup:5.1f}x FASTER (Zero pointer dereferencing!)", flush=True)


def benchmark_2d_spatial():
    print("\n" + "=" * 65)
    print("2. 2D SPATIAL TRANSFORM: Sublinear Pixel Query vs Full Image Decoding")
    print("=" * 65)
    H, W = 256, 256
    r = np.linspace(-2, 2, H)
    c = np.linspace(-2, 2, W)
    R, C = np.meshgrid(r, c)
    img = np.exp(-(R**2 + C**2)) + 0.2 * np.sin(5 * R) * np.cos(5 * C)

    gt2d = GabrielTransform2D(q=0.5)
    rep = gt2d.forward(img)

    # Full decode time
    t0 = time.perf_counter()
    for _ in range(10):
        _ = gt2d.inverse(rep)
    full_decode_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0

    # Sublinear pixel query (1000 random pixels)
    queries = [(np.random.randint(0, H), np.random.randint(0, W)) for _ in range(1000)]
    t0 = time.perf_counter()
    for r_idx, c_idx in queries:
        _, _ = rep.query_pixel(r_idx, c_idx, epsilon=1e-3)
    pixel_query_us = ((time.perf_counter() - t0) / 1000.0) * 1e6

    print(f"2D Image Shape: {H}x{W} ({H*W:,} pixels)")
    print(f"Full Image Decode Time: {full_decode_ms:6.2f} ms")
    print(f"Gabriel Pixel Query Time: {pixel_query_us:6.2f} us")
    print(f"-> Pixel Query is {full_decode_ms * 1000.0 / pixel_query_us:5.0f}x FASTER than full decode!", flush=True)


def benchmark_gabriel_attention():
    print("\n" + "=" * 65)
    print("3. GABRIEL ATTENTION: Sparsity & FLOPs vs Dense Attention")
    print("=" * 65)
    for N in [256, 512, 1024, 2048]:
        D = 32
        Q = np.random.randn(N, D)
        K = np.random.randn(N, D)
        V = np.random.randn(N, D)

        t0 = time.perf_counter()
        out, stats = gabriel_attention_numpy(Q, K, V, window_size=32, q_decay=0.5, epsilon=1e-3)
        elapsed = time.perf_counter() - t0

        dense_ops = N * N
        gabriel_ops = stats["gabriel_evaluations"]
        saved = stats["sparsity_ratio"] * 100.0

        print(
            f"N = {N:4d} | Dense Pairs: {dense_ops:8,d} | Gabriel Evals: {gabriel_ops:7,d} | "
            f"Saved: {saved:5.1f}% FLOPs | Time: {elapsed*1000:6.1f} ms",
            flush=True,
        )


def benchmark_spectral_convergence():
    print("\n" + "=" * 65)
    print("4. SPECTRAL CHEBYSHEV: Spectral vs Piecewise-Linear Rates")
    print("=" * 65)
    # Analytic Runge function
    func = lambda x: 1.0 / (1.0 + 25.0 * (x**2))

    cheb = SpectralChebyshevHornTransform(func=func, domain=(-1.0, 1.0), q=0.5)
    cheb.fit(max_degree=32)

    hat = FunctionalGabrielTransform(func=func, domain=(-1.0, 1.0), q=0.25)
    hat.fit(max_levels=8)

    test_x = 0.33
    exact = func(test_x)

    val_cheb, deg = cheb.evaluate(test_x, epsilon=1e-6)
    val_hat = hat.evaluate(test_x, epsilon=1e-6)

    err_cheb = abs(exact - val_cheb)
    err_hat = abs(exact - val_hat)

    print(f"Test Function: Runge 1 / (1 + 25 x^2) at x = {test_x}")
    print(f"Chebyshev Spectral Error (deg={deg}): {err_cheb:1.2e}")
    print(f"Faber-Schauder Hat Error:          {err_hat:1.2e}")
    print(f"-> Spectral Gabriel Horn is {err_hat / max(1e-16, err_cheb):1.0e}x more accurate!", flush=True)


def benchmark_streaming():
    print("\n" + "=" * 65)
    print("5. STREAMING INGESTION: Real-Time Stream Throughput")
    print("=" * 65)
    n_samples = 50000
    stream = StreamingGabrielTransform(q=0.5)

    data = np.random.randn(n_samples)
    t0 = time.perf_counter()
    stream.append_batch(data)
    elapsed = time.perf_counter() - t0

    throughput = n_samples / elapsed
    print(f"Ingested {n_samples:,} streaming samples in {elapsed*1000:6.2f} ms")
    print(f"-> Real-Time Throughput: {throughput:,.0f} samples/sec!", flush=True)


if __name__ == "__main__":
    benchmark_eytzinger_vs_object()
    benchmark_2d_spatial()
    benchmark_gabriel_attention()
    benchmark_spectral_convergence()
    benchmark_streaming()
    print("\nAll advanced benchmarks completed successfully!")
