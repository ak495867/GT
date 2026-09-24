"""
Command-Line Interface (CLI) for Gabriel Transform (GT) 2.0.

Usage:
    python -m gabriel_transform.cli info
    python -m gabriel_transform.cli benchmark [--suite all|scaling|advanced|multidomain]
    python -m gabriel_transform.cli visualize
"""

import sys
import argparse
import numpy as np

import gabriel_transform
from gabriel_transform.core import GabrielTransform
from gabriel_transform.eytzinger import EytzingerGabrielHornTree
from gabriel_transform.spatial_2d import GabrielTransform2D


def cmd_info(args):
    print("=" * 65)
    print(f"Gabriel Transform (GT) v{gabriel_transform.__version__}")
    print("Hierarchical Multiscale Progressive Compression & Sublinear Engine")
    print("Inspired by Evangelista Torricelli's Horn Paradox (1644)")
    print("=" * 65)
    print("Modules Available:")
    print("  - core.py         : 1D Discrete Gabriel Transform & Horn Profiles")
    print("  - tree.py         : Adaptive Gabriel Horn Tree with O(log log n) Skips")
    print("  - eytzinger.py    : Flat Contiguous Memory Buffer (Hardware Speedup)")
    print("  - spatial_2d.py   : 2D Spatial & Image Transform (Quadtree Funnel)")
    print("  - attention.py    : Gabriel Attention for Long-Context Transformers")
    print("  - spectral.py     : Spectral Chebyshev & Legendre Polynomial Horns")
    print("  - streaming.py    : Real-Time Dynamic Dyadic Ingestion Stream")
    print("  - multipole.py    : Gabriel Fast Multipole N-Body Potential Engine")
    print("  - operators.py    : Sublinear Point, Range, and Convolve Operators")
    print("=" * 65)


def cmd_benchmark(args):
    suite = args.suite
    print(f"Running Gabriel Transform benchmark suite: '{suite}'...")

    if suite in ("all", "scaling"):
        from benchmarks.benchmark_scaling import (
            benchmark_size_scaling,
            benchmark_epsilon_scaling,
            benchmark_compression_comparison,
        )
        benchmark_size_scaling()
        benchmark_epsilon_scaling()
        benchmark_compression_comparison()

    if suite in ("all", "advanced"):
        from benchmarks.benchmark_advanced import (
            benchmark_eytzinger_vs_object,
            benchmark_2d_spatial,
            benchmark_gabriel_attention,
            benchmark_spectral_convergence,
            benchmark_streaming,
        )
        benchmark_eytzinger_vs_object()
        benchmark_2d_spatial()
        benchmark_gabriel_attention()
        benchmark_spectral_convergence()
        benchmark_streaming()

    if suite in ("all", "multidomain"):
        from benchmarks.benchmark_multidomain import run_all_multidomain_benchmarks
        run_all_multidomain_benchmarks()

    print("\nBenchmark run completed successfully!")


def cmd_visualize(args):
    print("Generating all visualization figures...")
    from scripts.run_visual_demo import run_all_visualizations
    from scripts.run_advanced_visuals import generate_advanced_plots

    run_all_visualizations(".")
    generate_advanced_plots(".")
    print("All diagnostic figures generated successfully.")


def main():
    parser = argparse.ArgumentParser(description="Gabriel Transform (GT) CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # info
    p_info = subparsers.add_parser("info", help="Show system info and capabilities")
    p_info.set_defaults(func=cmd_info)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark suites")
    p_bench.add_argument(
        "--suite",
        choices=["all", "scaling", "advanced", "multidomain"],
        default="all",
        help="Benchmark suite to run (default: all)",
    )
    p_bench.set_defaults(func=cmd_benchmark)

    # visualize
    p_vis = subparsers.add_parser("visualize", help="Generate all plots")
    p_vis.set_defaults(func=cmd_visualize)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
