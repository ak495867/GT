import sys
import argparse
import gabriel_transform


def cmd_info(args):
    print("=" * 65)
    print(f"Gabriel Transform (GT) v{gabriel_transform.__version__}")
    print("Hierarchical Multiscale Progressive Compression & Sublinear Engine")
    print("Inspired by Evangelista Torricelli's Horn Paradox (1644)")
    print("=" * 65)
    print("Modules Available:")
    print("  - core/         : 1D Discrete Gabriel Transform & Horn Profiles")
    print("  - trees/        : Adaptive & Hardware-Optimized Eytzinger Trees")
    print("  - spatial/      : 2D Spatial & Image Transform (Quadtree Funnel)")
    print("  - neural/       : Gabriel Attention for Long-Context Transformers")
    print("  - spectral/     : Spectral Chebyshev & Legendre Polynomial Horns")
    print("  - streaming/    : Real-Time Dynamic Dyadic Ingestion Stream")
    print("  - physics/      : Gabriel Fast Multipole N-Body Potential Engine")
    print("  - functional/   : Continuous Faber-Schauder Functional Transform")
    print("  - operators/    : Sublinear Point, Range, and Convolve Operators")
    print("  - theory/       : Complexity Models & Verification Checkers")
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
    from scripts.generate_full_gallery import generate_master_gallery

    run_all_visualizations(".")
    generate_advanced_plots(".")
    generate_master_gallery(".")
    print("All diagnostic figures generated successfully.")


def main():
    parser = argparse.ArgumentParser(description="Gabriel Transform (GT) CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    p_info = subparsers.add_parser("info", help="Show system info and capabilities")
    p_info.set_defaults(func=cmd_info)

    p_bench = subparsers.add_parser("benchmark", help="Run benchmark suites")
    p_bench.add_argument(
        "--suite",
        choices=["all", "scaling", "advanced", "multidomain"],
        default="all",
        help="Benchmark suite to run (default: all)",
    )
    p_bench.set_defaults(func=cmd_benchmark)

    p_vis = subparsers.add_parser("visualize", help="Generate all plots")
    p_vis.set_defaults(func=cmd_visualize)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
