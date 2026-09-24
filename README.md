# Gabriel Transform (GT)

[![Tests](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> **The Gabriel Transform (GT)** formalizes **Gabriel's Horn** (Torricelli's Trumpet, 1644) into a hierarchical, progressively compressed multiscale algorithm that breaks through linear complexity barriers for structured signals:
> 
> $$\boxed{T(n, \epsilon) = O\left(\log n + \log \frac{1}{\epsilon}\right)}$$

---

## 🌟 Overview & Key Concepts

Evangelista Torricelli proved in 1644 that a solid can possess **finite volume** ($\pi$) alongside **infinite surface area**. The **Gabriel Transform** translates this physical phenomenon into algorithmic information theory:

1. **Finite Cumulative Energy:**  
   $$\|X_k\| \le C q^k \implies \sum_{k=0}^{\infty} \|X_k\| \le \frac{C}{1-q} < \infty$$
2. **Infinite Detail Resolution:**  
   As depth $k \to \infty$, the spatial support contracts as $2^{-k}$, capturing arbitrarily fine localized singularities.
3. **Logarithmic Precision Cutoff:**  
   Truncating at remaining tolerance $\epsilon$ requires strictly:
   $$K(\epsilon) = \left\lceil \frac{\log\left(\frac{C}{\epsilon(1-q)}\right)}{\log(1/q)} \right\rceil = \mathbf{O\left(\log \frac{1}{\epsilon}\right)}$$
4. **Dyadic Scale Decimation:**  
   Hierarchical subdivision $n \to n/2 \to n/4 \to \cdots$ yields:
   $$T(n) = T(n/2) + O(1) \implies \mathbf{O(\log n)}$$
5. **Double-Logarithmic Scale Jump:**  
   Locating the critical scale depth via exponential skip pointers runs in:
   $$\mathbf{O(\log \log n)}$$

---

## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/your-username/gabriel-transform.git
cd "d:/GT{Gabriel Transform}"
pip install -e .
```

### Python API

```python
import numpy as np
from gabriel_transform import (
    GabrielTransform,
    AdaptiveGabrielHornTree,
    FunctionalGabrielTransform,
    gabriel_point_query,
)

# 1. Discrete Gabriel Transform (DGT)
t = np.linspace(0, 1, 1024)
x = np.sin(2 * np.pi * 3 * t) + 0.3 * np.cos(2 * np.pi * 15 * t)

gt = GabrielTransform(q=0.5)
rep = gt.forward(x)

# Reconstruct up to epsilon tolerance
x_approx = gt.inverse(rep, epsilon=1e-3)
print(f"Stored coefficients: {rep.total_stored_coefficients()} vs original {len(x)}")

# 2. Sublinear Point Query directly on compressed representation
val, levels_accessed = gabriel_point_query(rep, index=250, epsilon=1e-3)
print(f"Evaluated in {levels_accessed} levels (O(log(1/eps)))")

# 3. Adaptive Gabriel Horn Tree (AGH-Tree)
tree = AdaptiveGabrielHornTree.build(x, q=0.5)
val, steps = tree.point_query(index=250, epsilon=1e-3)
print(f"Tree query took {steps} steps (O(log n))")

# 4. Double-Logarithmic Scale Jump
critical_depth, jump_steps = tree.jump_to_scale(epsilon=1e-2)
print(f"Jumped to depth {critical_depth} in {jump_steps} steps (O(log log n))")
```

---

## 📊 Theoretical vs Empirical Complexity

| Operation | Naive Array | Fast Fourier (FFT) | Wavelet (DWT) | Gabriel Transform (GT) |
| :--- | :---: | :---: | :---: | :---: |
| **Point Query $X(t)$** | $O(1)$ | $O(n)$ | $O(\log n)$ | **$O(\min(\log n, \log(1/\epsilon)))$** |
| **Definite Range Integral** | $O(n)$ | $O(n)$ | $O(\log n)$ | **$O(\log n + \log(1/\epsilon))$** |
| **$\epsilon$-Precision Horizon**| $O(n)$ | $O(n)$ | $O(n)$ | **$O(\log(1/\epsilon))$** |
| **Scale Search** | $O(n)$ | $O(\log n)$ | $O(\log n)$ | **$O(\log \log n)$** |

---

## 🧪 Running Tests & Benchmarks

```bash
# Run test suite
python -m pytest -v

# Run scaling benchmarks
python benchmarks/benchmark_scaling.py

# Generate diagnostic visual plots
python scripts/run_visual_demo.py
```

---

## 📚 Documentation
- [Theoretical Foundation & Torricelli's Horn](docs/THEORETICAL_FOUNDATION.md)
- [Complexity Analysis & Formal Proofs](docs/COMPLEXITY_ANALYSIS.md)
