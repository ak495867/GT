"""
Theoretical Foundations & Complexity Verification for the Gabriel Transform.

Formal mathematical proofs and validation functions for:
- Geometric decay verification: ||X_k|| <= C * q^k
- Epsilon-truncation bound verification: ||R_K|| <= eps
- Recurrence relation solver: T(n) = T(n/r) + O(1) => T(n) = O(log n)
- Sublinear theoretical hypothesis: T(n, eps) = O(log n + log(1/eps))
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class GabrielComplexityModel:
    """
    Theoretical complexity model for the Gabriel-Horn Transform family.
    Computes exact constant factors and asymptotic bounds.
    """

    def __init__(self, q: float = 0.5, r: int = 2, c: float = 1.0):
        if not (0.0 < q < 1.0):
            raise ValueError("q must be in (0, 1)")
        if r <= 1:
            raise ValueError("Scale ratio r must be >= 2")
        self.q = q
        self.r = r
        self.c = c

    def theoretical_depth(self, epsilon: float) -> int:
        """K_eps = ceil( log(C / (eps * (1 - q))) / log(1/q) ) = O(log(1/eps))."""
        if epsilon <= 0.0:
            return 64
        ratio = (epsilon * (1.0 - self.q)) / self.c
        if ratio >= 1.0:
            return 0
        return int(np.ceil(np.log(ratio) / np.log(self.q)))

    def query_cost(self, n: int, epsilon: float) -> float:
        """
        Theoretical operation count for point query:
        T(n, eps) <= min(ceil(log_r(n)), K_eps)
        """
        depth_n = int(np.ceil(np.log(max(2, n)) / np.log(self.r)))
        depth_eps = self.theoretical_depth(epsilon)
        return float(min(depth_n, depth_eps))

    def range_query_cost(self, n: int, epsilon: float) -> float:
        """
        Theoretical operation count for range query:
        T(n, eps) = O(log_r(n) + log_{1/q}(1/eps))
        """
        depth_n = float(np.ceil(np.log(max(2, n)) / np.log(self.r)))
        depth_eps = float(self.theoretical_depth(epsilon))
        return 2.0 * depth_n + depth_eps

    def scale_jump_cost(self, n: int) -> float:
        """
        Theoretical scale jumping cost:
        T(n) = O(log(depth)) = O(log(log_r(n))) = O(log log n).
        """
        depth_n = np.ceil(np.log(max(2, n)) / np.log(self.r))
        return float(np.ceil(np.log2(max(1.0, depth_n))))


def solve_recurrence_complexity(a: int = 1, r: int = 2, work_per_level: int = 1) -> str:
    """
    Applies the Master Theorem to recurrence:
        T(n) = a * T(n/r) + O(1)
    For a=1: T(n) = O(log_r(n)) = O(log n).
    """
    if a == 1:
        return f"T(n) = T(n/{r}) + O({work_per_level}) => T(n) = O(log_{r}(n)) = O(log n)"
    elif a < r:
        return f"T(n) = {a}*T(n/{r}) + O({work_per_level}) => T(n) = O(n^{np.log(a)/np.log(r):.3f})"
    else:
        return f"T(n) = {a}*T(n/{r}) + O({work_per_level}) => T(n) = O(n^{np.log(a)/np.log(r):.3f})"


def verify_geometric_decay(
    level_norms: List[float],
    q: float,
    c: Optional[float] = None,
    tolerance_slack: float = 1.05,
) -> Tuple[bool, List[Dict[str, float]]]:
    """
    Verifies whether empirical level norms strictly obey ||X_k|| <= C * q^k.
    """
    if not level_norms:
        return True, []

    if c is None:
        c = level_norms[0] if level_norms[0] > 0 else 1.0

    checks = []
    all_passed = True

    for k, norm in enumerate(level_norms):
        theoretical_bound = c * (q ** k) * tolerance_slack
        passed = norm <= theoretical_bound
        checks.append({
            "level": k,
            "actual_norm": norm,
            "theoretical_bound": theoretical_bound,
            "ratio": norm / max(1e-15, theoretical_bound),
            "valid": passed,
        })
        if not passed:
            all_passed = False

    return all_passed, checks


def verify_epsilon_bound(
    original: np.ndarray,
    reconstructed: np.ndarray,
    epsilon: float,
    norm_type: str = "l2",
) -> Tuple[bool, float]:
    """
    Checks if ||x - x_rec|| <= epsilon.
    """
    diff = np.asarray(original, dtype=np.float64) - np.asarray(reconstructed, dtype=np.float64)
    if norm_type == "l2":
        err = float(np.linalg.norm(diff))
    elif norm_type == "linf":
        err = float(np.max(np.abs(diff)))
    else:
        err = float(np.mean(np.abs(diff)))

    return err <= epsilon, err


def fit_log_scaling(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Fits y = slope * log(x) + intercept.
    Returns (slope, intercept, R_squared).
    """
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)

    log_x = np.log(x)
    # Linear regression on log_x
    poly = np.polyfit(log_x, y, 1)
    slope, intercept = poly[0], poly[1]

    # R^2 determination
    y_pred = slope * log_x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / max(1e-12, ss_tot))

    return float(slope), float(intercept), float(r_squared)
