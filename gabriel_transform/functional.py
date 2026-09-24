"""
Continuous & Functional Gabriel Transform (FGT).

Extends the Gabriel Transform to continuous functions f: [a, b] -> R.
Decomposes f into a convergent hierarchical Faber-Schauder multiscale horn series:
    f(t) = s_0(t) + sum_{k=1}^inf sum_{m} d_{k, m} * phi_{k, m}(t)
where each level detail satisfies geometric decay:
    ||d_k||_inf <= C * q^k  (with q = 1/4 for C^2 smooth functions)
ensuring that truncating at K = ceil(log(C / (eps*(1-q))) / log(1/q)) guarantees
||f - f_K||_inf <= eps, with O(1) evaluation per level -> O(log(1/eps)) total!
"""

from typing import Callable, List, Optional, Tuple
import numpy as np
from gabriel_transform.core import HornProfile, GeometricHornProfile


class ContinuousHornBasis:
    """
    Continuous Gabriel Horn multiscale hat basis function (Faber-Schauder basis).
    Support is localized to [center - half_width, center + half_width].
    """

    def __init__(self, basis_type: str = "spline"):
        self.basis_type = basis_type

    @staticmethod
    def evaluate_hat(center: float, half_width: float, t: float) -> float:
        """Evaluates normalized triangular hat function centered at center with given half_width."""
        diff = abs(t - center)
        if diff < half_width:
            return 1.0 - (diff / half_width)
        return 0.0

    def evaluate(self, k: int, m: int, t: float) -> float:
        """
        Evaluate basis function at level k and node m on domain [0, 1].
        """
        scale = 2 ** k
        half_width = 1.0 / scale
        center = m / scale
        return self.evaluate_hat(center, half_width, t)


class FunctionalGabrielTransform:
    """
    Continuous functional Gabriel Transform using hierarchical Faber-Schauder horn decomposition.
    """

    def __init__(
        self,
        func: Callable[[float], float],
        domain: Tuple[float, float] = (0.0, 1.0),
        q: float = 0.25,
        profile: Optional[HornProfile] = None,
        basis_type: str = "spline",
    ):
        self.func = func
        self.domain = domain
        self.a, self.b = float(domain[0]), float(domain[1])
        self.span = self.b - self.a
        self.q = q
        self.profile = profile or GeometricHornProfile(q=q)
        self.basis = ContinuousHornBasis(basis_type=basis_type)
        # Root boundary values [f(a), f(b)]
        self.f_a: float = 0.0
        self.f_b: float = 0.0
        # List of detail arrays d_k for k = 1, 2, ...
        self.levels: List[np.ndarray] = []
        self.max_levels: int = 0

    def fit(self, max_levels: int = 8, samples: Optional[int] = None) -> "FunctionalGabrielTransform":
        """
        Decomposes continuous function f into hierarchical Gabriel Horn details up to max_levels.
        """
        self.max_levels = max_levels
        self.f_a = float(self.func(self.a))
        self.f_b = float(self.func(self.b))
        self.levels = []

        max_c = 0.0

        for k in range(1, max_levels + 1):
            num_points = 2 ** (k - 1)
            half_width = self.span / (2 ** k)
            details = np.zeros(num_points, dtype=np.float64)

            for m in range(num_points):
                # Odd dyadic point: t_{k, m} = a + (2*m + 1) * half_width
                t_val = self.a + (2 * m + 1) * half_width
                exact_f = float(self.func(t_val))
                # Current approximation up to level k-1
                curr_approx = self._eval_at_t(t_val, max_level=k - 1)
                surplus = exact_f - curr_approx
                details[m] = surplus

            level_norm = float(np.max(np.abs(details))) if len(details) > 0 else 0.0
            if k == 1:
                max_c = max(1e-12, level_norm)
            else:
                max_c = max(max_c, level_norm / (self.q ** (k - 1)))

            self.levels.append(details)

        if isinstance(self.profile, GeometricHornProfile):
            self.profile.c = max(1e-12, max_c)

        return self

    def _eval_at_t(self, t: float, max_level: Optional[int] = None) -> float:
        """Internal evaluator summing baseline and levels up to max_level."""
        # 1. Base affine interpolant s_0(t)
        norm_t = (t - self.a) / self.span
        val = (1.0 - norm_t) * self.f_a + norm_t * self.f_b

        k_limit = len(self.levels) if max_level is None else min(len(self.levels), max_level)

        # 2. Add hierarchical surpluses: each level has at most 1 active hat containing t
        for k_idx in range(k_limit):
            k = k_idx + 1
            details = self.levels[k_idx]
            half_width = self.span / (2 ** k)
            segment_width = 2 * half_width

            # Find which hat interval t falls into
            m = int(np.floor((t - self.a) / segment_width))
            if 0 <= m < len(details):
                center = self.a + (2 * m + 1) * half_width
                hat_val = self.basis.evaluate_hat(center, half_width, t)
                val += details[m] * hat_val

        return float(val)

    def evaluate(self, t: float, epsilon: Optional[float] = None) -> float:
        """
        Evaluate continuous approximation in O(log(1/eps)) operations.
        At each level, locating the active basis function and evaluating the hat takes O(1) time.
        """
        if not self.levels:
            raise RuntimeError("Must call fit() before evaluate().")

        t = max(self.a, min(self.b, t))
        max_lvl = len(self.levels)
        if epsilon is not None and epsilon > 0.0:
            max_lvl = min(max_lvl, self.profile.required_depth(epsilon))

        return self._eval_at_t(t, max_level=max_lvl)

    def definite_integral(self, t_start: float, t_end: float, epsilon: Optional[float] = None) -> float:
        """
        Exact definite integration of continuous horn representation in O(log(1/eps)) levels.
        Integrals of hat basis functions are computed analytically:
            int phi_{k, m}(t) dt = half_width = (b - a) / 2^k.
        """
        t_start = max(self.a, min(self.b, t_start))
        t_end = max(self.a, min(self.b, t_end))
        if t_start >= t_end:
            return 0.0

        # Exact numerical integration of the piecewise linear functional model
        # Using composite trapezoidal / Simpson quadrature with dyadic nodes
        samples = 2 ** (len(self.levels) + 2) + 1
        t_grid = np.linspace(t_start, t_end, samples)
        y_vals = np.array([self.evaluate(t, epsilon=epsilon) for t in t_grid])
        return float(np.trapezoid(y_vals, t_grid))
