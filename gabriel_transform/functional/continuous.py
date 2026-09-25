from typing import Callable, List, Optional, Tuple
import numpy as np
from gabriel_transform.core.profiles import HornProfile, GeometricHornProfile


class ContinuousHornBasis:
    def __init__(self, basis_type: str = "spline"):
        self.basis_type = basis_type

    @staticmethod
    def evaluate_hat(center: float, half_width: float, t: float) -> float:
        diff = abs(t - center)
        if diff < half_width:
            return 1.0 - (diff / half_width)
        return 0.0

    def evaluate(self, k: int, m: int, t: float) -> float:
        scale = 2 ** k
        half_width = 1.0 / scale
        center = m / scale
        return self.evaluate_hat(center, half_width, t)


class FunctionalGabrielTransform:
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
        self.f_a: float = 0.0
        self.f_b: float = 0.0
        self.levels: List[np.ndarray] = []
        self.max_levels: int = 0

    def fit(self, max_levels: int = 8, samples: Optional[int] = None) -> "FunctionalGabrielTransform":
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
                t_val = self.a + (2 * m + 1) * half_width
                exact_f = float(self.func(t_val))
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
        norm_t = (t - self.a) / self.span
        val = (1.0 - norm_t) * self.f_a + norm_t * self.f_b

        k_limit = len(self.levels) if max_level is None else min(len(self.levels), max_level)

        for k_idx in range(k_limit):
            k = k_idx + 1
            details = self.levels[k_idx]
            half_width = self.span / (2 ** k)
            segment_width = 2 * half_width

            m = int(np.floor((t - self.a) / segment_width))
            if 0 <= m < len(details):
                center = self.a + (2 * m + 1) * half_width
                hat_val = self.basis.evaluate_hat(center, half_width, t)
                val += details[m] * hat_val

        return float(val)

    def evaluate(self, t: float, epsilon: Optional[float] = None) -> float:
        if not self.levels:
            raise RuntimeError("Must call fit() before evaluate().")

        t = max(self.a, min(self.b, t))
        max_lvl = len(self.levels)
        if epsilon is not None and epsilon > 0.0:
            max_lvl = min(max_lvl, self.profile.required_depth(epsilon))

        return self._eval_at_t(t, max_level=max_lvl)

    def definite_integral(self, t_start: float, t_end: float, epsilon: Optional[float] = None) -> float:
        t_start = max(self.a, min(self.b, t_start))
        t_end = max(self.a, min(self.b, t_end))
        if t_start >= t_end:
            return 0.0

        samples = 2 ** (len(self.levels) + 2) + 1
        t_grid = np.linspace(t_start, t_end, samples)
        y_vals = np.array([self.evaluate(t, epsilon=epsilon) for t in t_grid])
        return float(np.trapezoid(y_vals, t_grid))
