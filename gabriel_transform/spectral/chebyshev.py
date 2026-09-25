from typing import Callable, Optional, Tuple
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev, chebval, chebfit
from gabriel_transform.core.profiles import HornProfile, GeometricHornProfile


class SpectralChebyshevHornTransform:
    def __init__(
        self,
        func: Callable[[float], float],
        domain: Tuple[float, float] = (-1.0, 1.0),
        q: float = 0.5,
        profile: Optional[HornProfile] = None,
    ):
        self.func = func
        self.a, self.b = domain
        self.q = q
        self.profile = profile or GeometricHornProfile(q=q)
        self.coeffs: np.ndarray = np.empty(0)
        self.degree: int = 0

    def fit(self, max_degree: int = 32, num_sample_points: int = 128) -> "SpectralChebyshevHornTransform":
        nodes = np.sort(np.cos(np.pi * np.linspace(0, 1, max(max_degree + 1, num_sample_points))))
        x_mapped = self.a + (nodes + 1.0) * 0.5 * (self.b - self.a)
        y_vals = np.array([self.func(x) for x in x_mapped], dtype=np.float64)

        raw_coeffs = chebfit(nodes, y_vals, deg=max_degree)

        c_estimated = float(np.max([abs(raw_coeffs[k]) / (self.q ** k) for k in range(min(4, len(raw_coeffs)))]))
        c_estimated = max(1e-12, c_estimated)

        if isinstance(self.profile, GeometricHornProfile):
            self.profile.c = c_estimated

        bounded_coeffs = np.copy(raw_coeffs)
        for k in range(len(bounded_coeffs)):
            bound = self.profile.envelope(k)
            if abs(bounded_coeffs[k]) > bound and bound > 0:
                bounded_coeffs[k] = np.sign(bounded_coeffs[k]) * bound

        self.coeffs = bounded_coeffs
        self.degree = len(self.coeffs) - 1
        return self

    def evaluate(self, x: float, epsilon: Optional[float] = None) -> Tuple[float, int]:
        if len(self.coeffs) == 0:
            raise RuntimeError("Must call fit() before evaluate().")

        xi = 2.0 * (x - self.a) / (self.b - self.a) - 1.0
        xi = max(-1.0, min(1.0, xi))

        k_max = len(self.coeffs)
        if epsilon is not None and epsilon > 0.0:
            k_max = min(k_max, self.profile.required_depth(epsilon) + 1)

        active_coeffs = self.coeffs[:k_max]
        val = chebval(xi, active_coeffs)
        return float(val), len(active_coeffs)

    def definite_integral(self, x_start: float, x_end: float, epsilon: Optional[float] = None) -> float:
        k_max = len(self.coeffs)
        if epsilon is not None and epsilon > 0.0:
            k_max = min(k_max, self.profile.required_depth(epsilon) + 1)

        active_cheb = Chebyshev(self.coeffs[:k_max], domain=[-1, 1])
        antideriv = active_cheb.integ()

        xi_start = 2.0 * (x_start - self.a) / (self.b - self.a) - 1.0
        xi_end = 2.0 * (x_end - self.a) / (self.b - self.a) - 1.0

        jac = 0.5 * (self.b - self.a)
        res = jac * (antideriv(xi_end) - antideriv(xi_start))
        return float(res)
