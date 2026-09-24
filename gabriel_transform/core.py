"""
Core implementation of the Gabriel Transform (GT).

Formalized from the Gabriel's Horn (Torricelli's Trumpet) concept:
A mathematical decomposition into progressive multiscale levels:
    X = X_0 + X_1 + X_2 + ... + X_{K-1} + R_K
where the energy/amplitude contribution of each level decreases geometrically:
    ||X_k|| <= C * q^k,  0 < q < 1
enabling O(log(1/eps)) approximation complexity and O(log n) hierarchical queries.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import numpy as np


class HornProfile(ABC):
    """
    Abstract base class for horn geometry profiles.
    Defines how energy/amplitude narrows as we traverse deeper down the horn.
    """

    @abstractmethod
    def envelope(self, k: int) -> float:
        """Maximum allowed norm/amplitude at level k: w(k)."""
        pass

    @abstractmethod
    def cumulative_tail(self, k: int) -> float:
        """Upper bound on residual error if truncated at level k: sum_{j=k}^inf w(j)."""
        pass

    @abstractmethod
    def required_depth(self, epsilon: float) -> int:
        """Smallest integer K such that cumulative_tail(K) <= epsilon."""
        pass


class GeometricHornProfile(HornProfile):
    """
    Geometric / Exponential Horn Profile:
        w(k) = C * q^k,   0 < q < 1, C > 0
    Yields exponential convergence:
        Tail(k) = C * q^k / (1 - q)
        Required K(eps) = ceil( log(C / (eps * (1 - q))) / log(1/q) ) = O(log(1/eps)).
    """

    def __init__(self, q: float = 0.5, c: float = 1.0):
        if not (0.0 < q < 1.0):
            raise ValueError(f"Contraction factor q must be in (0, 1), got {q}")
        if c <= 0.0:
            raise ValueError(f"Amplitude scale C must be positive, got {c}")
        self.q = float(q)
        self.c = float(c)

    def envelope(self, k: int) -> float:
        if k < 0:
            return 0.0
        return self.c * (self.q ** k)

    def cumulative_tail(self, k: int) -> float:
        if k < 0:
            return self.c / (1.0 - self.q)
        return (self.c * (self.q ** k)) / (1.0 - self.q)

    def required_depth(self, epsilon: float) -> int:
        if epsilon <= 0.0:
            return 64  # Numerical machine limit for doubles
        total_tail = self.c / (1.0 - self.q)
        if epsilon >= total_tail:
            return 0
        ratio = (epsilon * (1.0 - self.q)) / self.c
        k = int(np.ceil(np.log(ratio) / np.log(self.q)))
        return max(0, k)

    def __repr__(self) -> str:
        return f"GeometricHornProfile(q={self.q}, c={self.c})"


class TorricelliHornProfile(HornProfile):
    """
    Classical Torricelli Horn Profile (Evangelista Torricelli, 1644):
        w(k) = C / (1 + alpha * k)^gamma,  gamma > 1
    Harmonic/algebraic decay directly mirroring Gabriel's horn cross-sectional radius:
        Tail(k) ~ C / (alpha * (gamma - 1) * (1 + alpha * k)^(gamma - 1))
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, c: float = 1.0):
        if alpha <= 0.0:
            raise ValueError(f"Alpha must be positive, got {alpha}")
        if gamma <= 1.0:
            raise ValueError(f"Gamma must be > 1.0 for finite total energy (finite horn volume), got {gamma}")
        if c <= 0.0:
            raise ValueError(f"Amplitude scale C must be positive, got {c}")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.c = float(c)

    def envelope(self, k: int) -> float:
        if k < 0:
            return 0.0
        return self.c / ((1.0 + self.alpha * k) ** self.gamma)

    def cumulative_tail(self, k: int) -> float:
        if k < 0:
            k = 0
        # Continuous integral approximation: int_k^inf C / (1 + alpha*t)^gamma dt
        denom = self.alpha * (self.gamma - 1.0) * ((1.0 + self.alpha * k) ** (self.gamma - 1.0))
        return self.c / denom

    def required_depth(self, epsilon: float) -> int:
        if epsilon <= 0.0:
            return 1000
        # eps >= C / (alpha*(gamma-1)*(1+alpha*k)^(gamma-1))
        # (1 + alpha*k)^(gamma-1) >= C / (eps * alpha * (gamma-1))
        target = self.c / (epsilon * self.alpha * (self.gamma - 1.0))
        if target <= 1.0:
            return 0
        base = target ** (1.0 / (self.gamma - 1.0))
        k = int(np.ceil((base - 1.0) / self.alpha))
        return max(0, k)

    def __repr__(self) -> str:
        return f"TorricelliHornProfile(alpha={self.alpha}, gamma={self.gamma}, c={self.c})"


class ExponentialHornProfile(GeometricHornProfile):
    """
    Exponential Horn Profile parameterized by continuous decay rate lambda > 0:
        w(k) = C * exp(-lambda * k)
    Identical to GeometricHornProfile with q = exp(-lambda).
    """

    def __init__(self, decay_rate: float = 0.693147, c: float = 1.0):
        if decay_rate <= 0.0:
            raise ValueError(f"Decay rate lambda must be positive, got {decay_rate}")
        self.decay_rate = float(decay_rate)
        q = float(np.exp(-decay_rate))
        super().__init__(q=q, c=c)

    def __repr__(self) -> str:
        return f"ExponentialHornProfile(decay_rate={self.decay_rate}, c={self.c})"


class GabrielHornRepresentation:
    """
    Data structure representing a signal decomposed into Gabriel's Horn levels.
    Levels H_0, H_1, ..., H_{K-1}, where each level represents a narrower horn slice.
    """

    def __init__(
        self,
        coarse_baseline: np.ndarray,
        levels: List[np.ndarray],
        profile: HornProfile,
        original_length: int,
        residual_norm: float = 0.0,
    ):
        self.coarse_baseline = np.asarray(coarse_baseline, dtype=np.float64)
        self.levels = [np.asarray(h, dtype=np.float64) for h in levels]
        self.profile = profile
        self.original_length = int(original_length)
        self.residual_norm = float(residual_norm)
        self.level_norms = [float(np.linalg.norm(h)) for h in self.levels]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def total_stored_coefficients(self) -> int:
        """Total scalar coefficients across coarse baseline and all levels."""
        return len(self.coarse_baseline) + sum(len(lvl) for lvl in self.levels)

    def compression_ratio(self) -> float:
        """Ratio of original elements to non-zero or stored coefficients."""
        stored = self.total_stored_coefficients()
        return float(self.original_length) / max(1, stored)

    def reconstruct(
        self,
        epsilon: Optional[float] = None,
        max_level: Optional[int] = None,
    ) -> np.ndarray:
        """
        Reconstruct signal up to accuracy epsilon or level max_level.
        If both are specified, the stricter (fewer levels) is taken.
        """
        k_eps = self.num_levels
        if epsilon is not None and epsilon > 0.0:
            k_eps = min(k_eps, self.profile.required_depth(epsilon))
        if max_level is not None:
            k_eps = min(k_eps, max_level)

        k_eps = max(0, min(k_eps, self.num_levels))

        # Start from coarse baseline upsampled to original length
        reconstructed = self._upsample_to_length(self.coarse_baseline, self.original_length)

        # Progressively add details up to level k_eps
        for k in range(k_eps):
            lvl = self.levels[k]
            detail = self._upsample_to_length(lvl, self.original_length)
            reconstructed += detail

        return reconstructed

    @staticmethod
    def _upsample_to_length(arr: np.ndarray, target_length: int) -> np.ndarray:
        """Upsample / interpolate 1D array to target_length using piecewise constant / linear."""
        m = len(arr)
        if m == target_length:
            return arr.copy()
        if m == 1:
            return np.full(target_length, arr[0], dtype=np.float64)
        # Use linear interpolation over [0, 1]
        x_old = np.linspace(0.0, 1.0, m, endpoint=True)
        x_new = np.linspace(0.0, 1.0, target_length, endpoint=True)
        return np.interp(x_new, x_old, arr)

    def prune_to_tolerance(self, epsilon: float) -> "GabrielHornRepresentation":
        """
        Prune levels whose cumulative energy is guaranteed to be <= epsilon,
        achieving optimal sparse representation.
        """
        req_k = self.profile.required_depth(epsilon)
        pruned_levels = self.levels[:req_k]
        rem_norm = self.profile.cumulative_tail(len(pruned_levels))
        return GabrielHornRepresentation(
            coarse_baseline=self.coarse_baseline.copy(),
            levels=pruned_levels,
            profile=self.profile,
            original_length=self.original_length,
            residual_norm=rem_norm,
        )

    def __repr__(self) -> str:
        return (
            f"GabrielHornRepresentation(length={self.original_length}, "
            f"levels={self.num_levels}, stored_coeffs={self.total_stored_coefficients()}, "
            f"profile={self.profile})"
        )


class GabrielTransform:
    """
    The Gabriel Transform engine.
    Decomposes an n-dimensional signal or discrete sequence into hierarchical levels
    satisfying geometric/Torricelli envelope bounds.
    """

    def __init__(
        self,
        q: float = 0.5,
        max_levels: Optional[int] = None,
        profile: Optional[HornProfile] = None,
        scale_ratio: int = 2,
    ):
        """
        Args:
            q: Geometric contraction factor (0 < q < 1).
            max_levels: Optional hard upper bound on levels. Defaults to floor(log2(n)).
            profile: Custom HornProfile. If None, GeometricHornProfile(q) is used.
            scale_ratio: Subsampling ratio r per level (default r=2 for dyadic tree).
        """
        self.q = q
        self.profile = profile or GeometricHornProfile(q=q)
        self.max_levels = max_levels
        self.scale_ratio = scale_ratio

    def forward(
        self,
        x: Union[np.ndarray, list],
        epsilon: Optional[float] = None,
    ) -> GabrielHornRepresentation:
        """
        Forward Gabriel Transform (FGT).
        Decomposes signal x into hierarchical Gabriel Horn levels:
            x = X_0 + X_1 + ... + X_{K-1} + R_K
        guaranteeing ||X_k|| <= C * q^k.

        Args:
            x: Input 1D signal.
            epsilon: Optional error tolerance. If given, stops once residual is <= epsilon.
        """
        arr = np.asarray(x, dtype=np.float64).flatten()
        n = len(arr)
        if n == 0:
            raise ValueError("Input signal cannot be empty.")

        # Determine total possible multiscale levels: floor(log_r(n))
        max_possible_levels = int(np.floor(np.log(max(2, n)) / np.log(self.scale_ratio)))
        k_limit = self.max_levels if self.max_levels is not None else max_possible_levels
        k_limit = max(1, min(k_limit, max_possible_levels))

        # Base coarse component (level 0: macro average / root of horn)
        # We start with the mean or coarse subsample
        coarse_len = max(1, n // (self.scale_ratio ** k_limit))
        coarse_baseline = self._downsample(arr, coarse_len)

        # Reconstruction accumulator
        approx = GabrielHornRepresentation._upsample_to_length(coarse_baseline, n)
        residual = arr - approx

        levels = []
        c_estimated = float(np.linalg.norm(residual))
        if isinstance(self.profile, GeometricHornProfile):
            # Calibrate profile constant C to initial residual energy
            self.profile.c = max(1e-12, c_estimated)

        current_res = residual.copy()

        for k in range(k_limit):
            # Check stopping tolerance if provided
            current_norm = float(np.linalg.norm(current_res))
            if epsilon is not None and current_norm <= epsilon:
                break

            # Current scale resolution
            level_len = min(n, coarse_len * (self.scale_ratio ** (k + 1)))

            # Project current residual onto level resolution
            coarse_res = self._downsample(current_res, level_len)

            # Contractive soft-thresholding / funnel weighting:
            # We enforce that the level detail adheres to the horn envelope: ||H_k|| <= w(k)
            allowed_envelope = self.profile.envelope(k)
            detail_norm = float(np.linalg.norm(coarse_res))

            if detail_norm > allowed_envelope and allowed_envelope > 0:
                # Shrink / contract to horn boundary
                shrink_factor = allowed_envelope / detail_norm
                contracted_detail = coarse_res * shrink_factor
            else:
                contracted_detail = coarse_res

            levels.append(contracted_detail)

            # Subtract projected detail from residual
            up_detail = GabrielHornRepresentation._upsample_to_length(contracted_detail, n)
            current_res = current_res - up_detail

        final_residual_norm = float(np.linalg.norm(current_res))

        return GabrielHornRepresentation(
            coarse_baseline=coarse_baseline,
            levels=levels,
            profile=self.profile,
            original_length=n,
            residual_norm=final_residual_norm,
        )

    def inverse(
        self,
        rep: GabrielHornRepresentation,
        epsilon: Optional[float] = None,
        max_level: Optional[int] = None,
    ) -> np.ndarray:
        """
        Inverse Gabriel Transform (IGT).
        Reconstructs the original signal from horn coefficients.
        """
        return rep.reconstruct(epsilon=epsilon, max_level=max_level)

    @staticmethod
    def _downsample(arr: np.ndarray, target_length: int) -> np.ndarray:
        """Downsample 1D array to target_length using local block averaging."""
        n = len(arr)
        if target_length >= n:
            return arr.copy()
        if target_length == 1:
            return np.array([np.mean(arr)], dtype=np.float64)

        # Block-average downsampling
        splits = np.array_split(arr, target_length)
        return np.array([np.mean(chunk) for chunk in splits], dtype=np.float64)


def discrete_gabriel_transform(
    x: Union[np.ndarray, list],
    q: float = 0.5,
    epsilon: Optional[float] = None,
) -> GabrielHornRepresentation:
    """Convenience function for forward Gabriel Transform."""
    gt = GabrielTransform(q=q)
    return gt.forward(x, epsilon=epsilon)


def inverse_gabriel_transform(
    rep: GabrielHornRepresentation,
    epsilon: Optional[float] = None,
) -> np.ndarray:
    """Convenience function for inverse Gabriel Transform."""
    gt = GabrielTransform()
    return gt.inverse(rep, epsilon=epsilon)
