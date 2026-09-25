from typing import List, Optional
import numpy as np
from gabriel_transform.core.profiles import HornProfile


class GabrielHornRepresentation:
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
        return len(self.coarse_baseline) + sum(len(lvl) for lvl in self.levels)

    def compression_ratio(self) -> float:
        stored = self.total_stored_coefficients()
        return float(self.original_length) / max(1, stored)

    def reconstruct(
        self,
        epsilon: Optional[float] = None,
        max_level: Optional[int] = None,
    ) -> np.ndarray:
        k_eps = self.num_levels
        if epsilon is not None and epsilon > 0.0:
            k_eps = min(k_eps, self.profile.required_depth(epsilon))
        if max_level is not None:
            k_eps = min(k_eps, max_level)

        k_eps = max(0, min(k_eps, self.num_levels))
        reconstructed = self._upsample_to_length(self.coarse_baseline, self.original_length)

        for k in range(k_eps):
            lvl = self.levels[k]
            detail = self._upsample_to_length(lvl, self.original_length)
            reconstructed += detail

        return reconstructed

    @staticmethod
    def _upsample_to_length(arr: np.ndarray, target_length: int) -> np.ndarray:
        m = len(arr)
        if m == target_length:
            return arr.copy()
        if m == 1:
            return np.full(target_length, arr[0], dtype=np.float64)
        x_old = np.linspace(0.0, 1.0, m, endpoint=True)
        x_new = np.linspace(0.0, 1.0, target_length, endpoint=True)
        return np.interp(x_new, x_old, arr)

    def prune_to_tolerance(self, epsilon: float) -> "GabrielHornRepresentation":
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
