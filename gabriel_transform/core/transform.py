from typing import Optional, Union
import numpy as np
from gabriel_transform.core.profiles import HornProfile, GeometricHornProfile
from gabriel_transform.core.representation import GabrielHornRepresentation


class GabrielTransform:
    def __init__(
        self,
        q: float = 0.5,
        max_levels: Optional[int] = None,
        profile: Optional[HornProfile] = None,
        scale_ratio: int = 2,
    ):
        self.q = q
        self.profile = profile or GeometricHornProfile(q=q)
        self.max_levels = max_levels
        self.scale_ratio = scale_ratio

    def forward(
        self,
        x: Union[np.ndarray, list],
        epsilon: Optional[float] = None,
    ) -> GabrielHornRepresentation:
        arr = np.asarray(x, dtype=np.float64).flatten()
        n = len(arr)
        if n == 0:
            raise ValueError("Input signal cannot be empty.")

        max_possible_levels = int(np.floor(np.log(max(2, n)) / np.log(self.scale_ratio)))
        k_limit = self.max_levels if self.max_levels is not None else max_possible_levels
        k_limit = max(1, min(k_limit, max_possible_levels))

        coarse_len = max(1, n // (self.scale_ratio ** k_limit))
        coarse_baseline = self._downsample(arr, coarse_len)

        approx = GabrielHornRepresentation._upsample_to_length(coarse_baseline, n)
        residual = arr - approx

        levels = []
        c_estimated = float(np.linalg.norm(residual))
        if isinstance(self.profile, GeometricHornProfile):
            self.profile.c = max(1e-12, c_estimated)

        current_res = residual.copy()

        for k in range(k_limit):
            current_norm = float(np.linalg.norm(current_res))
            if epsilon is not None and current_norm <= epsilon:
                break

            level_len = min(n, coarse_len * (self.scale_ratio ** (k + 1)))
            coarse_res = self._downsample(current_res, level_len)

            allowed_envelope = self.profile.envelope(k)
            detail_norm = float(np.linalg.norm(coarse_res))

            if detail_norm > allowed_envelope and allowed_envelope > 0:
                shrink_factor = allowed_envelope / detail_norm
                contracted_detail = coarse_res * shrink_factor
            else:
                contracted_detail = coarse_res

            levels.append(contracted_detail)
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
        return rep.reconstruct(epsilon=epsilon, max_level=max_level)

    @staticmethod
    def _downsample(arr: np.ndarray, target_length: int) -> np.ndarray:
        n = len(arr)
        if target_length >= n:
            return arr.copy()
        if target_length == 1:
            return np.array([np.mean(arr)], dtype=np.float64)

        splits = np.array_split(arr, target_length)
        return np.array([np.mean(chunk) for chunk in splits], dtype=np.float64)


def discrete_gabriel_transform(
    x: Union[np.ndarray, list],
    q: float = 0.5,
    epsilon: Optional[float] = None,
) -> GabrielHornRepresentation:
    gt = GabrielTransform(q=q)
    return gt.forward(x, epsilon=epsilon)


def inverse_gabriel_transform(
    rep: GabrielHornRepresentation,
    epsilon: Optional[float] = None,
) -> np.ndarray:
    gt = GabrielTransform()
    return gt.inverse(rep, epsilon=epsilon)
