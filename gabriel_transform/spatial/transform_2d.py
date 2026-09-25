from typing import List, Optional, Tuple
import numpy as np
from gabriel_transform.core.profiles import HornProfile, GeometricHornProfile


class Gabriel2DRepresentation:
    def __init__(
        self,
        coarse_base: np.ndarray,
        levels: List[np.ndarray],
        profile: HornProfile,
        shape: Tuple[int, int],
        residual_norm: float = 0.0,
    ):
        self.coarse_base = np.asarray(coarse_base, dtype=np.float64)
        self.levels = [np.asarray(d, dtype=np.float64) for d in levels]
        self.profile = profile
        self.shape = shape
        self.residual_norm = residual_norm
        self.level_norms = [float(np.linalg.norm(d)) for d in self.levels]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def total_stored_coefficients(self) -> int:
        return self.coarse_base.size + sum(d.size for d in self.levels)

    def compression_ratio(self) -> float:
        orig_size = self.shape[0] * self.shape[1]
        return float(orig_size) / max(1, self.total_stored_coefficients())

    def reconstruct(self, epsilon: Optional[float] = None, max_level: Optional[int] = None) -> np.ndarray:
        k_limit = self.num_levels
        if epsilon is not None and epsilon > 0.0:
            k_limit = min(k_limit, self.profile.required_depth(epsilon))
        if max_level is not None:
            k_limit = min(k_limit, max_level)

        rec = self._upsample_2d(self.coarse_base, self.shape)
        for k in range(min(k_limit, self.num_levels)):
            rec += self._upsample_2d(self.levels[k], self.shape)

        return rec

    def query_pixel(self, r: int, c: int, epsilon: Optional[float] = None) -> Tuple[float, int]:
        H, W = self.shape
        if not (0 <= r < H and 0 <= c < W):
            raise IndexError(f"Pixel ({r}, {c}) out of bounds for shape {self.shape}")

        norm_r = float(r) / max(1, H - 1)
        norm_c = float(c) / max(1, W - 1)

        val = self._sample_bilinear(self.coarse_base, norm_r, norm_c)

        k_limit = self.num_levels
        if epsilon is not None and epsilon > 0.0:
            k_limit = min(k_limit, self.profile.required_depth(epsilon))

        accessed = 0
        for k in range(k_limit):
            accessed += 1
            val += self._sample_bilinear(self.levels[k], norm_r, norm_c)

        return float(val), accessed

    @staticmethod
    def _sample_bilinear(arr: np.ndarray, norm_r: float, norm_c: float) -> float:
        H, W = arr.shape
        if H == 1 and W == 1:
            return float(arr[0, 0])

        r_pos = norm_r * (H - 1)
        c_pos = norm_c * (W - 1)

        r0 = int(np.floor(r_pos))
        c0 = int(np.floor(c_pos))
        r1 = min(H - 1, r0 + 1)
        c1 = min(W - 1, c0 + 1)

        dr = r_pos - r0
        dc = c_pos - c0

        v00 = arr[r0, c0]
        v01 = arr[r0, c1]
        v10 = arr[r1, c0]
        v11 = arr[r1, c1]

        val = (
            (1.0 - dr) * (1.0 - dc) * v00
            + (1.0 - dr) * dc * v01
            + dr * (1.0 - dc) * v10
            + dr * dc * v11
        )
        return float(val)

    @staticmethod
    def _upsample_2d(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        orig_h, orig_w = arr.shape
        target_h, target_w = target_shape
        if (orig_h, orig_w) == (target_h, target_w):
            return arr.copy()

        r_grid = np.linspace(0.0, 1.0, target_h)
        c_grid = np.linspace(0.0, 1.0, target_w)

        orig_r = np.linspace(0.0, 1.0, orig_h)
        orig_c = np.linspace(0.0, 1.0, orig_w)

        interp_cols = np.empty((orig_h, target_w), dtype=np.float64)
        for i in range(orig_h):
            interp_cols[i] = np.interp(c_grid, orig_c, arr[i])

        out = np.empty((target_h, target_w), dtype=np.float64)
        for j in range(target_w):
            out[:, j] = np.interp(r_grid, orig_r, interp_cols[:, j])

        return out

    def prune_to_tolerance(self, epsilon: float) -> "Gabriel2DRepresentation":
        req_k = self.profile.required_depth(epsilon)
        pruned_levels = self.levels[:req_k]
        rem_norm = self.profile.cumulative_tail(len(pruned_levels))
        return Gabriel2DRepresentation(
            coarse_base=self.coarse_base.copy(),
            levels=pruned_levels,
            profile=self.profile,
            shape=self.shape,
            residual_norm=rem_norm,
        )


class GabrielTransform2D:
    def __init__(
        self,
        q: float = 0.5,
        max_levels: Optional[int] = None,
        profile: Optional[HornProfile] = None,
    ):
        self.q = q
        self.profile = profile or GeometricHornProfile(q=q)
        self.max_levels = max_levels

    def forward(self, img: np.ndarray, epsilon: Optional[float] = None) -> Gabriel2DRepresentation:
        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Input must be a 2D array, got shape {arr.shape}")

        H, W = arr.shape
        min_dim = min(H, W)
        max_possible_levels = int(np.floor(np.log2(max(2, min_dim))))
        k_limit = self.max_levels if self.max_levels is not None else max_possible_levels
        k_limit = max(1, min(k_limit, max_possible_levels))

        base_h = max(1, H // (2 ** k_limit))
        base_w = max(1, W // (2 ** k_limit))
        coarse_base = self._downsample_2d(arr, (base_h, base_w))

        current_approx = Gabriel2DRepresentation._upsample_2d(coarse_base, (H, W))
        current_res = arr - current_approx

        init_energy = float(np.linalg.norm(current_res))
        if isinstance(self.profile, GeometricHornProfile):
            self.profile.c = max(1e-12, init_energy)

        levels = []

        for k in range(k_limit):
            curr_norm = float(np.linalg.norm(current_res))
            if epsilon is not None and curr_norm <= epsilon:
                break

            target_h = min(H, base_h * (2 ** (k + 1)))
            target_w = min(W, base_w * (2 ** (k + 1)))

            detail = self._downsample_2d(current_res, (target_h, target_w))
            detail_norm = float(np.linalg.norm(detail))
            allowed_bound = self.profile.envelope(k)

            if detail_norm > allowed_bound and allowed_bound > 0:
                detail *= (allowed_bound / detail_norm)

            levels.append(detail)
            current_res -= Gabriel2DRepresentation._upsample_2d(detail, (H, W))

        rem_norm = float(np.linalg.norm(current_res))

        return Gabriel2DRepresentation(
            coarse_base=coarse_base,
            levels=levels,
            profile=self.profile,
            shape=(H, W),
            residual_norm=rem_norm,
        )

    def inverse(
        self,
        rep: Gabriel2DRepresentation,
        epsilon: Optional[float] = None,
        max_level: Optional[int] = None,
    ) -> np.ndarray:
        return rep.reconstruct(epsilon=epsilon, max_level=max_level)

    @staticmethod
    def _downsample_2d(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        H, W = arr.shape
        th, tw = target_shape
        if (H, W) == (th, tw):
            return arr.copy()

        row_chunks = np.array_split(arr, th, axis=0)
        out = np.empty((th, tw), dtype=np.float64)
        for r_idx, r_chunk in enumerate(row_chunks):
            col_chunks = np.array_split(r_chunk, tw, axis=1)
            for c_idx, c_chunk in enumerate(col_chunks):
                out[r_idx, c_idx] = np.mean(c_chunk)

        return out
