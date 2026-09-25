from typing import Optional, Tuple
import numpy as np
from gabriel_transform.core.representation import GabrielHornRepresentation


def _interpolate_level_at(arr: np.ndarray, norm_coord: float) -> float:
    m = len(arr)
    if m == 1:
        return float(arr[0])
    pos = norm_coord * (m - 1)
    i0 = int(np.floor(pos))
    i1 = min(m - 1, i0 + 1)
    frac = pos - i0
    return float((1.0 - frac) * arr[i0] + frac * arr[i1])


def gabriel_point_query(
    rep: GabrielHornRepresentation,
    index: int,
    epsilon: Optional[float] = None,
) -> Tuple[float, int]:
    if index < 0 or index >= rep.original_length:
        raise IndexError(f"Index {index} out of range [0, {rep.original_length})")

    k_max = rep.num_levels
    if epsilon is not None and epsilon > 0.0:
        k_max = min(k_max, rep.profile.required_depth(epsilon))

    norm_coord = float(index) / max(1, rep.original_length - 1)
    val = _interpolate_level_at(rep.coarse_baseline, norm_coord)

    levels_accessed = 0
    for k in range(k_max):
        levels_accessed += 1
        lvl = rep.levels[k]
        val += _interpolate_level_at(lvl, norm_coord)

    return float(val), levels_accessed


def gabriel_range_integrate(
    rep: GabrielHornRepresentation,
    start: int,
    end: int,
    epsilon: Optional[float] = None,
) -> Tuple[float, int]:
    start = max(0, start)
    end = min(rep.original_length, end)
    if start >= end:
        return 0.0, 0

    k_max = rep.num_levels
    if epsilon is not None and epsilon > 0.0:
        k_max = min(k_max, rep.profile.required_depth(epsilon))

    base_interp = rep._upsample_to_length(rep.coarse_baseline, rep.original_length)
    total = float(np.sum(base_interp[start:end]))

    levels_accessed = 0
    for k in range(k_max):
        levels_accessed += 1
        lvl_interp = rep._upsample_to_length(rep.levels[k], rep.original_length)
        total += float(np.sum(lvl_interp[start:end]))

    return total, levels_accessed


def gabriel_inner_product(
    rep1: GabrielHornRepresentation,
    rep2: GabrielHornRepresentation,
    epsilon: Optional[float] = None,
) -> Tuple[float, int]:
    if rep1.original_length != rep2.original_length:
        raise ValueError("Representations must have matching lengths.")

    k_max = min(rep1.num_levels, rep2.num_levels)
    if epsilon is not None and epsilon > 0.0:
        k_req1 = rep1.profile.required_depth(np.sqrt(epsilon))
        k_req2 = rep2.profile.required_depth(np.sqrt(epsilon))
        k_max = min(k_max, max(k_req1, k_req2))

    n = rep1.original_length
    b1 = rep1._upsample_to_length(rep1.coarse_baseline, n)
    b2 = rep2._upsample_to_length(rep2.coarse_baseline, n)
    prod = float(np.dot(b1, b2))

    levels_computed = 0
    for k in range(k_max):
        levels_computed += 1
        d1 = rep1._upsample_to_length(rep1.levels[k], n)
        d2 = rep2._upsample_to_length(rep2.levels[k], n)
        prod += float(np.dot(d1, d2))

    return prod, levels_computed


def gabriel_fast_convolve(
    rep1: GabrielHornRepresentation,
    rep2: GabrielHornRepresentation,
    epsilon: float = 1e-4,
) -> np.ndarray:
    k_req1 = rep1.profile.required_depth(epsilon)
    k_req2 = rep2.profile.required_depth(epsilon)

    x1 = rep1.reconstruct(max_level=k_req1)
    x2 = rep2.reconstruct(max_level=k_req2)

    return np.convolve(x1, x2, mode="same")
