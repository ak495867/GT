"""
Gabriel Transform (GT) - Hierarchical, progressively compressed multiscale transform
inspired by Gabriel's Horn (Torricelli's Trumpet, 1644).
"""

from gabriel_transform.core import (
    HornProfile,
    GeometricHornProfile,
    TorricelliHornProfile,
    ExponentialHornProfile,
    GabrielHornRepresentation,
    GabrielTransform,
    discrete_gabriel_transform,
    inverse_gabriel_transform,
)
from gabriel_transform.tree import (
    AdaptiveGabrielHornTree,
    GabrielNode,
)
from gabriel_transform.functional import (
    FunctionalGabrielTransform,
    ContinuousHornBasis,
)
from gabriel_transform.operators import (
    gabriel_point_query,
    gabriel_range_integrate,
    gabriel_inner_product,
    gabriel_fast_convolve,
)
from gabriel_transform.theory import (
    GabrielComplexityModel,
    verify_geometric_decay,
    verify_epsilon_bound,
    solve_recurrence_complexity,
)

__version__ = "0.1.0"
__all__ = [
    "HornProfile",
    "GeometricHornProfile",
    "TorricelliHornProfile",
    "ExponentialHornProfile",
    "GabrielHornRepresentation",
    "GabrielTransform",
    "discrete_gabriel_transform",
    "inverse_gabriel_transform",
    "AdaptiveGabrielHornTree",
    "GabrielNode",
    "FunctionalGabrielTransform",
    "ContinuousHornBasis",
    "gabriel_point_query",
    "gabriel_range_integrate",
    "gabriel_inner_product",
    "gabriel_fast_convolve",
    "GabrielComplexityModel",
    "verify_geometric_decay",
    "verify_epsilon_bound",
    "solve_recurrence_complexity",
]
