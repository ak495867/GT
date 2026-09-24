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
from gabriel_transform.eytzinger import (
    EytzingerGabrielHornTree,
)
from gabriel_transform.spatial_2d import (
    GabrielTransform2D,
    Gabriel2DRepresentation,
)
from gabriel_transform.attention import (
    gabriel_attention_numpy,
)
try:
    from gabriel_transform.attention import GabrielAttention
except ImportError:
    GabrielAttention = None

from gabriel_transform.spectral import (
    SpectralChebyshevHornTransform,
    SpectralLegendreHornTransform,
)
from gabriel_transform.streaming import (
    StreamingGabrielTransform,
    StreamingHornBlock,
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

__version__ = "0.2.0"
__all__ = [
    # Core
    "HornProfile",
    "GeometricHornProfile",
    "TorricelliHornProfile",
    "ExponentialHornProfile",
    "GabrielHornRepresentation",
    "GabrielTransform",
    "discrete_gabriel_transform",
    "inverse_gabriel_transform",
    # Tree & Eytzinger
    "AdaptiveGabrielHornTree",
    "GabrielNode",
    "EytzingerGabrielHornTree",
    # 2D & Spatial
    "GabrielTransform2D",
    "Gabriel2DRepresentation",
    # Attention
    "gabriel_attention_numpy",
    "GabrielAttention",
    # Spectral
    "SpectralChebyshevHornTransform",
    "SpectralLegendreHornTransform",
    # Streaming
    "StreamingGabrielTransform",
    "StreamingHornBlock",
    # Functional & Operators
    "FunctionalGabrielTransform",
    "ContinuousHornBasis",
    "gabriel_point_query",
    "gabriel_range_integrate",
    "gabriel_inner_product",
    "gabriel_fast_convolve",
    # Theory
    "GabrielComplexityModel",
    "verify_geometric_decay",
    "verify_epsilon_bound",
    "solve_recurrence_complexity",
]
