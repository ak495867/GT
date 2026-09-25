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
from gabriel_transform.trees import (
    AdaptiveGabrielHornTree,
    GabrielNode,
    EytzingerGabrielHornTree,
)
from gabriel_transform.spatial import (
    GabrielTransform2D,
    Gabriel2DRepresentation,
)
from gabriel_transform.neural import (
    gabriel_attention_numpy,
)
try:
    from gabriel_transform.neural import GabrielAttention
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
from gabriel_transform.physics import (
    GabrielMultipoleTree,
    GabrielMassCluster,
    direct_nbody_potential,
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
    "EytzingerGabrielHornTree",
    "GabrielTransform2D",
    "Gabriel2DRepresentation",
    "gabriel_attention_numpy",
    "GabrielAttention",
    "SpectralChebyshevHornTransform",
    "SpectralLegendreHornTransform",
    "StreamingGabrielTransform",
    "StreamingHornBlock",
    "GabrielMultipoleTree",
    "GabrielMassCluster",
    "direct_nbody_potential",
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
