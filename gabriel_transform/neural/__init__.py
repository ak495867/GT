from gabriel_transform.neural.attention import (
    gabriel_attention_numpy,
    HAS_TORCH,
)

if HAS_TORCH:
    from gabriel_transform.neural.attention import GabrielAttention
else:
    GabrielAttention = None

__all__ = [
    "gabriel_attention_numpy",
    "GabrielAttention",
    "HAS_TORCH",
]
