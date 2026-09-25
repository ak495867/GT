from abc import ABC, abstractmethod
import numpy as np


class HornProfile(ABC):
    @abstractmethod
    def envelope(self, k: int) -> float:
        pass

    @abstractmethod
    def cumulative_tail(self, k: int) -> float:
        pass

    @abstractmethod
    def required_depth(self, epsilon: float) -> int:
        pass


class GeometricHornProfile(HornProfile):
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
            return 64
        total_tail = self.c / (1.0 - self.q)
        if epsilon >= total_tail:
            return 0
        ratio = (epsilon * (1.0 - self.q)) / self.c
        k = int(np.ceil(np.log(ratio) / np.log(self.q)))
        return max(0, k)

    def __repr__(self) -> str:
        return f"GeometricHornProfile(q={self.q}, c={self.c})"


class TorricelliHornProfile(HornProfile):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, c: float = 1.0):
        if alpha <= 0.0:
            raise ValueError(f"Alpha must be positive, got {alpha}")
        if gamma <= 1.0:
            raise ValueError(f"Gamma must be > 1.0, got {gamma}")
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
        denom = self.alpha * (self.gamma - 1.0) * ((1.0 + self.alpha * k) ** (self.gamma - 1.0))
        return self.c / denom

    def required_depth(self, epsilon: float) -> int:
        if epsilon <= 0.0:
            return 1000
        target = self.c / (epsilon * self.alpha * (self.gamma - 1.0))
        if target <= 1.0:
            return 0
        base = target ** (1.0 / (self.gamma - 1.0))
        k = int(np.ceil((base - 1.0) / self.alpha))
        return max(0, k)

    def __repr__(self) -> str:
        return f"TorricelliHornProfile(alpha={self.alpha}, gamma={self.gamma}, c={self.c})"


class ExponentialHornProfile(GeometricHornProfile):
    def __init__(self, decay_rate: float = 0.693147, c: float = 1.0):
        if decay_rate <= 0.0:
            raise ValueError(f"Decay rate lambda must be positive, got {decay_rate}")
        self.decay_rate = float(decay_rate)
        q = float(np.exp(-decay_rate))
        super().__init__(q=q, c=c)

    def __repr__(self) -> str:
        return f"ExponentialHornProfile(decay_rate={self.decay_rate}, c={self.c})"
