from typing import List, Optional, Tuple, Union
import numpy as np
from gabriel_transform.core.profiles import HornProfile, GeometricHornProfile


class StreamingHornBlock:
    def __init__(self, level: int, values: np.ndarray):
        self.level = level
        self.size = len(values)
        self.mean_val = float(np.mean(values))
        res = values - self.mean_val
        self.residual_energy = float(np.linalg.norm(res))
        self.raw_data = np.copy(values)


class StreamingGabrielTransform:
    def __init__(self, q: float = 0.5, profile: Optional[HornProfile] = None):
        self.q = q
        self.profile = profile or GeometricHornProfile(q=q)
        self.total_samples: int = 0
        self.slots: List[Optional[StreamingHornBlock]] = []
        self.active_levels_energy: List[float] = []

    def append(self, sample: float) -> int:
        val = float(sample)
        self.total_samples += 1

        carry_block = StreamingHornBlock(level=0, values=np.array([val]))
        k = 0
        merges = 0

        while True:
            if k >= len(self.slots):
                self.slots.append(None)

            if self.slots[k] is None:
                self.slots[k] = carry_block
                break
            else:
                merges += 1
                existing = self.slots[k]
                merged_values = np.concatenate([existing.raw_data, carry_block.raw_data])
                carry_block = StreamingHornBlock(level=k + 1, values=merged_values)
                self.slots[k] = None
                k += 1

        return merges

    def append_batch(self, samples: Union[np.ndarray, list]) -> int:
        total_merges = 0
        for s in samples:
            total_merges += self.append(s)
        return total_merges

    def query_recent(self, window_size: int, epsilon: float = 0.0) -> Tuple[float, int]:
        if self.total_samples == 0:
            return 0.0, 0

        target_count = min(window_size, self.total_samples)
        accum_sum = 0.0
        accum_count = 0
        blocks_accessed = 0

        for k in range(len(self.slots)):
            block = self.slots[k]
            if block is None:
                continue

            blocks_accessed += 1
            needed = target_count - accum_count
            if needed <= 0:
                break

            if block.size <= needed:
                accum_sum += block.mean_val * block.size
                accum_count += block.size
            else:
                suffix = block.raw_data[-needed:]
                accum_sum += float(np.sum(suffix))
                accum_count += needed
                break

            if block.residual_energy <= epsilon:
                break

        return (accum_sum / max(1, accum_count)), blocks_accessed

    def get_compressed_horn_summary(self, epsilon: float = 1e-3) -> dict:
        active_blocks = []
        for k, block in enumerate(self.slots):
            if block is not None:
                if block.residual_energy > epsilon or k < 2:
                    active_blocks.append({
                        "level": k,
                        "size": block.size,
                        "mean": block.mean_val,
                        "residual_energy": block.residual_energy,
                    })

        return {
            "total_samples": self.total_samples,
            "num_slots": len(self.slots),
            "active_horn_blocks": active_blocks,
            "compression_ratio": self.total_samples / max(1, len(active_blocks)),
        }
