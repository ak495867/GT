from typing import Optional, Tuple
import numpy as np
from gabriel_transform.core.profiles import HornProfile, GeometricHornProfile


class EytzingerGabrielHornTree:
    def __init__(self, q: float = 0.5, profile: Optional[HornProfile] = None):
        self.q = q
        self.profile = profile or GeometricHornProfile(q=q)
        self.n: int = 0
        self.tree_size: int = 0
        self.max_depth: int = 0

        self.values = np.empty(0, dtype=np.float64)
        self.subtree_energies = np.empty(0, dtype=np.float64)
        self.starts = np.empty(0, dtype=np.int32)
        self.ends = np.empty(0, dtype=np.int32)
        self.depths = np.empty(0, dtype=np.int32)
        self.is_leaf = np.empty(0, dtype=bool)

    @classmethod
    def build(
        cls,
        arr: np.ndarray,
        q: float = 0.5,
        profile: Optional[HornProfile] = None,
    ) -> "EytzingerGabrielHornTree":
        raw_arr = np.asarray(arr, dtype=np.float64).flatten()
        orig_n = len(raw_arr)
        if orig_n == 0:
            raise ValueError("Input array cannot be empty")

        d = int(np.ceil(np.log2(max(2, orig_n))))
        n_padded = 1 << d

        if orig_n < n_padded:
            padded_arr = np.pad(raw_arr, (0, n_padded - orig_n), mode="edge")
        else:
            padded_arr = raw_arr

        tree = cls(q=q, profile=profile)
        tree.n = orig_n
        tree.max_depth = d
        tree.tree_size = (2 * n_padded) - 1

        tree.values = np.zeros(tree.tree_size, dtype=np.float64)
        tree.subtree_energies = np.zeros(tree.tree_size, dtype=np.float64)
        tree.starts = np.zeros(tree.tree_size, dtype=np.int32)
        tree.ends = np.zeros(tree.tree_size, dtype=np.int32)
        tree.depths = np.zeros(tree.tree_size, dtype=np.int32)
        tree.is_leaf = np.zeros(tree.tree_size, dtype=bool)

        leaf_start_idx = n_padded - 1
        for i in range(n_padded):
            node_idx = leaf_start_idx + i
            tree.starts[node_idx] = i
            tree.ends[node_idx] = i + 1
            tree.values[node_idx] = padded_arr[i]
            tree.subtree_energies[node_idx] = 0.0
            tree.depths[node_idx] = d
            tree.is_leaf[node_idx] = True

        for node_idx in reversed(range(leaf_start_idx)):
            left = (node_idx << 1) + 1
            right = (node_idx << 1) + 2

            s = tree.starts[left]
            e = tree.ends[right]
            tree.starts[node_idx] = s
            tree.ends[node_idx] = e
            tree.depths[node_idx] = tree.depths[left] - 1
            tree.is_leaf[node_idx] = False

            chunk = padded_arr[s:e]
            mean_val = float(np.mean(chunk))
            tree.values[node_idx] = mean_val

            res = chunk - mean_val
            tree.subtree_energies[node_idx] = float(np.linalg.norm(res))

        if isinstance(tree.profile, GeometricHornProfile):
            tree.profile.c = max(1e-12, tree.subtree_energies[0])

        return tree

    def point_query(self, index: int, epsilon: float = 0.0) -> Tuple[float, int]:
        if index < 0 or index >= self.n:
            raise IndexError(f"Index {index} out of range [0, {self.n})")

        curr = 0
        steps = 0
        val = self.values[0]

        while curr < self.tree_size:
            steps += 1
            val = self.values[curr]

            if self.subtree_energies[curr] <= epsilon or self.is_leaf[curr]:
                break

            mid = (self.starts[curr] + self.ends[curr]) >> 1
            if index < mid:
                curr = (curr << 1) + 1
            else:
                curr = (curr << 1) + 2

        return float(val), steps

    def range_sum_query(self, start: int, end: int, epsilon: float = 0.0) -> Tuple[float, int]:
        start = max(0, start)
        end = min(self.n, end)
        if start >= end:
            return 0.0, 0

        steps = 0
        total = 0.0
        stack = [(0, start, end, epsilon)]

        while stack:
            curr, q_start, q_end, rem_eps = stack.pop()
            if curr >= self.tree_size or q_start >= q_end:
                continue

            steps += 1
            node_s = self.starts[curr]
            node_e = self.ends[curr]
            node_len = node_e - node_s

            if node_s == q_start and node_e == q_end:
                total += self.values[curr] * node_len
                continue

            if self.subtree_energies[curr] * (q_end - q_start) <= rem_eps or self.is_leaf[curr]:
                total += self.values[curr] * (q_end - q_start)
                continue

            mid = (node_s + node_e) >> 1
            left = (curr << 1) + 1
            right = (curr << 1) + 2

            if q_end > mid:
                stack.append((right, max(mid, q_start), q_end, rem_eps * 0.5))
            if q_start < mid:
                stack.append((left, q_start, min(mid, q_end), rem_eps * 0.5))

        return float(total), steps

    def batch_point_query(self, indices: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        idx_arr = np.asarray(indices, dtype=np.int32)
        M = len(idx_arr)
        out = np.empty(M, dtype=np.float64)

        tree_size = self.tree_size
        values = self.values
        energies = self.subtree_energies
        starts = self.starts
        ends = self.ends
        is_leaf = self.is_leaf

        for q_idx in range(M):
            idx = idx_arr[q_idx]
            curr = 0
            val = values[0]

            while curr < tree_size:
                val = values[curr]
                if energies[curr] <= epsilon or is_leaf[curr]:
                    break
                mid = (starts[curr] + ends[curr]) >> 1
                if idx < mid:
                    curr = (curr << 1) + 1
                else:
                    curr = (curr << 1) + 2

            out[q_idx] = val

        return out
