from typing import List, Optional, Tuple
import numpy as np
from gabriel_transform.core.profiles import HornProfile, GeometricHornProfile


class GabrielNode:
    def __init__(
        self,
        interval: Tuple[int, int],
        depth: int,
        value: float = 0.0,
        detail: float = 0.0,
        subtree_energy: float = 0.0,
    ):
        self.start, self.end = interval
        self.depth = depth
        self.value = value
        self.detail = detail
        self.subtree_energy = subtree_energy
        self.left: Optional["GabrielNode"] = None
        self.right: Optional["GabrielNode"] = None
        self.skip_links: List["GabrielNode"] = []

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class AdaptiveGabrielHornTree:
    def __init__(self, q: float = 0.5, c: float = 1.0, profile: Optional[HornProfile] = None):
        self.q = q
        self.profile = profile or GeometricHornProfile(q=q, c=c)
        self.root: Optional[GabrielNode] = None
        self.n: int = 0
        self.max_depth: int = 0

    @classmethod
    def build(
        cls,
        arr: np.ndarray,
        q: float = 0.5,
        profile: Optional[HornProfile] = None,
    ) -> "AdaptiveGabrielHornTree":
        arr = np.asarray(arr, dtype=np.float64).flatten()
        n = len(arr)
        if n == 0:
            raise ValueError("Array cannot be empty")

        tree = cls(q=q, profile=profile)
        tree.n = n

        def _build_node(start: int, end: int, depth: int, parent_val: float) -> GabrielNode:
            tree.max_depth = max(tree.max_depth, depth)
            length = end - start
            node_val = float(np.mean(arr[start:end]))
            detail = node_val - parent_val

            sub_res = arr[start:end] - node_val
            sub_energy = float(np.linalg.norm(sub_res))

            node = GabrielNode(
                interval=(start, end),
                depth=depth,
                value=node_val,
                detail=detail,
                subtree_energy=sub_energy,
            )

            if length > 1:
                mid = start + (length // 2)
                node.left = _build_node(start, mid, depth + 1, node_val)
                node.right = _build_node(mid, end, depth + 1, node_val)

            return node

        tree.root = _build_node(0, n, 0, 0.0)

        if isinstance(tree.profile, GeometricHornProfile):
            tree.profile.c = max(1e-12, tree.root.subtree_energy)

        tree._build_skip_links_postorder(tree.root)
        return tree

    def _build_skip_links_postorder(self, node: Optional[GabrielNode]):
        if node is None or node.is_leaf:
            return

        if node.left:
            self._build_skip_links_postorder(node.left)
        if node.right:
            self._build_skip_links_postorder(node.right)

        if node.left and node.right:
            heavy_child = node.left if node.left.subtree_energy >= node.right.subtree_energy else node.right
        else:
            heavy_child = node.left or node.right

        if heavy_child:
            node.skip_links.append(heavy_child)
            j = 1
            while True:
                prev_skip = node.skip_links[j - 1]
                if len(prev_skip.skip_links) >= j:
                    node.skip_links.append(prev_skip.skip_links[j - 1])
                    j += 1
                else:
                    break

    def point_query(self, index: int, epsilon: float = 0.0) -> Tuple[float, int]:
        if index < 0 or index >= self.n:
            raise IndexError(f"Index {index} out of bounds for size {self.n}")

        curr = self.root
        steps = 0
        val = 0.0

        while curr is not None:
            steps += 1
            val = curr.value

            if curr.subtree_energy <= epsilon or curr.is_leaf:
                break

            mid = curr.start + (curr.length // 2)
            if index < mid:
                curr = curr.left
            else:
                curr = curr.right

        return val, steps

    def range_sum_query(self, start: int, end: int, epsilon: float = 0.0) -> Tuple[float, int]:
        start = max(0, start)
        end = min(self.n, end)
        if start >= end:
            return 0.0, 0

        steps = 0

        def _query(node: Optional[GabrielNode], q_start: int, q_end: int, remaining_eps: float) -> float:
            nonlocal steps
            if node is None or q_start >= q_end:
                return 0.0

            steps += 1

            if node.start == q_start and node.end == q_end:
                return node.value * node.length

            if node.subtree_energy * (q_end - q_start) <= remaining_eps or node.is_leaf:
                return node.value * (q_end - q_start)

            mid = node.start + (node.length // 2)
            left_sum = 0.0
            right_sum = 0.0

            if q_start < mid:
                left_sum = _query(node.left, q_start, min(mid, q_end), remaining_eps / 2.0)
            if q_end > mid:
                right_sum = _query(node.right, max(mid, q_start), q_end, remaining_eps / 2.0)

            return left_sum + right_sum

        total = _query(self.root, start, end, epsilon)
        return total, steps

    def jump_to_scale(self, epsilon: float) -> Tuple[int, int]:
        if not self.root:
            return 0, 0

        curr = self.root
        steps = 0

        if curr.subtree_energy <= epsilon:
            return 0, 1

        while curr and curr.skip_links:
            jumped = False
            for j in reversed(range(len(curr.skip_links))):
                steps += 1
                candidate = curr.skip_links[j]
                if candidate and candidate.subtree_energy > epsilon:
                    curr = candidate
                    jumped = True
                    break

            if not jumped:
                if curr.left and curr.left.subtree_energy > epsilon:
                    curr = curr.left
                    steps += 1
                elif curr.right and curr.right.subtree_energy > epsilon:
                    curr = curr.right
                    steps += 1
                else:
                    break

        return (curr.depth if curr else self.max_depth), steps

    def update(self, index: int, delta: float) -> int:
        if index < 0 or index >= self.n:
            raise IndexError(f"Index {index} out of bounds")

        curr = self.root
        steps = 0
        path = []

        while curr:
            steps += 1
            path.append(curr)
            if curr.is_leaf:
                break
            mid = curr.start + (curr.length // 2)
            if index < mid:
                curr = curr.left
            else:
                curr = curr.right

        for node in reversed(path):
            node.value += delta / node.length
            node.subtree_energy += abs(delta)

        return steps
