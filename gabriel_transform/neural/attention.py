from typing import Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def gabriel_attention_numpy(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    window_size: int = 64,
    q_decay: float = 0.5,
    epsilon: float = 1e-4,
) -> Tuple[np.ndarray, dict]:
    N, D = Q.shape
    scale = 1.0 / np.sqrt(D)

    out = np.zeros_like(Q)
    total_evals = 0

    k_pyramid = [K]
    v_pyramid = [V]

    curr_k = K
    curr_v = V
    while curr_k.shape[0] > 1:
        curr_len = curr_k.shape[0]
        if curr_len % 2 != 0:
            curr_k = np.pad(curr_k, ((0, 1), (0, 0)), mode="edge")
            curr_v = np.pad(curr_v, ((0, 1), (0, 0)), mode="edge")

        next_k = 0.5 * (curr_k[0::2] + curr_k[1::2])
        next_v = 0.5 * (curr_v[0::2] + curr_v[1::2])
        k_pyramid.append(next_k)
        v_pyramid.append(next_v)
        curr_k = next_k
        curr_v = next_v

    for i in range(N):
        w_start = max(0, i - window_size)
        w_end = min(N, i + window_size + 1)

        local_k = K[w_start:w_end]
        local_v = V[w_start:w_end]

        local_scores = np.dot(local_k, Q[i]) * scale
        weights = [local_scores]
        values = [local_v]
        total_evals += len(local_scores)

        for lvl_idx in range(1, len(k_pyramid)):
            lvl_k = k_pyramid[lvl_idx]
            lvl_v = v_pyramid[lvl_idx]
            lvl_scale_bound = (q_decay ** lvl_idx)

            if lvl_scale_bound <= epsilon:
                break

            stride_len = 1 << lvl_idx
            for cluster_idx in range(len(lvl_k)):
                cluster_center = cluster_idx * stride_len + (stride_len // 2)
                if abs(cluster_center - i) > window_size:
                    score = float(np.dot(lvl_k[cluster_idx], Q[i]) * scale)
                    weights.append(np.array([score * lvl_scale_bound]))
                    values.append(lvl_v[cluster_idx:cluster_idx + 1])
                    total_evals += 1

        all_scores = np.concatenate(weights)
        all_vals = np.concatenate(values, axis=0)

        exp_scores = np.exp(all_scores - np.max(all_scores))
        probs = exp_scores / np.sum(exp_scores)
        out[i] = np.sum(all_vals * probs[:, np.newaxis], axis=0)

    dense_evals = N * N
    flops_saved_ratio = 1.0 - (float(total_evals) / float(dense_evals))

    stats = {
        "sequence_length": N,
        "dense_evaluations": dense_evals,
        "gabriel_evaluations": total_evals,
        "evals_per_query": total_evals / max(1, N),
        "sparsity_ratio": flops_saved_ratio,
    }

    return out, stats


if HAS_TORCH:
    class GabrielAttention(nn.Module):
        def __init__(
            self,
            d_model: int,
            num_heads: int = 8,
            window_size: int = 64,
            q_decay: float = 0.5,
            epsilon: float = 1e-4,
        ):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_head = d_model // num_heads
            self.window_size = window_size
            self.q_decay = q_decay
            self.epsilon = epsilon

            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, N, D = x.shape
            Q = self.q_proj(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
            K = self.k_proj(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
            V = self.v_proj(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)

            idx = torch.arange(N, device=x.device)
            dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))

            horn_penalty = torch.zeros((N, N), device=x.device, dtype=x.dtype)
            far_mask = dist > self.window_size
            if far_mask.any():
                scale_levels = torch.clamp(torch.ceil(torch.log2(torch.clamp(dist.float() / max(1, self.window_size), min=1.0))), min=1.0)
                penalty = scale_levels * float(np.log(self.q_decay))
                cutoff_level = float(np.ceil(np.log(self.epsilon) / np.log(self.q_decay)))
                penalty = torch.where(scale_levels >= cutoff_level, torch.tensor(-1e4, device=x.device, dtype=x.dtype), penalty.to(x.dtype))
                horn_penalty = torch.where(far_mask, penalty, horn_penalty)

            scores = (torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)) + horn_penalty.unsqueeze(0).unsqueeze(0)
            probs = F.softmax(scores, dim=-1)
            context = torch.matmul(probs, V)
            context = context.transpose(1, 2).contiguous().view(B, N, D)
            return self.out_proj(context)
