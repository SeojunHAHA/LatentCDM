from __future__ import annotations

import torch
import torch.nn as nn


class KVMemoryBridge(nn.Module):
    """Project each DA layer's KV cache into planner-readable memory tokens."""

    def __init__(
        self,
        num_layers: int,
        kv_dim: int,
        planner_dim: int,
        hidden_dim: int,
        memory_tokens_per_layer: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.memory_tokens_per_layer = memory_tokens_per_layer
        self.planner_dim = planner_dim
        self.layer_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(kv_dim),
                    nn.Linear(kv_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, memory_tokens_per_layer * planner_dim),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, past_key_values) -> torch.Tensor:
        layer_tokens = []
        for layer_idx, (key, value) in enumerate(past_key_values):
            # key/value: [batch, kv_heads, seq_len, head_dim]
            pooled_key = key.float().mean(dim=2).flatten(start_dim=1)
            pooled_value = value.float().mean(dim=2).flatten(start_dim=1)
            pooled = torch.cat([pooled_key, pooled_value], dim=-1)
            projected = self.layer_projectors[layer_idx](pooled)
            layer_tokens.append(
                projected.view(-1, self.memory_tokens_per_layer, self.planner_dim)
            )

        return torch.cat(layer_tokens, dim=1)


class BeliefProjector(nn.Module):
    """Map candidate-level diagnosis belief into planner-readable belief tokens."""

    def __init__(
        self,
        num_diseases: int,
        planner_dim: int,
        hidden_dim: int,
        num_belief_tokens: int,
    ) -> None:
        super().__init__()
        self.num_belief_tokens = num_belief_tokens
        self.planner_dim = planner_dim
        self.disease_embeddings = nn.Embedding(num_diseases, planner_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(planner_dim),
            nn.Linear(planner_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_belief_tokens * planner_dim),
        )

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        weighted = belief.float() @ self.disease_embeddings.weight.float()
        projected = self.projector(weighted)
        return projected.view(-1, self.num_belief_tokens, self.planner_dim)
