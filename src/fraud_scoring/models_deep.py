"""MLP with an embedding layer for the merchant category, the deep candidate."""

import torch
from torch import nn


class FraudMLP(nn.Module):
    def __init__(self, n_continuous: int, n_categories: int, embedding_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(n_categories + 1, embedding_dim)  # +1 for unknown/unseen
        dims = [n_continuous + embedding_dim] + hidden_dims
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_continuous: torch.Tensor, category_idx: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(category_idx)
        x = torch.cat([x_continuous, emb], dim=1)
        return self.mlp(x).squeeze(-1)
