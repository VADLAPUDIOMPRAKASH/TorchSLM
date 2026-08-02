import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-5
    ):
        super().__init__()

        self.eps = eps

        # Learnable scale (γ)
        self.gamma = nn.Parameter(
            torch.ones(embedding_dim)
        )

        # Learnable shift (β)
        self.beta = nn.Parameter(
            torch.zeros(embedding_dim)
        )

    def forward(self, x):

        # Mean of each token embedding
        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        # Variance of each token embedding
        variance = (
            (x - mean) ** 2
        ).mean(
            dim=-1,
            keepdim=True
        )

        # Normalize
        x_hat = (
            x - mean
        ) / torch.sqrt(
            variance + self.eps
        )

        # Scale and Shift
        output = (
            self.gamma * x_hat
        ) + self.beta

        return output