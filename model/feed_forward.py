import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        expansion_factor: int = 4
    ):
        super().__init__()

        hidden_dim = embedding_dim * expansion_factor

        self.fc1 = nn.Linear(
            embedding_dim,
            hidden_dim
        )

        self.gelu = nn.GELU()

        self.fc2 = nn.Linear(
            hidden_dim,
            embedding_dim
        )

    def forward(self, x):

        x = self.fc1(x)

        x = self.gelu(x)

        x = self.fc2(x)

        return x