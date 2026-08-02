import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Single Head Causal Self-Attention
    """

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Query projection
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        # Key projection
        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        # Value projection
        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

    def forward(self, x):
        """
        x shape:
            (batch_size, sequence_length, embedding_dim)
        """

        # ----------------------------
        # Step 1
        # ----------------------------

        Q = self.query(x)

        # ----------------------------
        # Step 2
        # ----------------------------

        K = self.key(x)

        # ----------------------------
        # Step 3
        # ----------------------------

        V = self.value(x)

        # ----------------------------
        # Step 4
        # Attention Scores
        # ----------------------------

        scores = Q @ K.transpose(-2, -1)

        # ----------------------------
        # Step 5
        # Scaling
        # ----------------------------

        scores = scores / math.sqrt(self.embedding_dim)

        # ----------------------------
        # Step 6
        # Causal Mask
        # ----------------------------

        sequence_length = x.size(1)

        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            ),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, float("-inf"))

        # ----------------------------
        # Step 7
        # Softmax
        # ----------------------------

        attention = torch.softmax(
            scores,
            dim=-1
        )

        # ----------------------------
        # Step 8
        # Weighted Sum
        # ----------------------------

        output = attention @ V

        return output, attention