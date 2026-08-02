import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim

        # One projection for Q,K,V
        self.qkv = nn.Linear(
            embedding_dim,
            embedding_dim * 3,
            bias=False
        )

    def forward(self, x):

        B, T, C = x.shape

        # -------------------------
        # Single projection
        # -------------------------

        qkv = self.qkv(x)

        # -------------------------
        # Split into Q,K,V
        # -------------------------

        Q, K, V = qkv.chunk(3, dim=-1)

        # -------------------------
        # Attention Scores
        # -------------------------

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(C)

        # -------------------------
        # Causal Mask
        # -------------------------

        mask = torch.triu(
            torch.ones(
                T,
                T,
                device=x.device
            ),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        # -------------------------
        # Softmax
        # -------------------------

        attention = torch.softmax(
            scores,
            dim=-1
        )

        # -------------------------
        # Weighted Sum
        # -------------------------

        output = attention @ V

        return output, attention