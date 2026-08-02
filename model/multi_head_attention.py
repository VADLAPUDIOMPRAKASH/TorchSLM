import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0, \
            "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # One projection for Q,K,V
        self.qkv = nn.Linear(
            embedding_dim,
            embedding_dim * 3,
            bias=False
        )

        # Output projection
        self.out_proj = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

    def forward(self, x):

        B, T, C = x.shape

        # -----------------------------------
        # Compute QKV
        # -----------------------------------

        qkv = self.qkv(x)

        Q, K, V = qkv.chunk(3, dim=-1)

        # -----------------------------------
        # Split into heads
        # -----------------------------------

        Q = Q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Shapes:
        #
        # (B,H,T,D)

        # -----------------------------------
        # Attention
        # -----------------------------------

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

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

        attention = torch.softmax(
            scores,
            dim=-1
        )

        output = attention @ V

        # -----------------------------------
        # Merge heads
        # -----------------------------------

        output = output.transpose(
            1,
            2
        ).contiguous()

        output = output.view(
            B,
            T,
            C
        )

        output = self.out_proj(output)

        return output, attention