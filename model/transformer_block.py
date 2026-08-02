import torch
import torch.nn as nn

from model.layer_norm import LayerNorm
from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        expansion_factor: int = 4
    ):
        super().__init__()

        # First LayerNorm
        self.ln1 = LayerNorm(embedding_dim)

        # Multi-Head Attention
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        # Second LayerNorm
        self.ln2 = LayerNorm(embedding_dim)

        # Feed Forward Network
        self.ffn = FeedForward(
            embedding_dim=embedding_dim,
            expansion_factor=expansion_factor
        )

    def forward(self, x):

        # -------------------------
        # Attention Block
        # -------------------------

        attention_output, attention_weights = self.attention(
            self.ln1(x)
        )

        x = x + attention_output

        # -------------------------
        # Feed Forward Block
        # -------------------------

        ffn_output = self.ffn(
            self.ln2(x)
        )

        x = x + ffn_output

        return x, attention_weights