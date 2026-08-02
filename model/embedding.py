import torch
import torch.nn as nn


class TeluguEmbedding(nn.Module):
    """
    A simple token embedding layer implemented from scratch.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Learnable embedding matrix
        self.weight = nn.Parameter(
            torch.randn(vocab_size, embedding_dim) * 0.02
        )

    def forward(self, token_ids: torch.Tensor):

        return self.weight[token_ids]