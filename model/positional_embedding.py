import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.
    """

    def __init__(self, max_sequence_length: int, embedding_dim: int):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim

        # Learnable position embedding matrix
        self.weight = nn.Parameter(
            torch.randn(max_sequence_length, embedding_dim) * 0.02
        )

    def forward(self, token_embeddings: torch.Tensor):

        """
        token_embeddings shape:
            (batch_size, sequence_length, embedding_dim)

        returns:
            token_embeddings + positional_embeddings
        """

        batch_size, sequence_length, embedding_dim = token_embeddings.shape

        # Create position indices
        positions = torch.arange(
            sequence_length,
            device=token_embeddings.device
        )

        # Lookup position embeddings
        position_embeddings = self.weight[positions]

        # Add batch dimension
        position_embeddings = position_embeddings.unsqueeze(0)

        # Broadcast addition
        return token_embeddings + position_embeddings