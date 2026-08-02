import torch
import torch.nn as nn
from typing import List, Tuple, Union

from model.config import GPTConfig
from model.embedding import TeluguEmbedding
from model.positional_embedding import PositionalEmbedding
from model.layer_norm import LayerNorm
from model.transformer_block import TransformerBlock


class GPT(nn.Module):
    """
    Decoder-only Transformer for next-token language modelling.

    The model combines custom token and positional embeddings with a stack of
    causal Transformer blocks, then projects each hidden state to vocabulary
    logits. Its input embedding and output projection share one weight matrix,
    as in GPT-2, which reduces parameters and improves language modelling.
    """

    def __init__(
        self,
        config: GPTConfig
    ) -> None:
        super().__init__()

        # Keep the model settings with the checkpointed model.
        self.config = config

        # Token Embedding
        self.token_embedding = TeluguEmbedding(
            config.vocab_size,
            config.embedding_dim
        )

        # Position Embedding
        self.position_embedding = PositionalEmbedding(
            config.max_sequence_length,
            config.embedding_dim
        )

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                expansion_factor=config.expansion_factor
            )
            for _ in range(config.num_layers)
        ])

        # Final LayerNorm
        self.final_ln = LayerNorm(
            embedding_dim=config.embedding_dim
        )

        # Language Modeling Head
        self.lm_head = nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.vocab_size,
            bias=False
        )

        # Input embedding -> same weights -> output projection.
        # Weight tying reduces parameters and follows the GPT-2 architecture.
        self.lm_head.weight = self.token_embedding.weight

        # Centralize initialization for standard linear layers. The tied output
        # head is excluded because its weight is the custom embedding weight.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize linear layers with GPT-style normal initialization.

        GPT commonly uses a normal distribution with standard deviation 0.02,
        which gives projection layers a small, stable starting scale. Custom
        embedding layers manage their own initialization and are left intact.
        """
        if isinstance(module, nn.Linear) and module is not self.lm_head:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        """Return the number of trainable model parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Return vocabulary logits, optionally including attention maps."""

        # Token Embeddings
        x = self.token_embedding(token_ids)

        # Position Embeddings
        x = self.position_embedding(x)

        attention_weights: List[torch.Tensor] = []

        # Transformer Blocks
        for block in self.blocks:
            x, attn = block(x)

            if return_attention:
                attention_weights.append(attn)

        # Final LayerNorm
        x = self.final_ln(x)

        # Vocabulary Projection
        logits = self.lm_head(x)

        if return_attention:
            return logits, attention_weights

        return logits
