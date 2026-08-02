import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.embedding import TeluguEmbedding
from model.positional_embedding import PositionalEmbedding
from model.transformer_block import TransformerBlock


VOCAB_SIZE = 8000
EMBED_DIM = 128
MAX_SEQ_LEN = 16
NUM_HEADS = 4


embedding = TeluguEmbedding(
    VOCAB_SIZE,
    EMBED_DIM
)

position = PositionalEmbedding(
    MAX_SEQ_LEN,
    EMBED_DIM
)

block = TransformerBlock(
    embedding_dim=EMBED_DIM,
    num_heads=NUM_HEADS
)

tokens = torch.tensor([
    [670, 902, 6335, 1515]
])

x = embedding(tokens)
x = position(x)

print("Input Shape:")
print(x.shape)

output, attention = block(x)

print("\nOutput Shape:")
print(output.shape)

print("\nAttention Shape:")
print(attention.shape)

print("\nFirst Output Vector:")
print(output[0, 0])