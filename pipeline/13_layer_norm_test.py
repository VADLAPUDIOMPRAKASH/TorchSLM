import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.embedding import TeluguEmbedding
from model.positional_embedding import PositionalEmbedding
from model.layer_norm import LayerNorm


VOCAB_SIZE = 8000
EMBED_DIM = 128
MAX_SEQ_LEN = 16


embedding = TeluguEmbedding(
    VOCAB_SIZE,
    EMBED_DIM
)

position = PositionalEmbedding(
    MAX_SEQ_LEN,
    EMBED_DIM
)

layer_norm = LayerNorm(
    EMBED_DIM
)

tokens = torch.tensor([
    [670, 902, 6335, 1515]
])

x = embedding(tokens)
x = position(x)

print("Before LayerNorm")

print("Shape:", x.shape)

print("Mean:")
print(x.mean(dim=-1))

print("Std:")
print(x.std(dim=-1))

print("-" * 50)

y = layer_norm(x)

print("After LayerNorm")

print("Shape:", y.shape)

print("Mean:")
print(y.mean(dim=-1))

print("Std:")
print(y.std(dim=-1))