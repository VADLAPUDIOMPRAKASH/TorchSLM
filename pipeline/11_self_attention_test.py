import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.embedding import TeluguEmbedding
from model.positional_embedding import PositionalEmbedding
from model.self_attention import SelfAttention


VOCAB_SIZE = 8000
EMBED_DIM = 128
MAX_SEQ_LEN = 16

token_embedding = TeluguEmbedding(
    VOCAB_SIZE,
    EMBED_DIM
)

position_embedding = PositionalEmbedding(
    MAX_SEQ_LEN,
    EMBED_DIM
)

attention = SelfAttention(
    EMBED_DIM
)

tokens = torch.tensor([
    [670, 902, 6335, 1515]
])

x = token_embedding(tokens)

x = position_embedding(x)

print("Input Shape:")
print(x.shape)

output, weights = attention(x)

print("\nOutput Shape:")
print(output.shape)

print("\nFirst Output Vector:")
print(output[0, 0])

print("Attention Shape:")
print(weights.shape)

print("\nAttention Matrix:")
print(weights[0])