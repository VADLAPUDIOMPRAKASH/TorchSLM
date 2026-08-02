import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.embedding import TeluguEmbedding
from model.positional_embedding import PositionalEmbedding
from model.multi_head_attention import MultiHeadAttention


VOCAB_SIZE = 8000
EMBED_DIM = 128
MAX_SEQ_LEN = 16
HEADS = 4


token_embedding = TeluguEmbedding(
    VOCAB_SIZE,
    EMBED_DIM
)

position_embedding = PositionalEmbedding(
    MAX_SEQ_LEN,
    EMBED_DIM
)

mha = MultiHeadAttention(
    EMBED_DIM,
    HEADS
)

tokens = torch.tensor([
    [670, 902, 6335, 1515]
])

x = token_embedding(tokens)

x = position_embedding(x)

print("Input Shape:")
print(x.shape)

output, attention = mha(x)

print("\nOutput Shape:")
print(output.shape)

print("\nAttention Shape:")
print(attention.shape)

print("\nHead 0 Attention:")
print(attention[0, 0])