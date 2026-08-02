import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.embedding import TeluguEmbedding
from model.positional_embedding import PositionalEmbedding


VOCAB_SIZE = 8000
EMBED_DIM = 128
MAX_SEQ_LEN = 16


token_embedding = TeluguEmbedding(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBED_DIM
)

position_embedding = PositionalEmbedding(
    max_sequence_length=MAX_SEQ_LEN,
    embedding_dim=EMBED_DIM
)

tokens = torch.tensor([
    [670, 902, 6335, 1515],
    [100, 200, 300, 400]
])

token_vectors = token_embedding(tokens)

final_vectors = position_embedding(token_vectors)

print("Input Tokens Shape:")
print(tokens.shape)

print("\nToken Embedding Shape:")
print(token_vectors.shape)

print("\nFinal Embedding Shape:")
print(final_vectors.shape)

print("\nFirst Token Before Position:")
print(token_vectors[0, 0])

print("\nFirst Token After Position:")
print(final_vectors[0, 0])