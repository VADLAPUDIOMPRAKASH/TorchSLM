import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.embedding import TeluguEmbedding


VOCAB_SIZE = 8000
EMBED_DIM = 128


embedding = TeluguEmbedding(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBED_DIM
)

tokens = torch.tensor([
    670,
    902,
    6335,
    1515
])

vectors = embedding(tokens)

print("Input Tokens")
print(tokens)

print("\nEmbedding Shape")
print(vectors.shape)

print("\nEmbedding Vector (first token)")
print(vectors[0])