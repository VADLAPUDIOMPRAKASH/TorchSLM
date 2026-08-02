import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.embedding import TeluguEmbedding


VOCAB_SIZE = 8000
EMBED_DIM = 128

# Our implementation
my_embedding = TeluguEmbedding(VOCAB_SIZE, EMBED_DIM)

# Official PyTorch implementation
torch_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

# Copy weights
with torch.no_grad():
    torch_embedding.weight.copy_(my_embedding.weight)

tokens = torch.tensor([670, 902, 6335, 1515])

my_output = my_embedding(tokens)
torch_output = torch_embedding(tokens)

print("Outputs Equal:", torch.allclose(my_output, torch_output))

print("\nMaximum Difference:")
print((my_output - torch_output).abs().max())