import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.feed_forward import FeedForward


EMBED_DIM = 128

my_ffn = FeedForward(EMBED_DIM)

torch_ffn = nn.Sequential(
    nn.Linear(EMBED_DIM, EMBED_DIM * 4),
    nn.GELU(),
    nn.Linear(EMBED_DIM * 4, EMBED_DIM),
)

with torch.no_grad():
    torch_ffn[0].weight.copy_(my_ffn.fc1.weight)
    torch_ffn[0].bias.copy_(my_ffn.fc1.bias)

    torch_ffn[2].weight.copy_(my_ffn.fc2.weight)
    torch_ffn[2].bias.copy_(my_ffn.fc2.bias)

x = torch.randn(2, 8, EMBED_DIM)

my_output = my_ffn(x)

torch_output = torch_ffn(x)

print("Outputs Equal:")
print(torch.allclose(
    my_output,
    torch_output,
    atol=1e-6
))

print("\nMaximum Difference:")
print(torch.max(
    torch.abs(my_output - torch_output)
))