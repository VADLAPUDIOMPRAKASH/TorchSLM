import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.config import GPTConfig
from model.gpt import GPT
from training.loss import GPTLoss


# ------------------------------------
# Configuration
# ------------------------------------

config = GPTConfig(
    vocab_size=8000,
    max_sequence_length=8,
    embedding_dim=128,
    num_heads=4,
    num_layers=4
)

# ------------------------------------
# Build Model
# ------------------------------------

model = GPT(config).to(config.device)

criterion = GPTLoss()

# ------------------------------------
# Dummy Input
# ------------------------------------

batch_size = 2
sequence_length = 8

inputs = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(batch_size, sequence_length)
)

# Random targets (unit test only)
targets = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(batch_size, sequence_length)
)

# ------------------------------------
# Forward
# ------------------------------------

logits = model(inputs)

loss = criterion(
    logits,
    targets
)

# ------------------------------------
# Results
# ------------------------------------

print("Input Shape:")
print(inputs.shape)

print("\nTarget Shape:")
print(targets.shape)

print("\nLogits Shape:")
print(logits.shape)

print("\nLoss:")
print(loss)

print("\nLoss Value:")
print(loss.item())

print("\nLoss Shape:")
print(loss.shape)

print("\nRequires Gradient:")
print(loss.requires_grad)

# ------------------------------------
# Assertions
# ------------------------------------

assert logits.shape == (
    batch_size,
    sequence_length,
    config.vocab_size
)

assert loss.ndim == 0

assert loss.requires_grad

print("\n✅ GPT Loss Test Passed!")