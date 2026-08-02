import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.config import GPTConfig
from model.gpt import GPT

from training.optimizer import GPTOptimizer
from training.checkpoint import GPTCheckpoint


# --------------------------------------------------
# Configuration
# --------------------------------------------------

config = GPTConfig(
    vocab_size=8000,
    max_sequence_length=8,
    embedding_dim=128,
    num_heads=4,
    num_layers=4,
)

# --------------------------------------------------
# Original Model
# --------------------------------------------------

model = GPT(config).to(config.device)

optimizer = GPTOptimizer(
    model=model,
    config=config,
).get_optimizer()

# --------------------------------------------------
# Save Checkpoint
# --------------------------------------------------

checkpoint = GPTCheckpoint()

checkpoint.save(
    model=model,
    optimizer=optimizer,
    epoch=5,
    loss=2.3145,
    filename="test_checkpoint.pt",
)

print("Checkpoint Saved!")

# --------------------------------------------------
# New Model
# --------------------------------------------------

new_model = GPT(config).to(config.device)

new_optimizer = GPTOptimizer(
    model=new_model,
    config=config,
).get_optimizer()

# --------------------------------------------------
# Load Checkpoint
# --------------------------------------------------

epoch, loss = checkpoint.load(
    model=new_model,
    optimizer=new_optimizer,
    filename="test_checkpoint.pt",
)

print()

print("Loaded Epoch:")
print(epoch)

print()

print("Loaded Loss:")
print(loss)

# --------------------------------------------------
# Verify Model Weights
# --------------------------------------------------

weights_match = True

for p1, p2 in zip(model.parameters(), new_model.parameters()):
    if not torch.equal(p1, p2):
        weights_match = False
        break

print()

print("Weights Match:")
print(weights_match)

assert epoch == 5
assert abs(loss - 2.3145) < 1e-6
assert weights_match

print("\n✅ Checkpoint Test Passed!")