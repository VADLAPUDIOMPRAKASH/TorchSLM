import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.config import GPTConfig
from model.gpt import GPT
from training.loss import GPTLoss
from training.optimizer import GPTOptimizer


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
# Model
# --------------------------------------------------

model = GPT(config).to(config.device)

criterion = GPTLoss()

optimizer = GPTOptimizer(
    model=model,
    config=config
).get_optimizer()

# --------------------------------------------------
# Dummy Batch
# --------------------------------------------------

batch_size = 2
sequence_length = 8

inputs = torch.randint(
    0,
    config.vocab_size,
    (batch_size, sequence_length)
)

targets = torch.randint(
    0,
    config.vocab_size,
    (batch_size, sequence_length)
)

# --------------------------------------------------
# Save Initial Weight
# --------------------------------------------------

before = model.token_embedding.weight.clone()

# --------------------------------------------------
# Training Step
# --------------------------------------------------

optimizer.zero_grad()

logits = model(inputs)

loss = criterion(
    logits,
    targets
)

loss.backward()

optimizer.step()

# --------------------------------------------------
# Save Updated Weight
# --------------------------------------------------

after = model.token_embedding.weight

# --------------------------------------------------
# Verification
# --------------------------------------------------

#weight_changed = not torch.equal(before, after)
weight_changed = not torch.allclose(before, after)

gradient_exists = (
    model.token_embedding.weight.grad is not None
)

print("Loss:", loss.item())
print()

print("Gradient Exists:", gradient_exists)

print("Weights Updated:", weight_changed)

assert gradient_exists
assert weight_changed

print("\n✅ First Training Step Successful!")