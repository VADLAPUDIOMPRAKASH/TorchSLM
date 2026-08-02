import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.config import GPTConfig
from model.gpt import GPT

from training.loss import GPTLoss
from training.optimizer import GPTOptimizer
from training.trainer import GPTTrainer


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
    config=config,
).get_optimizer()

# --------------------------------------------------
# Dummy Dataset
# --------------------------------------------------

inputs = torch.randint(
    0,
    config.vocab_size,
    (2, 8)
)

targets = torch.randint(
    0,
    config.vocab_size,
    (2, 8)
)

dataset = TensorDataset(inputs, targets)

train_loader = DataLoader(
    dataset,
    batch_size=2,
)

# --------------------------------------------------
# Trainer
# --------------------------------------------------

trainer = GPTTrainer(
    model=model,
    train_loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
)

# --------------------------------------------------
# One Training Step
# --------------------------------------------------

loss = trainer.train_step(
    inputs,
    targets,
)

print("Loss:")
print(loss)

print()

print("Loss Type:")
print(type(loss))

assert isinstance(loss, float)

assert loss > 0

print("\n✅ Trainer Test Passed!")