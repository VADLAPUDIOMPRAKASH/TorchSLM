import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.config import GPTConfig
from model.gpt import GPT

from training.loss import GPTLoss
from training.evaluator import GPTEvaluator


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

# --------------------------------------------------
# Dummy Validation Dataset
# --------------------------------------------------

inputs = torch.randint(
    0,
    config.vocab_size,
    (16, 8)
)

targets = torch.randint(
    0,
    config.vocab_size,
    (16, 8)
)

dataset = TensorDataset(
    inputs,
    targets,
)

valid_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
)

# --------------------------------------------------
# Evaluator
# --------------------------------------------------

evaluator = GPTEvaluator(
    model=model,
    valid_loader=valid_loader,
    criterion=criterion,
)

loss = evaluator.evaluate()

# --------------------------------------------------
# Results
# --------------------------------------------------

print("Validation Loss:")
print(loss)

assert isinstance(loss, float)
assert loss > 0

print("\n✅ Evaluator Test Passed!")