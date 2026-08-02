import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.config import GPTConfig
from model.gpt import GPT
from training.optimizer import GPTOptimizer

config = GPTConfig()

model = GPT(config).to(config.device)

optimizer = GPTOptimizer(
    model=model,
    config=config
).get_optimizer()

print(type(optimizer).__name__)
print()

print("Learning Rate:", optimizer.param_groups[0]["lr"])
print("Betas:", optimizer.param_groups[0]["betas"])
print("Epsilon:", optimizer.param_groups[0]["eps"])
print("Weight Decay:", optimizer.param_groups[0]["weight_decay"])

assert optimizer.param_groups[0]["lr"] == 3e-4
assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)
assert optimizer.param_groups[0]["eps"] == 1e-8
assert optimizer.param_groups[0]["weight_decay"] == 0.1

print("\n✅ Optimizer Test Passed!")