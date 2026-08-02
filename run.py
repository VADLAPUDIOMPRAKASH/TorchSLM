import sys
from pathlib import Path

import torch
import sentencepiece as spm

# --------------------------------------------------
# Project Root
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# --------------------------------------------------
# Imports
# --------------------------------------------------

from model.config import GPTConfig
from model.gpt import GPT

from training.generator import GPTGenerator
from training.checkpoint import GPTCheckpoint
from training.optimizer import GPTOptimizer

# --------------------------------------------------
# Load Tokenizer
# --------------------------------------------------

tokenizer = spm.SentencePieceProcessor()
tokenizer.load("data/tokenizer/telugu.model")

# --------------------------------------------------
# Configuration
# --------------------------------------------------

config = GPTConfig()

# --------------------------------------------------
# Model
# --------------------------------------------------

model = GPT(config).to(config.device)

# --------------------------------------------------
# Optimizer (required by current checkpoint loader)
# --------------------------------------------------

optimizer = GPTOptimizer(
    model=model,
    config=config,
).get_optimizer()

# --------------------------------------------------
# Load Checkpoint
# --------------------------------------------------

checkpoint = GPTCheckpoint()

epoch, batch_idx, global_step, loss = checkpoint.load(
    model=model,
    optimizer=optimizer,
    filename="epoch_4.pt",
)

print("=" * 60)
print("Checkpoint Loaded Successfully")
print("=" * 60)
print(f"Completed Epoch : {epoch}")
print(f"Batch           : {batch_idx}")
print(f"Global Step     : {global_step}")
print(f"Loss            : {loss:.4f}")
print("=" * 60)

# --------------------------------------------------
# Evaluation Mode
# --------------------------------------------------

model.eval()

generator = GPTGenerator(
    model=model,
    tokenizer=tokenizer,
)

# --------------------------------------------------
# Interactive Generation
# --------------------------------------------------

print("\nType a Telugu prompt.")
print("Type 'exit' to quit.\n")

while True:

    prompt = input("Prompt > ").strip()

    if prompt.lower() in ("exit", "quit"):
        break

    if not prompt:
        continue

    with torch.no_grad():

        output = generator.generate(
            prompt=prompt,
            max_new_tokens=100,
        )

    print("\nGenerated:\n")
    print(output)
    print("\n" + "-" * 60 + "\n")