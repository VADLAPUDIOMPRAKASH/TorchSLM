import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.config import GPTConfig
from model.gpt import GPT

from training.generator import GPTGenerator

import sentencepiece as spm


# --------------------------------------------------
# Load Tokenizer
# --------------------------------------------------

tokenizer = spm.SentencePieceProcessor()

tokenizer.load(
    "data/tokenizer/telugu.model"
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

config = GPTConfig(
    vocab_size=8000,
    max_sequence_length=128,
    embedding_dim=128,
    num_heads=4,
    num_layers=4,
)

# --------------------------------------------------
# Model
# --------------------------------------------------

model = GPT(config).to(config.device)

# --------------------------------------------------
# Generator
# --------------------------------------------------

generator = GPTGenerator(
    model=model,
    tokenizer=tokenizer,
)

# --------------------------------------------------
# Generate
# --------------------------------------------------

text = generator.generate(
    prompt="భారతదేశం",
    max_new_tokens=20,
)

print("Prompt:")
print("భారతదేశం")

print()

print("Generated:")
print(text)

print()

assert isinstance(text, str)

print("✅ Generator Test Passed!")