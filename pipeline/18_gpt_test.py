import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.gpt import GPT
from model.config import GPTConfig


VOCAB_SIZE = 8000
EMBED_DIM = 128
MAX_SEQ_LEN = 16
HEADS = 4
LAYERS = 4

config = GPTConfig(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBED_DIM,
    max_sequence_length=MAX_SEQ_LEN,
    num_heads=HEADS,
    num_layers=LAYERS
)
model = GPT(config).to(config.device)

tokens = torch.tensor([
    [670, 902, 6335, 1515]
])

print("Input Shape:")
print(tokens.shape)

logits = model(tokens)

assert isinstance(logits, torch.Tensor)
assert logits.shape == (1, 4, VOCAB_SIZE)

print("\nTrainable Parameters:")
print(f"{model.num_parameters():,}")

print("\nLogits Shape:")
print(logits.shape)

output_with_attention = model(
    tokens,
    return_attention=True
)

assert isinstance(output_with_attention, tuple)

logits_with_attention, attention = output_with_attention

assert logits_with_attention.shape == (1, 4, VOCAB_SIZE)
assert len(attention) == config.num_layers

print("\nNumber of Transformer Blocks:")
print(len(attention))

print("\nAttention Shape (Block 1):")
print(attention[0].shape)

print("\nPrediction Vector Shape:")
print(logits[0, 0].shape)

print("\nFirst 10 Logits:")
print(logits[0, 0, :10])


print("\n==============================")
print("Weight Tying Test")
print("==============================")

print(
    model.lm_head.weight.data_ptr()
    ==
    model.token_embedding.weight.data_ptr()
)

print("\n==============================")
print("Weight Initialization Test")
print("==============================")

print("Mean :", model.lm_head.weight.mean().item())
print("Std  :", model.lm_head.weight.std().item())