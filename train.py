import logging
from pathlib import Path

import sentencepiece as spm
from torch.utils.data import DataLoader

from dataset.dataset import TeluguDataset

from model.config import GPTConfig
from model.gpt import GPT

from training.loss import GPTLoss
from training.optimizer import GPTOptimizer
from training.evaluator import GPTEvaluator
from training.checkpoint import GPTCheckpoint
from training.generator import GPTGenerator
from training.trainer import GPTTrainer


# =====================================================
# Paths
# =====================================================

ROOT = Path(__file__).resolve().parent

TRAIN_BIN = ROOT / "data" / "encoded" / "train.bin"
VALID_BIN = ROOT / "data" / "encoded" / "valid.bin"

TOKENIZER_MODEL = ROOT / "data" / "tokenizer" / "telugu.model"


# =====================================================
# Logging
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)


# =====================================================
# Configuration
# =====================================================

config = GPTConfig()


# =====================================================
# Tokenizer
# =====================================================

tokenizer = spm.SentencePieceProcessor()
tokenizer.load(str(TOKENIZER_MODEL))


# =====================================================
# Dataset
# =====================================================

train_dataset = TeluguDataset(
    bin_file=TRAIN_BIN,
    context_length=config.max_sequence_length,
)

valid_dataset = TeluguDataset(
    bin_file=VALID_BIN,
    context_length=config.max_sequence_length,
)


# =====================================================
# DataLoader
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=True,
)


# =====================================================
# Model
# =====================================================

model = GPT(config).to(config.device)


# =====================================================
# Loss
# =====================================================

criterion = GPTLoss()


# =====================================================
# Optimizer
# =====================================================

optimizer = GPTOptimizer(
    model=model,
    config=config,
).get_optimizer()


# =====================================================
# Evaluator
# =====================================================

evaluator = GPTEvaluator(
    model=model,
    valid_loader=valid_loader,
    criterion=criterion,
)


# =====================================================
# Checkpoint
# =====================================================

checkpoint = GPTCheckpoint()

if checkpoint.exists("latest.pt"):

    print("Loading latest checkpoint...")

    completed_epoch, batch_idx, global_step, _ = checkpoint.load(
        model=model,
        optimizer=optimizer,
        filename="latest.pt",
    )

    start_epoch = completed_epoch
    start_batch = 0 if batch_idx == -1 else batch_idx + 1

    print(f"Completed Epoch : {completed_epoch}")
    print(f"Resume Batch    : {start_batch}")
    print(f"Global Step     : {global_step}")
    print("Continuing...")

else:

    start_epoch = 0
    start_batch = 0
    global_step = 0

    print("Starting training from scratch.")


# =====================================================
# Generator
# =====================================================

generator = GPTGenerator(
    model=model,
    tokenizer=tokenizer,
)


# =====================================================
# Trainer
# =====================================================

trainer = GPTTrainer(
    model=model,
    train_loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    evaluator=evaluator,
    checkpoint=checkpoint,
)


# =====================================================
# Start Training
# =====================================================

print("=" * 60)
print("Starting Telugu GPT Training")
print("=" * 60)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = model.num_parameters()

print("\nTraining Configuration")
print("-" * 40)
print(f"Device             : {config.device}")
print(f"Batch Size         : {config.batch_size}")
print(f"Context Length     : {config.max_sequence_length}")
print(f"Train Samples      : {len(train_dataset):,}")
print(f"Validation Samples : {len(valid_dataset):,}")
print(f"Train Batches      : {len(train_loader):,}")
print(f"Validation Batches : {len(valid_loader):,}")
print(f"Total Parameters   : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")
print("-" * 40)

trainer.fit(
    epochs=config.epochs,
    start_epoch=start_epoch,
    start_batch=start_batch,
    global_step=global_step,
)

print("\nTraining Complete!")

print("\nSample Generation:\n")

print(
    generator.generate(
        prompt="భారతదేశం",
        max_new_tokens=50,
    )
)