from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


# ==========================================================
# Paths
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "cleaned" / "telugu_clean.txt"

MODEL_FILE = ROOT / "data" / "tokenizer" / "telugu.model"

OUTPUT_DIR = ROOT / "data" / "encoded"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "train.bin"
VALID_FILE = OUTPUT_DIR / "valid.bin"


# ==========================================================
# Configuration
# ==========================================================

TRAIN_SPLIT = 0.99

DTYPE = np.uint16


# ==========================================================
# Load SentencePiece
# ==========================================================

print("Loading tokenizer...")

sp = spm.SentencePieceProcessor()
sp.load(str(MODEL_FILE))

print("Tokenizer loaded.\n")


# ==========================================================
# Count total lines
# ==========================================================

print("Counting lines...")

with INPUT_FILE.open("r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

print(f"Total lines : {total_lines:,}")

train_limit = int(total_lines * TRAIN_SPLIT)

print(f"Training lines   : {train_limit:,}")
print(f"Validation lines : {total_lines - train_limit:,}\n")


# ==========================================================
# Encode + Stream directly to binary files
# ==========================================================

print("Encoding corpus...\n")

train_tokens = 0
valid_tokens = 0

with (
    INPUT_FILE.open("r", encoding="utf-8") as infile,
    TRAIN_FILE.open("wb") as train_out,
    VALID_FILE.open("wb") as valid_out,
):

    for idx, line in enumerate(tqdm(infile, total=total_lines)):

        line = line.strip()

        if not line:
            continue

        ids = sp.encode(line)

        # Append EOS token
        ids.append(sp.eos_id())

        arr = np.asarray(ids, dtype=DTYPE)

        if idx < train_limit:
            arr.tofile(train_out)
            train_tokens += len(arr)
        else:
            arr.tofile(valid_out)
            valid_tokens += len(arr)


# ==========================================================
# Done
# ==========================================================

print("\nEncoding completed successfully.\n")

print(f"Training tokens   : {train_tokens:,}")
print(f"Validation tokens : {valid_tokens:,}")

print(f"\nTrain file : {TRAIN_FILE}")
print(f"Valid file : {VALID_FILE}")

print("\nFile Sizes:")

print(f"Train : {TRAIN_FILE.stat().st_size / (1024**3):.2f} GB")
print(f"Valid : {VALID_FILE.stat().st_size / (1024**2):.2f} MB")