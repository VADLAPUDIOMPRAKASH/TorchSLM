import os
import sentencepiece as spm
from pathlib import Path

INPUT = Path("data/cleaned/telugu_clean.txt")
OUTPUT_DIR = Path("data/tokenizer")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

spm.SentencePieceTrainer.train(
    input=str(INPUT),
    model_prefix=str(OUTPUT_DIR / "telugu"),

    model_type="unigram",
    vocab_size=8000,

    character_coverage=1.0,

    # Sample 1 million sentences to keep memory use within this machine's 16GB RAM
    # (5M sentences ~1.2B chars was OOM-killing the process during EM training)
    input_sentence_size=1_000_000,
    shuffle_input_sentence=True,

    train_extremely_large_corpus=True,

    max_sentence_length=4096,

    split_digits=False,

    num_threads=os.cpu_count()
)

print("Tokenizer training completed!")