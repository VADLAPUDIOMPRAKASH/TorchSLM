from pathlib import Path
import os

CORPUS = Path("data/raw/telugu.txt")

total_lines = 0
total_chars = 0

longest = ""
shortest = None

samples = []

with CORPUS.open("r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        total_lines += 1

        total_chars += len(line)

        if shortest is None or len(line) < len(shortest):
            shortest = line

        if len(line) > len(longest):
            longest = line

        if len(samples) < 10:
            samples.append(line)

file_size = os.path.getsize(CORPUS)

print("=" * 60)
print("TELUGU CORPUS ANALYSIS")
print("=" * 60)

print(f"File Size           : {file_size/1024/1024/1024:.2f} GB")
print(f"Total Sentences     : {total_lines:,}")
print(f"Average Length      : {total_chars/total_lines:.2f} characters")
print(f"Longest Sentence    : {len(longest)} characters")
print(f"Shortest Sentence   : {len(shortest)} characters")

print("\nSample Sentences\n")

for s in samples:
    print("-", s[:150])