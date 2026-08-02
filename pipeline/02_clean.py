from pathlib import Path
import re

INPUT = Path("data/raw/telugu.txt")
OUTPUT = Path("data/cleaned/telugu_clean.txt")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 4096
MIN_LENGTH = 5

stats = {
    "kept": 0,
    "empty": 0,
    "too_short": 0,
    "too_long": 0,
}

with INPUT.open("r", encoding="utf-8") as fin, \
     OUTPUT.open("w", encoding="utf-8") as fout:

    for line in fin:

        # Remove leading/trailing whitespace
        line = line.strip()

        # Remove empty lines
        if not line:
            stats["empty"] += 1
            continue

        # Replace multiple spaces/tabs with a single space
        line = re.sub(r"\s+", " ", line)

        # Filter very short lines
        if len(line) < MIN_LENGTH:
            stats["too_short"] += 1
            continue

        # Filter very long lines
        if len(line) > MAX_LENGTH:
            stats["too_long"] += 1
            continue

        fout.write(line + "\n")
        stats["kept"] += 1

print("=" * 60)
print("Cleaning Complete")
print("=" * 60)

for key, value in stats.items():
    print(f"{key:12}: {value:,}")