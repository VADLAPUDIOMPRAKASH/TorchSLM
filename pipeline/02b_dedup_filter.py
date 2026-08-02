import hashlib
import re
import time
from pathlib import Path

INPUT = Path("data/cleaned/telugu_clean.txt")
OUTPUT = Path("data/cleaned/telugu_clean_v2.txt")

MIN_WORDS = 4
TELUGU_RATIO_MIN = 0.5

TELUGU_RE = re.compile(r"[ఀ-౿]")
ARTIFACT_RE = re.compile(r"నుండి వెలికితీశారు|wikipedia\.org|https?://")

stats = {
    "kept": 0,
    "duplicate": 0,
    "too_few_words": 0,
    "not_telugu_majority": 0,
    "extraction_artifact": 0,
}

seen = set()
start = time.time()

with INPUT.open("r", encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        line = line.strip()
        if not line:
            continue

        if ARTIFACT_RE.search(line):
            stats["extraction_artifact"] += 1
            continue

        if len(line.split()) < MIN_WORDS:
            stats["too_few_words"] += 1
            continue

        telugu_chars = len(TELUGU_RE.findall(line))
        if telugu_chars / len(line) < TELUGU_RATIO_MIN:
            stats["not_telugu_majority"] += 1
            continue

        h = hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest()
        if h in seen:
            stats["duplicate"] += 1
            continue
        seen.add(h)

        fout.write(line + "\n")
        stats["kept"] += 1

        if (i + 1) % 1_000_000 == 0:
            elapsed = time.time() - start
            print(f"processed={i+1:,} kept={stats['kept']:,} elapsed={elapsed:,.0f}s", flush=True)

elapsed = time.time() - start
print("=" * 60)
print(f"Done in {elapsed:,.0f}s")
for k, v in stats.items():
    print(f"{k:20}: {v:,}")
print(f"Output: {OUTPUT} ({OUTPUT.stat().st_size / 1e9:.2f} GB)")
