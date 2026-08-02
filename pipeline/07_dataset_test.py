import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.dataset import TeluguDataset

TRAIN_FILE = ROOT / "data" / "encoded" / "train.bin"

dataset = TeluguDataset(
    bin_file=TRAIN_FILE,
    context_length=16,
)

print("Dataset Size:", len(dataset))

x, y = dataset[0]

print("\nInput Shape :", x.shape)
print("Target Shape:", y.shape)

print("\nInput:")
print(x)

print("\nTarget:")
print(y)