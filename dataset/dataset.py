from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TeluguDataset(Dataset):
    """
    GPT-style dataset using memory-mapped binary token files.

    Samples are non-overlapping blocks of context_length tokens:
        x = tokens[i*context_length : i*context_length + context_length]
        y = tokens[i*context_length + 1 : i*context_length + context_length + 1]
    """

    def __init__(self, bin_file: str | Path, context_length: int):

        self.context_length = context_length

        self.data = np.memmap(
            bin_file,
            dtype=np.uint16,
            mode="r"
        )

    def __len__(self):

        return (len(self.data) - 1) // self.context_length

    def __getitem__(self, idx):

        start = idx * self.context_length

        x = torch.tensor(
            self.data[start: start + self.context_length],
            dtype=torch.long
        )

        y = torch.tensor(
            self.data[start + 1: start + self.context_length + 1],
            dtype=torch.long
        )

        return x, y