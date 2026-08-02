from dataclasses import dataclass
import torch


@dataclass
class GPTConfig:
    # ==========================
    # Model Architecture
    # ==========================

    vocab_size: int = 8000
    max_sequence_length: int = 128

    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    expansion_factor: int = 4

    dropout: float = 0.0
    bias: bool = True

    # ==========================
    # Training Hyperparameters
    # ==========================

    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1


    # Data
    batch_size: int = 32
    epochs: int = 10
    num_workers: int = 0
    checkpoint_interval: int = 2000

    # Device
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

