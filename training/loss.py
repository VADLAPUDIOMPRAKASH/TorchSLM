import torch
import torch.nn as nn


class GPTLoss(nn.Module):
    """
    Cross Entropy Loss for GPT Language Modeling.

    Expected Shapes
    ----------------
    Logits:
        (Batch, Sequence, Vocabulary)

    Targets:
        (Batch, Sequence)

    Internally reshapes tensors to

        Logits  -> (Batch * Sequence, Vocabulary)
        Targets -> (Batch * Sequence)

    before computing CrossEntropyLoss.
    """

    def __init__(self):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:

        batch_size, sequence_length, vocab_size = logits.shape

        logits = logits.view(
            batch_size * sequence_length,
            vocab_size
        )

        targets = targets.view(
            batch_size * sequence_length
        )

        loss = self.criterion(
            logits,
            targets
        )

        return loss