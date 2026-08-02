import torch

from model.gpt import GPT
from training.loss import GPTLoss
from torch.utils.data import DataLoader


class GPTEvaluator:
    """
    Evaluate a GPT model on a validation dataset.
    """

    def __init__(
        self,
        model: GPT,
        valid_loader: DataLoader,
        criterion: GPTLoss,
    ) -> None:

        self.model = model
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.device = model.config.device

    def evaluate(self) -> float:

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():

            for inputs, targets in self.valid_loader:

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(inputs)

                loss = self.criterion(
                    logits,
                    targets,
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches