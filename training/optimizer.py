from torch.optim import AdamW

from model.gpt import GPT
from model.config import GPTConfig


class GPTOptimizer:
    """
    Creates an AdamW optimizer for GPT training.
    """

    def __init__(
        self,
        model: GPT,
        config: GPTConfig,
    ) -> None:

        self.optimizer = AdamW(
            params=model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    def get_optimizer(self) -> AdamW:
        """
        Returns the configured AdamW optimizer.
        """
        return self.optimizer