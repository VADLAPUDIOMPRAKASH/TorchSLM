import logging
import time

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.gpt import GPT
from training.loss import GPTLoss

logger = logging.getLogger(__name__)


class GPTTrainer:
    """
    Train a GPT model.
    """

    def __init__(
        self,
        model: GPT,
        train_loader: DataLoader,
        criterion: GPTLoss,
        optimizer: Optimizer,
        evaluator=None,
        checkpoint=None,

    ) -> None:

        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer

        self.evaluator = evaluator
        self.checkpoint = checkpoint

        self.device = model.config.device
        self.global_step = 0

    def train_step(
        self,
        inputs,
        targets,
    ) -> float:
        """
        Perform one optimization step.

        Returns
        -------
        float
            Training loss.
        """

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        logits = self.model(inputs)

        loss = self.criterion(
            logits,
            targets,
        )

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train_epoch(
        self,
        epoch: int,
        start_batch: int = 0,
    ) -> float:
        """
        Train the model for one epoch.

        Parameters
        ----------
        epoch : int
            Number of epochs already fully completed (0-indexed epoch
            currently being trained). Used only for checkpoint labeling.
        start_batch : int
            Batch index to resume from within this epoch (batches before
            this are skipped). Only relevant when resuming mid-epoch.

        Returns
        -------
        float
            Average training loss over the batches actually processed.
        """

        self.model.train()

        total_loss = 0.0
        num_batches = 0

        save_interval = self.model.config.checkpoint_interval

        progress = tqdm(
            self.train_loader,
            desc="Training",
            unit="batch",
        )

        for batch_idx, (inputs, targets) in enumerate(progress):

            if batch_idx < start_batch:
                continue

            loss = self.train_step(
                inputs,
                targets,
            )

            total_loss += loss
            num_batches += 1
            self.global_step += 1

            progress.set_postfix(loss=f"{loss:.4f}")

            # -------------------------
            # Periodic mid-epoch checkpoint
            # -------------------------

            if (
                self.checkpoint is not None
                and save_interval > 0
                and (batch_idx + 1) % save_interval == 0
            ):

                self.checkpoint.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=self.global_step,
                    loss=loss,
                    filename=f"epoch_{epoch + 1}.pt",
                )

                logger.info(
                    "  checkpoint saved (epoch %d, batch %d, global_step %d)",
                    epoch + 1,
                    batch_idx + 1,
                    self.global_step,
                )

        average_loss = total_loss / num_batches if num_batches else 0.0

        return average_loss

    def fit(
        self,
        epochs: int,
        start_epoch: int = 0,
        start_batch: int = 0,
        global_step: int = 0,
    ) -> None:
        """
        Train the model for multiple epochs.

        Parameters
        ----------
        epochs : int
            Number of training epochs.
        start_epoch : int
            Epoch to resume from (0 for a fresh start). Equal to the
            number of epochs already completed, as stored in a checkpoint.
        start_batch : int
            Batch index to resume from within start_epoch (0 for a fresh
            start or when the checkpoint was saved at an epoch boundary).
            Only applied to start_epoch itself, not later epochs.
        global_step : int
            Total optimizer steps already taken, as stored in a checkpoint.
        """

        self.global_step = global_step

        for epoch in range(start_epoch, epochs):

            epoch_start_batch = start_batch if epoch == start_epoch else 0

            # -------------------------
            # Training
            # -------------------------

            start = time.time()

            train_loss = self.train_epoch(
                epoch=epoch,
                start_batch=epoch_start_batch,
            )

            elapsed = time.time() - start

            logger.info("Epoch [%d/%d]", epoch + 1, epochs)
            logger.info("Train Loss : %.4f", train_loss)
            logger.info("Epoch Time : %.2f seconds", elapsed)

            # -------------------------
            # Validation
            # -------------------------

            valid_loss = None

            if self.evaluator is not None:

                valid_loss = self.evaluator.evaluate()

                logger.info("Valid Loss : %.4f", valid_loss)

            # -------------------------
            # Save Checkpoint
            # -------------------------

            if self.checkpoint is not None:

                self.checkpoint.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    batch_idx=-1,
                    global_step=self.global_step,
                    loss=valid_loss if valid_loss is not None else train_loss,
                    filename=f"epoch_{epoch + 1}.pt",
                )

            print("-" * 50)