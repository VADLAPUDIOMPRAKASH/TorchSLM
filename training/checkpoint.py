import os
from pathlib import Path

import torch

from model.gpt import GPT
from torch.optim import Optimizer


class GPTCheckpoint:
    """
    Save and load GPT training checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
    ) -> None:

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        model: GPT,
        optimizer: Optimizer,
        epoch: int,
        loss: float,
        filename: str,
        batch_idx: int = -1,
        global_step: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        epoch : int
            Number of epochs fully completed before this checkpoint.
        batch_idx : int
            Last batch index (0-indexed) completed within the epoch
            currently in progress, or -1 if that epoch finished fully.
        global_step : int
            Total optimizer steps taken since training started.
        """

        checkpoint = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "global_step": global_step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        # "latest.pt" always points at the newest training state, so
        # resuming never has to figure out which epoch_N.pt is newest.
        for target_name in (filename, "latest.pt"):

            self._atomic_save(checkpoint, target_name)

    def _atomic_save(
        self,
        checkpoint: dict,
        filename: str,
    ) -> None:
        """
        Write to a temporary file, then rename over the target.

        os.replace is atomic on both POSIX and Windows (same volume), so a
        crash or kill mid-write leaves either the old file intact or the
        new one fully written -- never a half-written, corrupted file.
        """

        target_path = self.checkpoint_dir / filename
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")

        torch.save(checkpoint, tmp_path)

        os.replace(tmp_path, target_path)

    def exists(
        self,
        filename: str = "latest.pt",
    ) -> bool:

        return (self.checkpoint_dir / filename).exists()

    def load(
        self,
        model: GPT,
        optimizer: Optimizer,
        filename: str,
    ) -> tuple[int, int, int, float]:
        """
        Returns
        -------
        epoch, batch_idx, global_step, loss
            batch_idx and global_step default to -1 and 0 respectively
            when loading a checkpoint saved before mid-epoch resume
            support was added.
        """

        checkpoint = torch.load(
            self.checkpoint_dir / filename,
            map_location=model.config.device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        epoch = checkpoint["epoch"]
        batch_idx = checkpoint.get("batch_idx", -1)
        global_step = checkpoint.get("global_step", 0)
        loss = checkpoint["loss"]

        return epoch, batch_idx, global_step, loss