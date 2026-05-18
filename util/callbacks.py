from typing import List
import os
import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from pathlib import Path

from util.visualization import plot_losses, visualize_precipitation_maps
from models.SmaAt_UNet.model import SmaAt_UNet
from models.TA_SmaAt_UNet.model import TA_SmaAt_UNet

class LossPlotCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, model):
        self.train_losses.append(trainer.callback_metrics.get("train_loss").item())
        self.val_losses.append(trainer.callback_metrics.get("val_loss").item())

        epochs = np.arange(len(self.train_losses))

        log_dir = Path(trainer.logger.log_dir)
        plot_dir = log_dir / "loss_plots"
        if not plot_dir.exists():
            os.makedirs(plot_dir)

        plot_losses(
            epochs=epochs, 
            train_losses=self.train_losses, 
            val_losses=self.val_losses, 
            save_path=str(plot_dir / f"full.png"),
            y_scale="linear"
        )

        plot_losses(
            epochs=epochs, 
            train_losses=self.train_losses, 
            val_losses=self.val_losses, 
            save_path=str(plot_dir / f"full_log.png"),
            y_scale="log"
        )

        if len(self.train_losses) > 4:
            length = 2*len(self.train_losses) // 3 if len(self.train_losses) > 10 else len(self.train_losses)-2

            plot_losses(
                epochs=epochs[-length:], 
                train_losses=self.train_losses[-length:], 
                val_losses=self.val_losses[-length:], 
                save_path=str(plot_dir / f"recent.png"),
                y_scale="linear"
            )

            plot_losses(
                epochs=epochs[-length:], 
                train_losses=self.train_losses[-length:], 
                val_losses=self.val_losses[-length:], 
                save_path=str(plot_dir / f"recent_log.png"),
                y_scale="log"
            )

        print("") # Prints new line so progress bar doesn't overwrite  

class VisualizationCallback(Callback):
    def __init__(self, val_indices: List[int]=[0, 21, 42]) -> None:
        super().__init__()

        self.val_indices = val_indices

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, model):
        if trainer.sanity_checking:
            return
        
        xs, ts, ys = [], [], []
        for idx in self.val_indices:
            (x, t), y = trainer.datamodule.val_ds[idx]

            xs.append(x)
            ts.append(t)
            ys.append(y)

        x = torch.stack(xs).to(model.device)
        t = torch.stack(ts).to(model.device)
        y_true = torch.stack(ys).to(model.device)

        if isinstance(model.backbone, SmaAt_UNet):
            y_pred = model((x))
        elif isinstance(model.backbone, TA_SmaAt_UNet):
            y_pred = model((x, t))

        # Select final frame from sequence for visualization
        persistences = x[:, -1, :, :]
        predictions = y_pred[:, -1, :, :]
        ground_truths = y_true[:, -1, :, :]

        precipitation_maps = torch.stack([persistences, predictions, ground_truths]).cpu().numpy()

        log_dir = Path(trainer.logger.log_dir)
        plot_dir = log_dir / "intermediate_predictions"
        if not plot_dir.exists():
            os.makedirs(plot_dir)
        save_path = plot_dir / f"predictions_{trainer.current_epoch}.png"

        visualize_precipitation_maps(
            precipitation_maps=precipitation_maps,
            row_labels=[f"Sample {str(idx)}" for idx in self.val_indices],
            column_labels=['Persistence', 'Prediction', 'Ground Truth'],
            suptitle=f"Epoch {trainer.current_epoch}",
            save_path=str(save_path),
        )