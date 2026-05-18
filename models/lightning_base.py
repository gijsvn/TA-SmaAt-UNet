from typing import Tuple
import torch
from torch import nn, optim
import pytorch_lightning as pl

from models.SmaAt_UNet.model import SmaAt_UNet
from models.TA_SmaAt_UNet.model import TA_SmaAt_UNet

class LightningBaseModel(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        learning_rate: float=1e-3,
        lr_patience: int=5,
        loss: nn.Module=nn.MSELoss(reduction="sum"),
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["backbone"])

        self.backbone = backbone
        self.loss = loss

    def forward(self, model_input: tuple[torch.Tensor]) -> torch.Tensor:
        if isinstance(self.backbone, SmaAt_UNet):
            return self.backbone(model_input)
        elif isinstance(self.backbone, TA_SmaAt_UNet):
            return self.backbone(model_input[0], model_input[1])

    def _get_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (self.loss(y_hat, y) / y.size(0)) # Normalize by batch size

    def training_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int) -> torch.Tensor:
        (x, t), y_true = batch

        if isinstance(self.backbone, SmaAt_UNet):
            y_pred = self((x))
        elif isinstance(self.backbone, TA_SmaAt_UNet):
            y_pred = self((x, t))

        loss = self._get_loss(y_pred, y_true)

        self.log(f"train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int) -> None:
        (x, t), y_true = batch

        if isinstance(self.backbone, SmaAt_UNet):
            y_pred = self((x))
        elif isinstance(self.backbone, TA_SmaAt_UNet):
            y_pred = self((x, t))

        loss = self._get_loss(y_pred, y_true)

        self.log(f"val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int) -> None:
        (x, t), y_true = batch

        if isinstance(self.backbone, SmaAt_UNet):
            y_pred = self((x))
        elif isinstance(self.backbone, TA_SmaAt_UNet):
            y_pred = self((x, t))

        loss = self._get_loss(y_pred, y_true)

        self.log(f"test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self) -> dict:
        opt = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.1,
            patience=self.hparams.lr_patience,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss",
            },
        }