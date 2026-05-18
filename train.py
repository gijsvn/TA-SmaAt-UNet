"""
Train a SmaAt-UNet-based precipitation nowcasting model.

The script saves training logs, model checkpoints, and a JSON configuration file
for each training run.
"""

import argparse
import datetime
import pathlib
import random
import json
import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
import numpy as np

from util.load_dataset import PrecipitationDataModule
from util.callbacks import LossPlotCallback, VisualizationCallback
from models.lightning_base import LightningBaseModel
from models.SmaAt_UNet.model import SmaAt_UNet
from models.TA_SmaAt_UNet.model import TA_SmaAt_UNet

SEED = 42

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a precipitation nowcasting model."
    )

    parser.add_argument(
        "--model",
        choices=["SmaAt-UNet", "TA-SmaAt-UNet"],
        type=str,
        required=True,
        help="Which model architecture to train.",
    )

    parser.add_argument(
        "--data-file",
        type=pathlib.Path,
        required=True,
        help="Path to the HDF5 precipitation dataset.",
    )

    parser.add_argument(
        "--result-log-dir",
        type=pathlib.Path,
        default=pathlib.Path("results"),
        help="Directory where logs, checkpoints, and config files are saved.",
    )

    parser.add_argument(
        "--n-input-imgs",
        type=int,
        default=18,
        help="Number of input precipitation frames.",
    )

    parser.add_argument(
        "--n-output-imgs",
        type=int,
        default=12,
        help="Number of output precipitation frames.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of the training data used for validation.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate.",
    )

    parser.add_argument(
        "--lr-patience",
        type=int,
        default=4,
        help="Patience for the learning-rate scheduler.",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=300,
        help="Maximum number of training epochs.",
    )

    parser.add_argument(
        "--es-patience",
        type=int,
        default=15,
        help="Early stopping patience.",
    )

    parser.add_argument(
        "--loss",
        type=str,
        choices=["mse", "l1"],
        default="mse",
        help="Loss function to use.",
    )

    parser.add_argument(
        "--loss-reduction",
        type=str,
        choices=["mean", "sum", "none"],
        default="sum",
        help="Reduction method used by the loss function.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )

    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Logging frequency in training steps.",
    )

    parser.add_argument(
        "--visualization-indices",
        type=int,
        nargs="+",
        default=[50, 150, 333],
        help="Validation sample indices visualized during training.",
    )

    return parser.parse_args()

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    seed_everything(seed, workers=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_data_module(args: argparse.Namespace) -> PrecipitationDataModule:
    return PrecipitationDataModule(
        file_path=args.data_file,
        n_input_imgs=args.n_input_imgs,
        n_output_imgs=args.n_output_imgs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
    )

def build_model(
        args: argparse.Namespace,
        loss: torch.nn.Module,
    ) -> LightningBaseModel:

    if args.model == "SmaAt-UNet":
        backbone = SmaAt_UNet(
            in_channels=args.n_input_imgs,
            out_channels=args.n_output_imgs,
            kernels_per_layer=2,
            bilinear=True,
            reduction_ratio=16,
        )
    elif args.model == "TA-SmaAt-UNet":
        backbone = TA_SmaAt_UNet(
            in_channels=args.n_input_imgs,
            out_channels=args.n_output_imgs,
            kernels_per_layer=2,
            bilinear=True,
            reduction_ratio=16,
        )

    model = LightningBaseModel(
        backbone=backbone,
        learning_rate=args.learning_rate,
        lr_patience=args.lr_patience,
        loss=loss
    )

    return model

def create_logger(result_log_dir: pathlib.Path, model_name: str) -> CSVLogger:
    result_log_dir.mkdir(parents=True, exist_ok=True)

    run_nr = 1
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    logging_file_name = f"{model_name}_{date_str}_run-{run_nr}"

    while (result_log_dir / logging_file_name).exists():
        run_nr += 1
        logging_file_name = f"{model_name}_{date_str}_run-{run_nr}"

    return CSVLogger(
        save_dir=result_log_dir,
        name=logging_file_name,
        version=".",
    )

def save_config(
        args: argparse.Namespace,
        log_dir: pathlib.Path,
        loss: torch.nn.Module
    ) -> None:
    config_dict = {
        "model": args.model,
        "data_file": str(args.data_file),
        "result_log_dir": str(args.result_log_dir),
        "n_input_imgs": args.n_input_imgs,
        "n_output_imgs": args.n_output_imgs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_fraction": args.val_fraction,
        "learning_rate": args.learning_rate,
        "lr_patience": args.lr_patience,
        "max_epochs": args.max_epochs,
        "es_patience": args.es_patience,
        "loss": loss._get_name(),
        "loss_reduction": args.loss_reduction,
        "seed": args.seed
    }

    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)

def main(args: argparse.Namespace) -> None:
    if args.loss == "mse":
        loss = torch.nn.MSELoss(reduction=args.loss_reduction)
    elif args.loss == "l1":
        loss = torch.nn.L1Loss(reduction=args.loss_reduction)

    data_module = build_data_module(args)
    model = build_model(args, loss)

    logger = create_logger(
        result_log_dir=args.result_log_dir,
        model_name=model.backbone.name,
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        filename="{epoch}-{val_loss:.5f}",
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.es_patience,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    loss_plotter = LossPlotCallback()
    prediction_visualizer = VisualizationCallback(
        val_indices=args.visualization_indices
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
        callbacks=[
            loss_plotter,
            prediction_visualizer,
            checkpoint,
            early_stop_cb,
            lr_monitor,
        ],
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=True,
    )

    log_dir = pathlib.Path(trainer.logger.log_dir)
    save_config(
        args=args,
        log_dir=log_dir,
        loss=loss
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)