"""
Evaluate a trained precipitation nowcasting model.

The script computes and stores verification metrics and qualitative prediction
visualizations for a selected validation or test split.
"""

import argparse
import pathlib
import torch
import json

from util.load_dataset import PrecipitationDataModule
from util.evaluate_model import Evaluator, count_parameters
from models.lightning_base import LightningBaseModel
from models.SmaAt_UNet.model import SmaAt_UNet
from models.TA_SmaAt_UNet.model import TA_SmaAt_UNet

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained precipitation nowcasting model."
    )

    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        required=True,
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--data-file",
        type=pathlib.Path,
        required=True,
        help="Path to the HDF5 precipitation dataset.",
    )

    parser.add_argument(
        "--evaluation-set",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--precipitation-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 10.0, 20.0],
        help="Rainfall thresholds in mm/h used for verification metrics.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size. Defaults to the value stored in config.json.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )

    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory where evaluation outputs are saved. Defaults to the checkpoint folder.",
    )

    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    model_path = args.checkpoint_path
    data_file_path = args.data_file
    evaluation_set = args.evaluation_set
    precipitation_thresholds = args.precipitation_thresholds

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = model_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_path.parent / "config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint: {model_path}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    n_input_imgs = model_config['n_input_imgs']
    n_output_imgs = model_config['n_output_imgs']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_config["model"] == "SmaAt-UNet":
        backbone = SmaAt_UNet(
            in_channels=n_input_imgs,
            out_channels=n_output_imgs,
            kernels_per_layer=2,
            bilinear=True,
            reduction_ratio=16
        )
    elif model_config["model"] == "TA-SmaAt-UNet":
        backbone = TA_SmaAt_UNet(
            in_channels=n_input_imgs,
            out_channels=n_output_imgs,
            kernels_per_layer=2,
            bilinear=True,
            reduction_ratio=16
        )
    else:
        raise ValueError(f"Model type specified in config does not match 'SmaAt-UNet' or 'TA-SmaAt-UNet")

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = model_config.get("batch_size", 16)

    data_module = PrecipitationDataModule(
        file_path=data_file_path,
        n_input_imgs=n_input_imgs,
        n_output_imgs=n_output_imgs,
        batch_size=batch_size,
        num_workers=0,
        val_fraction=0.1
    )

    if evaluation_set == "val":
        data_module.setup("fit")
        dataloader = data_module.val_dataloader()
    elif evaluation_set == "test":
        data_module.setup("test")
        dataloader = data_module.test_dataloader()
    else:
        raise ValueError(f"Unknown evaluation set: {evaluation_set}")

    model = LightningBaseModel.load_from_checkpoint(
        checkpoint_path=model_path,
        backbone=backbone,
        weights_only=False,
        strict=True
    ).to(device)

    evaluator = Evaluator(
        model=model,
        dataloader=dataloader
    )

    results, seasonal_results = evaluator.compute_metrics(
        rain_thresholds=precipitation_thresholds
    )

    total_params, trainable_params = count_parameters(backbone)
    results['parameters'] = {"total": total_params, "trainable": trainable_params}

    results_path = output_dir / f"{evaluation_set}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    seasonal_results_path = output_dir / f"{evaluation_set}_results_seasonal.json"
    with open(seasonal_results_path, "w", encoding="utf-8") as f:
        json.dump(seasonal_results, f, indent=4)
    
    predictions_path = output_dir / f"{evaluation_set}_predictions.png"
    evaluator.visualize_predictions(
        title=str(model_path),
        save_path=predictions_path
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)