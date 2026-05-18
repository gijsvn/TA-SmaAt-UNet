"""
Evaluate the persistence baseline for precipitation nowcasting.

The script computes verification metrics for a persistence forecast and saves
the results in the same format as trained model evaluations.
"""

import argparse
import pathlib
import json

from util.load_dataset import PrecipitationDataModule
from util.evaluate_model import Evaluator

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the persistence baseline for precipitation nowcasting."
    )

    parser.add_argument(
        "--data-file",
        type=pathlib.Path,
        required=True,
        help="Path to the HDF5 precipitation dataset.",
    )

    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/persistence"),
        help="Directory where persistence baseline results are saved.",
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
        help="Evaluation batch size.",
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

    return parser.parse_args()

def get_dataloader(
        data_module: PrecipitationDataModule,
        evaluation_set: str,
    ):
    if evaluation_set == "val":
        data_module.setup("fit")
        return data_module.val_dataloader()

    if evaluation_set == "test":
        data_module.setup("test")
        return data_module.test_dataloader()

    raise ValueError(f"Unknown evaluation set: {evaluation_set}")

def main(args: argparse.Namespace) -> None:
    data_module = PrecipitationDataModule(
        file_path=args.data_file,
        n_input_imgs=args.n_input_imgs,
        n_output_imgs=args.n_output_imgs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction
    )

    dataloader = get_dataloader(
        data_module=data_module,
        evaluation_set=args.evaluation_set,
    )

    evaluator = Evaluator(
        model='persistence',
        dataloader=dataloader
    )

    results = evaluator.compute_metrics(
        rain_thresholds=args.precipitation_thresholds,
    )

    results['parameters'] = {
        "total": 0, 
        "trainable": 0
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_file = args.output_dir / f"{args.evaluation_set}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)