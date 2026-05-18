"""
Run the TA-SmaAt-UNet layer-conductance analysis.

Default behavior reproduces the layer-conductance figure from the paper:
  * attribution target: mean predicted rainfall over pixels predicted >= 0.5 mm/h
  * forecast leads: all model output leads
  * baselines: zero radar input plus temporal vectors sampled from the training set
  * reported quantity: relative absolute conductance grouped into
    encoder, CBAM, temporal-conditioning, and decoder modules
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import LayerConductance
from tqdm import tqdm

from util.load_dataset import PrecipitationDataModule
from models.lightning_base import LightningBaseModel
from models.TA_SmaAt_UNet.model import TA_SmaAt_UNet


MMH_CONVERSION = 47.83 * 12
MODULE_ORDER = ("encoder", "CBAM", "TA", "decoder")
PLOT_ORDER = ("encoder", "CBAM", "TA", "decoder")
MODULE_DISPLAY_NAMES = {
    "encoder": "Encoder",
    "CBAM": "CBAM",
    "TA": "Temp. Cond.",
    "decoder": "Decoder",
}
MODULE_COLORS = {
    "encoder": "#FF4040",
    "CBAM": "#E8C547",
    "TA": "#FF79B3",
    "decoder": "#76CC6A",
}
REGION_CHOICES = ("pred_pos", "gt_pos", "tp", "fp", "fn")
REGION_LABELS = {
    "pred_pos": "Pred",
    "gt_pos": "Observed",
    "tp": "TP",
    "fp": "FP",
    "fn": "FN",
}


@dataclass(frozen=True)
class ComponentSpec:
    """A named model component and the module-level group it belongs to."""

    name: str
    group: str
    layer: nn.Module


@dataclass(frozen=True)
class AnalysisConfig:
    data_file: str
    checkpoint: str
    output_dir: str
    evaluation_set: str
    n_input_imgs: int
    n_output_imgs: int
    num_workers: int
    threshold_mmh: float
    region: str
    reduction: str
    mmh_conversion: float
    sample_stride: int
    max_samples: Optional[int]
    lead_indices: str
    n_steps: int
    n_temporal_baselines: int
    temporal_baseline_seed: int
    attribute_to_layer_input: bool
    internal_batch_size: Optional[int]
    device: str
    save_component_rows: bool
    skip_analysis: bool
    skip_plot: bool
    figure_formats: List[str]
    plot_dpi: int
    show_plot: bool


class RainRegionTarget(nn.Module):
    """
    Converts a TA-SmaAt-UNet prediction into a scalar attribution target.

    The default target is the mean predicted rainfall at a single forecast lead
    over pixels where the model predicts rainfall above a threshold. This is the
    target used for the manuscript figure. Other region definitions are provided
    only to make nearby follow-up experiments easy.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        threshold_mmh: float,
        region: str,
        lead_idx: int,
        reduction: str,
        mmh_conversion: float,
    ) -> None:
        super().__init__()
        if region not in REGION_CHOICES:
            raise ValueError(f"Unknown region '{region}'. Expected one of {REGION_CHOICES}.")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'.")

        self.model = model
        self.threshold_mmh = float(threshold_mmh)
        self.region = region
        self.lead_idx = int(lead_idx)
        self.reduction = reduction
        self.mmh_conversion = float(mmh_conversion)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x, t)
        return self._reduce_to_scalar(y_hat, y_true)

    def _select_lead(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if y_hat.ndim != 4 or y_true.ndim != 4:
            raise ValueError(
                "Expected y_hat and y_true with shape [B, T, H, W]; "
                f"got y_hat={tuple(y_hat.shape)}, y_true={tuple(y_true.shape)}."
            )
        if y_hat.shape[1] != y_true.shape[1]:
            raise ValueError(
                f"Prediction/target lead mismatch: y_hat has {y_hat.shape[1]} leads, "
                f"y_true has {y_true.shape[1]} leads."
            )
        if not 0 <= self.lead_idx < y_hat.shape[1]:
            raise IndexError(f"lead_idx={self.lead_idx} is out of range for {y_hat.shape[1]} leads.")

        lead_slice = slice(self.lead_idx, self.lead_idx + 1)
        return y_hat[:, lead_slice], y_true[:, lead_slice]

    def region_mask(self, y_hat_lead: torch.Tensor, y_true_lead: torch.Tensor) -> torch.Tensor:
        """Return a float mask for the configured region at one forecast lead."""
        y_hat_mmh = y_hat_lead * self.mmh_conversion
        y_true_mmh = y_true_lead * self.mmh_conversion
        pred_pos = y_hat_mmh >= self.threshold_mmh
        true_pos = y_true_mmh >= self.threshold_mmh

        if self.region == "pred_pos":
            mask = pred_pos
        elif self.region == "gt_pos":
            mask = true_pos
        elif self.region == "tp":
            mask = pred_pos & true_pos
        elif self.region == "fp":
            mask = pred_pos & (~true_pos)
        elif self.region == "fn":
            mask = (~pred_pos) & true_pos
        else:  # pragma: no cover; guarded in __init__
            raise RuntimeError(f"Unhandled region: {self.region}")
        return mask.float()

    def _reduce_to_scalar(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_hat_lead, y_true_lead = self._select_lead(y_hat, y_true)
        mask = self.region_mask(y_hat_lead, y_true_lead)
        masked_sum = (y_hat_lead * mask).sum(dim=(1, 2, 3))

        if self.reduction == "sum":
            return masked_sum

        pixel_count = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        return masked_sum / pixel_count


def parse_args() -> AnalysisConfig:
    parser = argparse.ArgumentParser(
        description="Compute module-level layer conductance for TA-SmaAt-UNet and plot conductance by lead time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-file")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", default=f"./layer_conductance_outputs/ta_smaat_unet_by_lead")
    parser.add_argument("--evaluation-set", choices=("val", "test"), default="test")
    parser.add_argument("--n-input-imgs", type=int, default=18)
    parser.add_argument("--n-output-imgs", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--threshold-mmh", type=float, default=0.5, help="Rain-rate threshold used to define the attribution region.")
    parser.add_argument("--region", choices=REGION_CHOICES, default="pred_pos", help="Pixels used for the scalar attribution target.")
    parser.add_argument("--reduction", choices=("mean", "sum"), default="mean", help="How predicted rainfall is reduced over the selected pixels.")
    parser.add_argument("--mmh-conversion", type=float, default=MMH_CONVERSION, help="Factor converting normalized model output to mm/h.")

    parser.add_argument("--sample-stride", type=int, default=1, help="Analyze every Nth sample from the evaluation set.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on analyzed samples after applying sample stride.")
    parser.add_argument(
        "--lead-indices",
        default="all",
        help="0-based forecast lead indices to analyze. Use 'all', a comma list such as '0,1,2', or a Python-style range such as '0:12:2'.",
    )
    parser.add_argument("--n-steps", type=int, default=25, help="Integrated-gradient steps used by Captum LayerConductance.")
    parser.add_argument("--n-temporal-baselines", type=int, default=5, help="Number of temporal baseline vectors sampled from the training set.")
    parser.add_argument("--temporal-baseline-seed", type=int, default=42)
    parser.add_argument(
        "--attribute-to-layer-input",
        action="store_true",
        help="Attribute to layer inputs instead of layer outputs. The paper figure uses layer outputs.",
    )
    parser.add_argument(
        "--internal-batch-size",
        type=int,
        default=None,
        help="Optional Captum internal batch size for path integration.",
    )
    parser.add_argument("--device", default="auto", help="Use 'auto', 'cpu', 'cuda', or a torch device string such as 'cuda:0'.")

    parser.add_argument("--save-component-rows", action="store_true", help="Save per-sample component conductance rows in addition to module rows.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip conductance computation and only plot an existing module_summary.csv.")
    parser.add_argument("--skip-plot", action="store_true", help="Run analysis but do not create the figure.")
    parser.add_argument("--figure-formats", nargs="+", default=["png", "pdf"], help="Figure formats to save.")
    parser.add_argument("--plot-dpi", type=int, default=200)
    parser.add_argument("--show-plot", action="store_true")

    args = parser.parse_args()
    if args.sample_stride < 1:
        parser.error("--sample-stride must be >= 1")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be >= 1 when provided")
    if args.n_steps < 1:
        parser.error("--n-steps must be >= 1")
    if args.n_temporal_baselines < 1:
        parser.error("--n-temporal-baselines must be >= 1")

    return AnalysisConfig(**vars(args))


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_lead_indices(spec: str, n_leads: int) -> List[int]:
    if spec.lower() == "all":
        return list(range(n_leads))

    indices: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            parts = token.split(":")
            if len(parts) > 3:
                raise ValueError(f"Invalid lead range '{token}'. Use start:stop[:step].")
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1]) if len(parts) >= 2 and parts[1] else n_leads
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            indices.extend(range(start, stop, step))
        else:
            indices.append(int(token))

    unique_sorted = sorted(set(indices))
    bad = [idx for idx in unique_sorted if idx < 0 or idx >= n_leads]
    if bad:
        raise ValueError(f"Lead indices out of range for {n_leads} model outputs: {bad}")
    if not unique_sorted:
        raise ValueError("No lead indices were selected.")
    return unique_sorted


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, defaultdict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_json(data: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(data), f, indent=2)


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(w)
    if not np.any(mask):
        return None
    v = v[mask]
    w = w[mask]
    if np.isclose(w.sum(), 0.0):
        return float(v.mean())
    return float(np.average(v, weights=w))


def count_trainable_params(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def build_model(config: AnalysisConfig, device: torch.device) -> LightningBaseModel:
    backbone = TA_SmaAt_UNet(
        in_channels=config.n_input_imgs,
        out_channels=config.n_output_imgs,
        bilinear=True,
        reduction_ratio=16,
    )
    model = LightningBaseModel.load_from_checkpoint(
        checkpoint_path=config.checkpoint,
        backbone=backbone,
        strict=True,
        weights_only=False
    )
    return model.to(device).eval()


def build_datamodule(config: AnalysisConfig) -> PrecipitationDataModule:
    return PrecipitationDataModule(
        file_path=config.data_file,
        n_input_imgs=config.n_input_imgs,
        n_output_imgs=config.n_output_imgs,
        num_workers=config.num_workers
    )


def get_dataset(data_module: PrecipitationDataModule, evaluation_set: str):
    if evaluation_set == "val":
        return data_module.val_ds
    if evaluation_set == "test":
        return data_module.test_ds
    raise ValueError("evaluation_set must be 'val' or 'test'.")


def get_components(model: LightningBaseModel) -> List[ComponentSpec]:
    """Return the exact module grouping used for the manuscript analysis."""
    backbone = model.backbone
    specs = [
        ("inc", "encoder"),
        ("down1", "encoder"),
        ("down2", "encoder"),
        ("down3", "encoder"),
        ("down4", "encoder"),
        ("cbam1", "CBAM"),
        ("cbam2", "CBAM"),
        ("cbam3", "CBAM"),
        ("cbam4", "CBAM"),
        ("cbam5", "CBAM"),
        ("cond1", "TA"),
        ("cond2", "TA"),
        ("cond3", "TA"),
        ("cond4", "TA"),
        ("cond5", "TA"),
        ("up1", "decoder"),
        ("up2", "decoder"),
        ("up3", "decoder"),
        ("up4", "decoder"),
        ("outc", "decoder"),
    ]
    components: List[ComponentSpec] = []
    missing: List[str] = []
    for name, group in specs:
        if not hasattr(backbone, name):
            missing.append(name)
        else:
            components.append(ComponentSpec(name=name, group=group, layer=getattr(backbone, name)))
    if missing:
        raise AttributeError(
            "The loaded backbone is missing expected TA-SmaAt-UNet modules: "
            + ", ".join(missing)
        )
    return components


def parameter_group_counts(components: Sequence[ComponentSpec]) -> Dict[str, int]:
    counts = {group: 0 for group in MODULE_ORDER}
    for component in components:
        counts[component.group] += count_trainable_params(component.layer)
    counts["total"] = sum(counts.values())
    return counts


def parameter_group_fractions(counts: Mapping[str, int]) -> Dict[str, float]:
    total = counts.get("total", 0)
    if total <= 0:
        return {group: 0.0 for group in MODULE_ORDER}
    return {group: counts[group] / total for group in MODULE_ORDER}


def infer_n_leads(model: LightningBaseModel, dataset: Any, device: torch.device) -> int:
    (x, t), y = dataset[0]
    x = x.unsqueeze(0).to(device)
    t = t.unsqueeze(0).to(device)
    with torch.no_grad():
        y_hat = model.backbone(x, t)
    if y_hat.ndim != 4:
        raise ValueError(f"Expected model output [B, T, H, W], got {tuple(y_hat.shape)}.")
    return int(y_hat.shape[1])


def sample_temporal_baselines(
    data_module: PrecipitationDataModule,
    *,
    n_baselines: int,
    seed: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[int], List[List[float]]]:
    """Sample temporal baseline vectors from the training set."""
    data_module.setup("fit")
    train_ds = data_module.train_ds
    if train_ds is None or len(train_ds) == 0:
        raise RuntimeError("Training dataset is unavailable or empty; cannot sample temporal baselines.")

    rng = random.Random(seed)
    n_to_sample = min(n_baselines, len(train_ds))
    indices = rng.sample(range(len(train_ds)), k=n_to_sample)

    baselines: List[torch.Tensor] = []
    vectors: List[List[float]] = []
    for idx in indices:
        (x, t), y = train_ds[idx]
        baselines.append(t.unsqueeze(0).to(device))
        vectors.append([float(v) for v in t.tolist()])
    return baselines, indices, vectors


def flatten_attribution(attr: Any) -> torch.Tensor:
    if isinstance(attr, tuple):
        parts = [a.reshape(-1) for a in attr if a is not None]
        if not parts:
            raise RuntimeError("LayerConductance returned an empty attribution tuple.")
        return torch.cat(parts)
    return attr.reshape(-1)


def layer_conductance_stats(
    conductor: LayerConductance,
    *,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    temporal_baselines: Sequence[torch.Tensor],
    n_steps: int,
    attribute_to_layer_input: bool,
    internal_batch_size: Optional[int],
) -> Dict[str, float]:
    """Average absolute and signed layer conductance over temporal baselines."""
    abs_values: List[float] = []
    signed_values: List[float] = []

    for t_baseline in temporal_baselines:
        kwargs: Dict[str, Any] = {
            "inputs": (x, t),
            "baselines": (torch.zeros_like(x), t_baseline),
            "additional_forward_args": (y,),
            "n_steps": n_steps,
            "attribute_to_layer_input": attribute_to_layer_input,
        }
        if internal_batch_size is not None:
            kwargs["internal_batch_size"] = internal_batch_size

        attr = conductor.attribute(**kwargs)
        flat_attr = flatten_attribution(attr).detach()
        abs_values.append(float(flat_attr.abs().mean().item()))
        signed_values.append(float(flat_attr.mean().item()))

    return {
        "mean_abs": float(np.mean(abs_values)),
        "std_abs": float(np.std(abs_values)),
        "mean_signed": float(np.mean(signed_values)),
        "std_signed": float(np.std(signed_values)),
    }


def selected_sample_indices(dataset_size: int, *, sample_stride: int, max_samples: Optional[int]) -> List[int]:
    indices = list(range(0, dataset_size, sample_stride))
    if max_samples is not None:
        indices = indices[:max_samples]
    return indices


def summarize_module_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Create one summary row per lead from compact per-sample module rows."""
    by_lead: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_lead[int(row["lead_idx"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for lead_idx in sorted(by_lead):
        lead_rows = by_lead[lead_idx]
        weights = [float(r["region_pixel_count"]) for r in lead_rows]
        summary: Dict[str, Any] = {
            "lead_idx": lead_idx,
            "forecast_minutes": int(5 * (lead_idx + 1)),
            "threshold_mmh": float(lead_rows[0]["threshold_mmh"]),
            "region": lead_rows[0]["region"],
            "n_samples": len(lead_rows),
            "weighted_total_region_pixels": float(np.sum(weights)),
            "mean_region_pixel_count": float(np.mean(weights)) if weights else 0.0,
        }
        for module in MODULE_ORDER:
            summary[f"{module}_mean_abs"] = weighted_mean(
                [float(r[f"{module}_mean_abs"]) for r in lead_rows], weights
            )
            summary[f"{module}_mean_signed"] = weighted_mean(
                [float(r[f"{module}_mean_signed"]) for r in lead_rows], weights
            )
            summary[f"{module}_mean_abs_share"] = weighted_mean(
                [float(r[f"{module}_abs_share"]) for r in lead_rows], weights
            )
        summary_rows.append(summary)
    return summary_rows


def summarize_component_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Create one summary row per lead and component."""
    grouped: Dict[Tuple[int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["lead_idx"]), str(row["component"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (lead_idx, component), comp_rows in sorted(grouped.items()):
        weights = [float(r["region_pixel_count"]) for r in comp_rows]
        first = comp_rows[0]
        out.append(
            {
                "lead_idx": lead_idx,
                "forecast_minutes": int(5 * (lead_idx + 1)),
                "threshold_mmh": float(first["threshold_mmh"]),
                "region": first["region"],
                "module": first["module"],
                "component": component,
                "mean_abs": weighted_mean([float(r["mean_abs"]) for r in comp_rows], weights),
                "mean_signed": weighted_mean([float(r["mean_signed"]) for r in comp_rows], weights),
                "abs_share_of_total": weighted_mean([float(r["abs_share_of_total"]) for r in comp_rows], weights),
                "abs_share_within_module": weighted_mean([float(r["abs_share_within_module"]) for r in comp_rows], weights),
            }
        )
    return out


def run_analysis(config: AnalysisConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)

    data_module = build_datamodule(config)
    temporal_baselines, temporal_baseline_indices, temporal_baseline_vectors = sample_temporal_baselines(
        data_module,
        n_baselines=config.n_temporal_baselines,
        seed=config.temporal_baseline_seed,
        device=device,
    )

    data_module.setup("fit" if config.evaluation_set == "val" else "test")
    dataset = get_dataset(data_module, config.evaluation_set)
    if dataset is None or len(dataset) == 0:
        raise RuntimeError(f"No samples found for evaluation set '{config.evaluation_set}'.")

    model = build_model(config, device)
    components = get_components(model)
    param_counts = parameter_group_counts(components)
    param_fracs = parameter_group_fractions(param_counts)

    n_leads = infer_n_leads(model, dataset, device)
    lead_indices = parse_lead_indices(config.lead_indices, n_leads)
    sample_indices = selected_sample_indices(len(dataset), sample_stride=config.sample_stride, max_samples=config.max_samples)

    print("\nLayer-conductance configuration")
    print(f"  device                 : {device}")
    print(f"  evaluation set         : {config.evaluation_set} ({len(dataset)} total samples)")
    print(f"  analyzed samples       : {len(sample_indices)} with stride={config.sample_stride}")
    print(f"  lead indices           : {lead_indices}")
    print(f"  threshold / region     : {config.threshold_mmh:g} mm/h / {config.region}")
    print(f"  temporal baselines     : {len(temporal_baselines)}")
    print("\nTrainable parameters by module group")
    for module in MODULE_ORDER:
        print(f"  {module:<8}: {param_counts[module]:>10,d} ({param_fracs[module]:.4%})")
    print(f"  {'total':<8}: {param_counts['total']:>10,d}")

    metadata = {
        "analysis": "TA-SmaAt-UNet module-level layer conductance",
        "config": asdict(config),
        "n_leads_in_model_output": n_leads,
        "selected_lead_indices": lead_indices,
        "sample_indices": sample_indices,
        "parameter_group_counts": param_counts,
        "parameter_group_fractions": param_fracs,
        "temporal_baseline_indices_in_train_ds": temporal_baseline_indices,
        "temporal_baseline_vectors": temporal_baseline_vectors,
        "module_order": list(MODULE_ORDER),
        "components": [{"name": c.name, "module": c.group} for c in components],
        "target_description": (
            "Predicted rainfall is reduced over the selected pixel region at each forecast lead. "
            "Default region pred_pos selects pixels where the model prediction is at or above the rain threshold."
        ),
        "baselines": "zero radar input plus real temporal encodings sampled from train_ds",
    }
    write_json(metadata, output_dir / "metadata.json")

    conductors: Dict[Tuple[int, str], LayerConductance] = {}
    for lead_idx in lead_indices:
        target_model = RainRegionTarget(
            model=model.backbone,
            threshold_mmh=config.threshold_mmh,
            region=config.region,
            lead_idx=lead_idx,
            reduction=config.reduction,
            mmh_conversion=config.mmh_conversion,
        ).to(device).eval()
        for component in components:
            conductors[(lead_idx, component.name)] = LayerConductance(target_model, component.layer)

    module_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []

    for sample_idx in tqdm(sample_indices, desc="Samples"):
        (x, t), y = dataset[sample_idx]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        t = t.unsqueeze(0).to(device)

        with torch.no_grad():
            y_hat = model.backbone(x, t)
        if y_hat.ndim != 4 or y.ndim != 4:
            raise ValueError(f"Expected y_hat and y to be [B, T, H, W]; got {tuple(y_hat.shape)} and {tuple(y.shape)}.")

        for lead_idx in lead_indices:
            target_for_mask = RainRegionTarget(
                model=model.backbone,
                threshold_mmh=config.threshold_mmh,
                region=config.region,
                lead_idx=lead_idx,
                reduction=config.reduction,
                mmh_conversion=config.mmh_conversion,
            )
            y_hat_lead = y_hat[:, lead_idx : lead_idx + 1]
            y_true_lead = y[:, lead_idx : lead_idx + 1]
            region_mask = target_for_mask.region_mask(y_hat_lead, y_true_lead)
            region_pixel_count = int(region_mask.sum().item())
            if region_pixel_count == 0:
                continue

            component_stats: List[Dict[str, Any]] = []
            module_abs_totals = {module: 0.0 for module in MODULE_ORDER}
            module_signed_totals = {module: 0.0 for module in MODULE_ORDER}

            for component in components:
                model.zero_grad(set_to_none=True)
                stats = layer_conductance_stats(
                    conductors[(lead_idx, component.name)],
                    x=x,
                    t=t,
                    y=y,
                    temporal_baselines=temporal_baselines,
                    n_steps=config.n_steps,
                    attribute_to_layer_input=config.attribute_to_layer_input,
                    internal_batch_size=config.internal_batch_size,
                )
                module_abs_totals[component.group] += stats["mean_abs"]
                module_signed_totals[component.group] += stats["mean_signed"]
                component_stats.append(
                    {
                        "sample_idx": sample_idx,
                        "lead_idx": lead_idx,
                        "forecast_minutes": int(5 * (lead_idx + 1)),
                        "threshold_mmh": config.threshold_mmh,
                        "region": config.region,
                        "region_pixel_count": region_pixel_count,
                        "module": component.group,
                        "component": component.name,
                        **stats,
                    }
                )

            total_abs = float(sum(module_abs_totals.values()))
            for row in component_stats:
                module_abs = module_abs_totals[row["module"]]
                row["abs_share_of_total"] = 0.0 if total_abs <= 0.0 else row["mean_abs"] / total_abs
                row["abs_share_within_module"] = 0.0 if module_abs <= 0.0 else row["mean_abs"] / module_abs
                if config.save_component_rows:
                    component_rows.append(row)

            module_row: Dict[str, Any] = {
                "sample_idx": sample_idx,
                "lead_idx": lead_idx,
                "forecast_minutes": int(5 * (lead_idx + 1)),
                "threshold_mmh": config.threshold_mmh,
                "region": config.region,
                "region_pixel_count": region_pixel_count,
            }
            for module in MODULE_ORDER:
                module_row[f"{module}_mean_abs"] = module_abs_totals[module]
                module_row[f"{module}_mean_signed"] = module_signed_totals[module]
                module_row[f"{module}_abs_share"] = 0.0 if total_abs <= 0.0 else module_abs_totals[module] / total_abs
            module_rows.append(module_row)

    if not module_rows:
        raise RuntimeError(
            "No valid sample/lead pairs were found. Try a lower threshold, a different region, "
            "or a smaller sample stride."
        )

    module_summary_rows = summarize_module_rows(module_rows)
    write_csv(module_rows, output_dir / "sample_module_conductance.csv")
    write_csv(module_summary_rows, output_dir / "module_summary.csv")

    if config.save_component_rows:
        component_summary_rows = summarize_component_rows(component_rows)
        write_csv(component_rows, output_dir / "sample_component_conductance.csv")
        write_csv(component_summary_rows, output_dir / "component_summary.csv")

    print("\nSaved analysis outputs")
    print(f"  {output_dir / 'metadata.json'}")
    print(f"  {output_dir / 'sample_module_conductance.csv'}")
    print(f"  {output_dir / 'module_summary.csv'}")
    if config.save_component_rows:
        print(f"  {output_dir / 'sample_component_conductance.csv'}")
        print(f"  {output_dir / 'component_summary.csv'}")


def plot_summary(config: AnalysisConfig) -> None:
    output_dir = Path(config.output_dir)
    summary_path = output_dir / "module_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot plot because {summary_path} does not exist.")

    df = pd.read_csv(summary_path)
    if df.empty:
        raise RuntimeError(f"{summary_path} is empty.")

    df = df.sort_values("lead_idx")
    threshold = float(df["threshold_mmh"].iloc[0])
    region = str(df["region"].iloc[0])
    region_label = REGION_LABELS.get(region, region)

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "legend.title_fontsize": 15,
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    markers = {"encoder": "o", "CBAM": "s", "TA": "^", "decoder": "D"}
    all_values: List[float] = []

    for module in PLOT_ORDER:
        share_col = f"{module}_mean_abs_share"
        if share_col not in df.columns:
            continue
        x = df["forecast_minutes"].to_numpy(dtype=float)
        y = df[share_col].to_numpy(dtype=float)
        all_values.extend([float(v) for v in y if np.isfinite(v)])
        ax.plot(
            x,
            y,
            marker=markers.get(module, "o"),
            label=MODULE_DISPLAY_NAMES.get(module, module),
            color=MODULE_COLORS.get(module),
        )

    ax.set_xlabel("Nowcasting Time (Minutes)")
    if region == "pred_pos":
        ax.set_ylabel(f"Relative Contribution ({region_label} ≥ {threshold:g} mm/h)")
    else:
        ax.set_ylabel("Relative Module Contribution")
    ax.set_xticks(df["forecast_minutes"].to_numpy(dtype=float))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend()

    if all_values:
        ymin = min(all_values)
        ymax = max(all_values)
        pad = max(0.03, 0.08 * (ymax - ymin))
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)

    fig.tight_layout()

    saved_paths: List[Path] = []
    for fmt in config.figure_formats:
        fmt = fmt.lower().lstrip(".")
        out_path = output_dir / f"layer_conductance_by_lead.{fmt}"
        fig.savefig(out_path, dpi=config.plot_dpi, bbox_inches="tight")
        saved_paths.append(out_path)

    if config.show_plot:
        plt.show()
    else:
        plt.close(fig)

    print("\nSaved figure")
    for path in saved_paths:
        print(f"  {path}")


def main() -> None:
    config = parse_args()
    if not config.skip_analysis:
        run_analysis(config)
    if not config.skip_plot:
        plot_summary(config)


if __name__ == "__main__":
    main()
