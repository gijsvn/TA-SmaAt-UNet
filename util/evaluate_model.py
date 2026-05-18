from typing import List, Dict
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from util.visualization import visualize_precipitation_maps
from models.lightning_base import LightningBaseModel
from models.SmaAt_UNet.model import SmaAt_UNet
from models.TA_SmaAt_UNet.model import TA_SmaAt_UNet

MONTH_ABBR_TO_KEY = {
    b"JAN": "jan",
    b"FEB": "feb",
    b"MAR": "mar",
    b"APR": "apr",
    b"MAY": "may",
    b"JUN": "jun",
    b"JUL": "jul",
    b"AUG": "aug",
    b"SEP": "sep",
    b"OCT": "oct",
    b"NOV": "nov",
    b"DEC": "dec",
}

MONTH_KEYS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
]

SEASON_KEYS = ["Winter", "Spring", "Summer", "Autumn"]

SEASON_BY_MONTH_KEY = {
    "dec": "Winter", "jan": "Winter", "feb": "Winter",
    "mar": "Spring", "apr": "Spring", "may": "Spring",
    "jun": "Summer", "jul": "Summer", "aug": "Summer",
    "sep": "Autumn", "oct": "Autumn", "nov": "Autumn",
}

def safe_div_nan(num, den):
    return float(num / den) if den != 0 else np.nan

def safe_div(num, den):
    return float(num / den) if den != 0 else 0.0

def timestamp_to_month_key(timestamp) -> str:
    """
    Extract month from timestamps formatted like:
    DD-MMM-YYYY HH:MM:SS.mmm

    Works with bytes, np.bytes_, bytearray, or str.
    """
    if isinstance(timestamp, str):
        timestamp = timestamp.encode("ascii")
    elif not isinstance(timestamp, (bytes, bytearray)):
        timestamp = bytes(timestamp)

    month_abbr = timestamp[3:6].upper()
    return MONTH_ABBR_TO_KEY[month_abbr]

def count_parameters(model: LightningBaseModel):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

class Evaluator:
    def __init__(
        self, 
        model: LightningBaseModel | str, 
        dataloader: DataLoader,
        distribution_bin_edges: np.ndarray = np.array(
            [0.0, 0.6, 1.2, 2.4, 4.8, 9.6, 19.2, 38.4, 76.8],
            dtype=np.float32
        ) # bin edges for predicted precipitation distribution histogram (in mm/h)
    ) -> None:
        
        self.model = model
        if isinstance(self.model, LightningBaseModel):
            self.model.eval()
            self.device = next(self.model.parameters()).device
        elif self.model != "persistence":
            raise ValueError("Model must be a LightningBaseModel or 'persistence' baseline")

        self.dataloader = dataloader
        self.bin_edges = distribution_bin_edges

    @torch.no_grad()
    def compute_metrics(self, rain_thresholds: List[float]) -> tuple[dict, dict]:
        threshold_metrics = None

        bin_counts = np.zeros(len(self.bin_edges) - 1, dtype=np.int64)
        bin_counts_gt = np.zeros(len(self.bin_edges) - 1, dtype=np.int64)

        mse_sums = None
        mae_sums = None
        total_samples = 0

        seasonal_groups = MONTH_KEYS + SEASON_KEYS

        seasonal_mse_values = {group: [] for group in seasonal_groups}

        seasonal_csi_values = {
            thr: {group: [] for group in seasonal_groups}
            for thr in rain_thresholds
        }

        cur_idx = 0

        # ------------------------------------------------------
        # Iterate over dataset
        # ------------------------------------------------------
        for (x, t), y_true in tqdm(self.dataloader, desc="Evaluating model"):
            if isinstance(self.model, LightningBaseModel):
                x = x.to(self.device)
                t = t.to(self.device)
                y_true = y_true.to(self.device)

                if isinstance(self.model.backbone, SmaAt_UNet):
                    y_pred = self.model((x))
                elif isinstance(self.model.backbone, TA_SmaAt_UNet):
                    y_pred = self.model((x, t))
            elif self.model == "persistence":
                # Repeat last input image across all output horizons
                y_pred = x[:, -1, :, :].unsqueeze(1).repeat(1, y_true.shape[1], 1, 1)

            if y_pred.shape != y_true.shape:
                raise ValueError(f"Shape mismatch: y_pred {y_pred.shape}, y_true {y_true.shape}")

            B, T, H, W = y_pred.shape

            batch_timestamps = self.dataloader.dataset.timestamps[cur_idx:cur_idx + B]
            batch_month_keys = [timestamp_to_month_key(ts) for ts in batch_timestamps]
            cur_idx += B

            # Initialize accumulators
            if threshold_metrics is None:
                threshold_metrics: Dict[float, np.ndarray] = {
                    thr: np.zeros((T, 4), dtype=np.float64) for thr in rain_thresholds
                }
                mse_sums = np.zeros(T, dtype=np.float64)
                mae_sums = np.zeros(T, dtype=np.float64)

            # Denormalize predictions and ground truth to mm/5min
            y_pred_mm_5min = y_pred * 47.83
            y_true_mm_5min = y_true * 47.83

            # Compute MSE and MAE per horizon in original units (mm/5min)
            mse_per_horizon = ((y_pred_mm_5min - y_true_mm_5min) ** 2).mean(dim=(0, 2, 3))   # [T]
            mae_per_horizon = torch.abs(y_pred_mm_5min - y_true_mm_5min).mean(dim=(0, 2, 3)) # [T]

            mse_sums += mse_per_horizon.detach().cpu().numpy() * B
            mae_sums += mae_per_horizon.detach().cpu().numpy() * B
            total_samples += B

            # Extrapolate to mm/hour for threshold-based metrics
            y_pred_mm_hour = y_pred_mm_5min * 12
            y_true_mm_hour = y_true_mm_5min * 12

            # ------------------------------------------------------
            # Update monthly/seasonal metrics
            # ------------------------------------------------------
            # Per-sample MSE and MAE over all horizons and pixels.
            seasonal_sqerr = (y_pred_mm_5min - y_true_mm_5min) ** 2

            seasonal_mse_per_sample = seasonal_sqerr.flatten(start_dim=1).mean(dim=1)

            # Per-sample CSI over all horizons and pixels.
            seasonal_csi_per_threshold = {}

            for thr in rain_thresholds:
                pred_mask = y_pred_mm_hour > thr
                true_mask = y_true_mm_hour > thr

                pred_flat = pred_mask.flatten(start_dim=1)
                true_flat = true_mask.flatten(start_dim=1)

                tp = (true_flat & pred_flat).sum(dim=1).float()
                fp = (~true_flat & pred_flat).sum(dim=1).float()
                fn = (true_flat & ~pred_flat).sum(dim=1).float()

                denom = tp + fp + fn
                csi = tp / denom
                csi[denom == 0] = torch.nan

                seasonal_csi_per_threshold[thr] = csi


            def update_seasonal_group(group_key: str, sample_indices_np: np.ndarray) -> None:
                if len(sample_indices_np) == 0:
                    return

                for sample_i in sample_indices_np:
                    seasonal_mse_values[group_key].append(
                        float(seasonal_mse_per_sample[sample_i].detach().cpu())
                    )

                    for thr in rain_thresholds:
                        csi_value = seasonal_csi_per_threshold[thr][sample_i]
                        if not torch.isnan(csi_value):
                            seasonal_csi_values[thr][group_key].append(
                                float(csi_value.detach().cpu())
                            )


            for month_key in set(batch_month_keys):
                sample_indices_np = np.array(
                    [i for i, mk in enumerate(batch_month_keys) if mk == month_key],
                    dtype=np.int64
                )

                season_key = SEASON_BY_MONTH_KEY[month_key]

                update_seasonal_group(month_key, sample_indices_np)
                update_seasonal_group(season_key, sample_indices_np)

            # Update predicted precipitation distribution histogram
            predicted_values = y_pred_mm_hour.detach().cpu().numpy().ravel()
            counts, _ = np.histogram(predicted_values, bins=self.bin_edges)
            bin_counts += counts

            gt_values = y_true_mm_hour.detach().cpu().numpy().ravel()
            counts, _ = np.histogram(gt_values, bins=self.bin_edges)
            bin_counts_gt += counts

            # Update threshold-based metrics
            for thr in rain_thresholds:
                for h_idx in range(T):
                    y_pred_mask = y_pred_mm_hour[:, h_idx] > thr
                    y_true_mask = y_true_mm_hour[:, h_idx] > thr

                    # 0: tn, 1: fp, 2: fn, 3: tp
                    codes = (y_true_mask.int() * 2 + y_pred_mask.int()).view(-1).cpu().numpy()
                    tn, fp, fn, tp = np.bincount(codes, minlength=4).astype(np.float64)

                    threshold_metrics[thr][h_idx] += np.array([tn, fp, fn, tp], dtype=np.float64)

        # ------------------------------------------------------
        # Process results
        # ------------------------------------------------------
        results = {}

        results["MSE"] = (mse_sums / total_samples).tolist()
        results["MAE"] = (mae_sums / total_samples).tolist()
        results["pred_distribution"] = {
            "model": {
                "bin_edges": self.bin_edges.tolist(),
                "counts": bin_counts.tolist()
            },
            "ground_truth": {
                "bin_edges": self.bin_edges.tolist(),
                "counts": bin_counts_gt.tolist()
            },
        }

        for threshold in threshold_metrics:
            tn = threshold_metrics[threshold][:, 0]
            fp = threshold_metrics[threshold][:, 1]
            fn = threshold_metrics[threshold][:, 2]
            tp = threshold_metrics[threshold][:, 3]

            csi = [safe_div(tp_i, tp_i + fn_i + fp_i) for tp_i, fn_i, fp_i in zip(tp, fn, fp)]
            pod = [safe_div(tp_i, tp_i + fn_i) for tp_i, fn_i in zip(tp, fn)]
            far = [safe_div(fp_i, tp_i + fp_i) for tp_i, fp_i in zip(tp, fp)]

            mcc = []
            for tn_i, fp_i, fn_i, tp_i in zip(tn, fp, fn, tp):
                mcc_den = np.sqrt((tp_i + fp_i) * (tp_i + fn_i) * (tn_i + fp_i) * (tn_i + fn_i))
                mcc.append(safe_div(tp_i * tn_i - fp_i * fn_i, mcc_den))

            results[str(threshold)] = {
                "CSI": csi,
                "POD": pod,
                "FAR": far,
                "MCC": mcc
            }

        def mean_or_nan(values: list[float]) -> float:
            return float(np.mean(values)) if len(values) > 0 else np.nan

        def seasonal_mse(group_key: str) -> float:
            return mean_or_nan(seasonal_mse_values[group_key])

        def seasonal_csi(thr: float, group_key: str) -> float:
            return mean_or_nan(seasonal_csi_values[thr][group_key])

        seasonal_results = {}

        seasonal_results["MSE"] = {
            "per_month": {
                month_key: seasonal_mse(month_key)
                for month_key in MONTH_KEYS
            },
            "per_season": {
                season_key: seasonal_mse(season_key)
                for season_key in SEASON_KEYS
            },
        }

        for threshold in rain_thresholds:
            metric_name = f"CSI (≥ {threshold:g} mm/h)"

            seasonal_results[metric_name] = {
                "per_month": {
                    month_key: seasonal_csi(threshold, month_key)
                    for month_key in MONTH_KEYS
                },
                "per_season": {
                    season_key: seasonal_csi(threshold, season_key)
                    for season_key in SEASON_KEYS
                },
            }

        return results, seasonal_results
    
    @torch.no_grad()
    def visualize_predictions(
        self, 
        indices: List[int]=[222, 444, 777, 1337], 
        title: str|None=None,  
        save_path: str|None=None
    ) -> None:
        xs, ts, ys = [], [], []
        for idx in indices:
            (x, t), y = self.dataloader.dataset[idx]
            xs.append(x)
            ts.append(t)
            ys.append(y)
        
        x = torch.stack(xs).to(self.device)
        t = torch.stack(ts).to(self.device)
        y_true = torch.stack(ys).to(self.device)

        if isinstance(self.model, LightningBaseModel):
            if isinstance(self.model.backbone, SmaAt_UNet):
                y_pred = self.model((x))
            elif isinstance(self.model.backbone, TA_SmaAt_UNet):
                y_pred = self.model((x, t))
            # Seclect final frame from sequence for visualization
            y_pred = y_pred[:, -1, :, :]
        elif self.model == "persistence":
            y_pred = x[:, -1, :, :]

        visualize_precipitation_maps(
            precipitation_maps=torch.stack([x[:, -1, :, :], y_pred, y_true[:, -1, :, :]]).cpu().numpy(),
            row_labels=[f"Sample {idx}" for idx in indices],
            column_labels=["Input (Persistence)", "Prediction", "Ground Truth"],
            suptitle=title,
            save_path=save_path
        )
