from typing import Tuple
import pathlib
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
import h5py
import numpy as np

from models.TA_SmaAt_UNet.util import timestamp_to_vector

class PrecipitationDataset(Dataset):
    def __init__(
            self, 
            file_path: pathlib.Path, 
            n_input_imgs: int, 
            n_output_imgs: int, 
            test: bool=False
        ) -> None:
        super().__init__()
        
        self.file_path = file_path
        self.n_input_imgs = n_input_imgs
        self.n_output_imgs = n_output_imgs
        self.sequence_len = self.n_input_imgs + self.n_output_imgs
        self.dataset_type = "test" if test else "train"

        self.data = None

        with h5py.File(self.file_path, "r") as f:
            self.n_samples = f[self.dataset_type]['images'].shape[0]
            self.timestamps = f[self.dataset_type]['timestamps'][:,self.sequence_len-1,0]
            self.time_vectors = np.array([timestamp_to_vector(timestamp) for timestamp in self.timestamps])

    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.tensor]:
        # Open file if not open already
        if self.data is None:
            self.data = h5py.File(self.file_path, "r")[self.dataset_type]

        input_imgs = torch.from_numpy(self.data['images'][idx])[:self.n_input_imgs]
        temp_feature_vector = torch.from_numpy(self.time_vectors[idx])
        output_imgs = torch.from_numpy(self.data['images'][idx])[self.n_input_imgs:self.sequence_len]

        return (input_imgs, temp_feature_vector), output_imgs

class PrecipitationDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            file_path: pathlib.Path, 
            n_input_imgs: int, 
            n_output_imgs: int,
            batch_size: int=8,
            num_workers: int=4,
            val_fraction: float=0.1
        ) -> None:
        super().__init__()

        self.file_path = file_path
        self.n_input_imgs = n_input_imgs
        self.n_output_imgs = n_output_imgs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction

    def setup(self, stage: str) -> None:
        if stage in (None, "fit"):
            full_train = PrecipitationDataset(
                file_path=self.file_path,
                n_input_imgs=self.n_input_imgs,
                n_output_imgs=self.n_output_imgs,
                test=False
            )
            
            num_train = len(full_train)
            indices = np.arange(num_train)

            angles = np.arctan2(full_train.time_vectors[:,0], full_train.time_vectors[:,1])
            angles = (angles + 2.0 * np.pi) % (2.0 * np.pi)

            # Discretize into "pseudo-months" (12 bins around the year)
            num_bins = 12
            month_bins = (angles / (2.0 * np.pi) * num_bins).astype(int)
            month_bins = np.clip(month_bins, 0, num_bins - 1)

            train_indices = []
            val_indices = []

            rng = np.random.default_rng(42)
            for b in range(num_bins):
                bin_idx = indices[month_bins == b]
                n_bin = bin_idx.size
                if n_bin == 0:
                    continue

                n_val_bin = int(np.floor(self.val_fraction * n_bin))

                if n_val_bin <= 0:
                    train_indices.extend(bin_idx.tolist())
                    continue

                if n_val_bin >= n_bin:
                    val_indices.extend(bin_idx.tolist())
                    continue

                bin_idx = bin_idx.copy()
                rng.shuffle(bin_idx)

                val_idx_bin = bin_idx[:n_val_bin]
                train_idx_bin = bin_idx[n_val_bin:]

                val_indices.extend(val_idx_bin.tolist())
                train_indices.extend(train_idx_bin.tolist())

            train_indices = np.array(train_indices, dtype=np.int64)
            val_indices = np.array(val_indices, dtype=np.int64)

            self.train_ds = Subset(full_train, train_indices)

            val = PrecipitationDataset(
                file_path=self.file_path,
                n_input_imgs=self.n_input_imgs,
                n_output_imgs=self.n_output_imgs,
                test=False
            )

            self.val_ds = Subset(val, val_indices)

        if stage in (None, "test"):
            # Load test data
            self.test_ds = PrecipitationDataset(
                file_path=self.file_path,
                n_input_imgs=self.n_input_imgs,
                n_output_imgs=self.n_output_imgs,
                test=True,
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False,
            pin_memory=True,
        )