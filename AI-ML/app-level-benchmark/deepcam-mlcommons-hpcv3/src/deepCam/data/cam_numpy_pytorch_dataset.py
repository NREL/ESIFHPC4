import os
from glob import glob

import h5py
import numpy as np
from torch.utils.data import Dataset


class CamNumpyDataset(Dataset):
    """Plain PyTorch dataset for separate DeepCAM NumPy files."""

    def __init__(
        self,
        source,
        statsfile,
        channels,
        allow_uneven_distribution=False,
        shuffle=False,
        preprocess=True,
        transpose=True,
        augmentations=None,
        num_shards=1,
        shard_id=0,
        seed=12345,
    ):
        self.source = source
        self.channels = channels
        self.preprocess = preprocess
        self.transpose = transpose
        self.augmentations = augmentations or []
        self.rng = np.random.default_rng(seed + shard_id)

        data_files = sorted(glob(os.path.join(source, "data-*.npy")))
        if not data_files:
            raise RuntimeError(f"No data-*.npy files found in {source}")

        all_samples = []
        for data_file in data_files:
            label_name = os.path.basename(data_file).replace(
                "data-", "label-", 1
            )
            label_file = os.path.join(source, label_name)

            if not os.path.isfile(label_file):
                raise RuntimeError(
                    f"Missing label for {data_file}: {label_file}"
                )

            all_samples.append((data_file, label_file))

        if shuffle:
            order = self.rng.permutation(len(all_samples))
            all_samples = [all_samples[index] for index in order]

        self.global_size = len(all_samples)

        if num_shards > 1:
            local_size = self.global_size // num_shards
            start = shard_id * local_size
            end = start + local_size
            samples = all_samples[start:end]

            if allow_uneven_distribution:
                remainder_start = num_shards * local_size
                for index in range(remainder_start, self.global_size):
                    if index % num_shards == shard_id:
                        samples.append(all_samples[index])
            else:
                self.global_size = num_shards * local_size
        else:
            samples = all_samples

        self.samples = samples

        first_data = np.load(self.samples[0][0], mmap_mode="r")
        first_label = np.load(self.samples[0][1], mmap_mode="r")
        self.data_shape = first_data.shape
        self.label_shape = first_label.shape

        with h5py.File(statsfile, "r") as stats:
            data_min = stats["climate"]["minval"][...][channels]
            data_max = stats["climate"]["maxval"][...][channels]

        self.data_shift = data_min.reshape(1, 1, -1).astype(np.float32)
        self.data_scale = (
            1.0 / (data_max - data_min)
        ).reshape(1, 1, -1).astype(np.float32)

        if shard_id == 0:
            print(
                f"Initialized PyTorch NumPy dataset with "
                f"{self.global_size} samples."
            )

    def __len__(self):
        return len(self.samples)

    @property
    def shapes(self):
        return self.data_shape, self.label_shape

    def __getitem__(self, index):
        data_file, label_file = self.samples[index]

        data = np.load(data_file, allow_pickle=False)[..., self.channels]
        label = np.load(label_file, allow_pickle=False).astype(
            np.int64, copy=False
        )

        if self.preprocess:
            data = self.data_scale * (data - self.data_shift)

        if "roll" in self.augmentations:
            shift = int(self.rng.integers(0, data.shape[1]))
            data = np.roll(data, shift, axis=1)
            label = np.roll(label, shift, axis=1)

        if "flip" in self.augmentations and self.rng.random() > 0.5:
            data = np.flip(data, axis=0).copy()
            label = np.flip(label, axis=0).copy()

        if self.transpose:
            data = np.transpose(data, (2, 0, 1))

        data = np.ascontiguousarray(data, dtype=np.float32)
        label = np.ascontiguousarray(label, dtype=np.int64)

        return data, label, data_file
