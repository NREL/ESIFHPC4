# The MIT License (MIT)
#
# Modifications Copyright (c) 2020-2023 NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import glob
import h5py as h5
import numpy as np
import torch
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
from amd.rocal.plugin.pytorch import ROCALNumpyIterator
from .common import get_datashapes


class CamRocalNumpyDataloader(object):

    def get_pipeline(self, data_files, label_files, num_shards, shard_id):
        pipeline = Pipeline(batch_size=self.batchsize,
                            num_threads=self.num_threads,
                            device_id=self.device.index,
                            seed=self.seed,
                            rocal_cpu=False)

        with pipeline:
            data = fn.readers.numpy(file_root=self.root_dir,
                                    files=[os.path.basename(x) for x in data_files],
                                    num_shards=num_shards,
                                    shard_id=shard_id,
                                    stick_to_shard=self.stick_to_shard,
                                    random_shuffle=self.shuffle,
                                    pad_last_batch=True,
                                    seed=self.seed,
                                    last_batch_policy=types.LAST_BATCH_PARTIAL if self.is_validation
                                                      else types.LAST_BATCH_DROP)

            label = fn.readers.numpy(file_root=self.root_dir,
                                     files=[os.path.basename(x) for x in label_files],
                                     num_shards=num_shards,
                                     shard_id=shard_id,
                                     stick_to_shard=self.stick_to_shard,
                                     random_shuffle=self.shuffle,
                                     pad_last_batch=True,
                                     seed=self.seed,
                                     last_batch_policy=types.LAST_BATCH_PARTIAL if self.is_validation
                                                       else types.LAST_BATCH_DROP)

            # rocAL's normalize and transpose operators currently do not
            # support DeepCAM's 3D HWC NumPy tensors. Keep rocAL responsible
            # for reading directly into GPU memory and perform these two
            # operations with PyTorch in __iter__.
            pipeline.set_outputs(data, label)

        return pipeline


    def init_files(self, root_dir, prefix_data, prefix_label, statsfile, file_list_data=None, file_list_label=None):
        self.root_dir = root_dir
        self.prefix_data = prefix_data
        self.prefix_label = prefix_label

        # get files
        # data
        if file_list_data is not None and os.path.isfile(os.path.join(root_dir, file_list_data)):
            with open(os.path.join(root_dir, file_list_data), "r") as f:
                token = f.readlines()
            self.data_files = sorted([os.path.join(root_dir, x.strip()) for x in token])
        else:
            self.data_files = sorted(glob.glob(os.path.join(self.root_dir, self.prefix_data)))
        # label
        if file_list_label is not None and os.path.isfile(os.path.join(root_dir, file_list_label)):
            with open(os.path.join(root_dir, file_list_label), "r") as f:
                token = f.readlines()
            self.label_files = sorted([os.path.join(root_dir, x.strip()) for x in token])
        else:
            self.label_files = sorted(glob.glob(os.path.join(self.root_dir, self.prefix_label)))

        # Pair files by sample suffix instead of relying on two independent
        # lexicographical sorts.
        data_by_suffix = {}
        for path in self.data_files:
            name = os.path.basename(path)
            if not name.startswith("data-"):
                raise RuntimeError(f"Unexpected DeepCAM data filename: {path}")
            data_by_suffix[name[len("data-"):]] = path

        label_by_suffix = {}
        for path in self.label_files:
            name = os.path.basename(path)
            if not name.startswith("label-"):
                raise RuntimeError(f"Unexpected DeepCAM label filename: {path}")
            label_by_suffix[name[len("label-"):]] = path

        missing_labels = sorted(data_by_suffix.keys() - label_by_suffix.keys())
        missing_data = sorted(label_by_suffix.keys() - data_by_suffix.keys())
        if missing_labels or missing_data:
            raise RuntimeError(
                "DeepCAM data/label files do not match: "
                f"missing_labels={missing_labels[:20]}, "
                f"missing_data={missing_data[:20]}"
            )

        sample_suffixes = sorted(data_by_suffix)
        if not sample_suffixes:
            raise RuntimeError(
                f"No matching DeepCAM NumPy samples found in {root_dir}"
            )

        self.data_files = [
            data_by_suffix[suffix] for suffix in sample_suffixes
        ]
        self.label_files = [
            label_by_suffix[suffix] for suffix in sample_suffixes
        ]

        # get shapes
        self.data_shape, self.label_shape = get_datashapes()

        # open statsfile
        with h5.File(statsfile, "r") as f:
            data_mean = f["climate"]["minval"][...]
            data_stddev = (f["climate"]["maxval"][...] - data_mean)

        # reshape into broadcastable shape: channels last (H, W, C) -> list per channel
        self.data_mean_list = data_mean.astype(np.float32).tolist()
        self.data_stddev_list = data_stddev.astype(np.float32).tolist()

        # clean up old iterator
        if self.iterator is not None:
            del(self.iterator)
            self.iterator = None

        # clean up old pipeline
        if self.pipeline is not None:
            del(self.pipeline)
            self.pipeline = None

        # restrict file list depending on shuffle mode:
        data_files = self.data_files
        label_files = self.label_files
        num_shards = self.num_shards
        shard_id = self.shard_id

        # Match DistributedSampler(drop_last=True): discard the global
        # remainder before sharding so every training rank receives the same
        # number of unique samples instead of padding with duplicates.
        if not self.is_validation and num_shards > 1:
            usable_samples = (
                len(data_files) // num_shards
            ) * num_shards
            data_files = data_files[:usable_samples]
            label_files = label_files[:usable_samples]

        # This is the global sample count actually consumed in this epoch.
        global_size = len(data_files)

        # modify sharding for gpu-local and node-local shuffling
        if (self.shuffle_mode == "gpu") or (self.shuffle_mode == "node"):

            # modifier
            if self.shuffle_mode == "node":
                num_local_ranks = torch.cuda.device_count()
                num_shards = num_shards // num_local_ranks
                local_shard_id = shard_id % num_local_ranks
                shard_id = shard_id // num_local_ranks

            # shard the bulk first
            num_files = len(data_files)
            num_files_per_shard = num_files // num_shards
            shard_start = shard_id * num_files_per_shard
            shard_end = shard_start + num_files_per_shard

            # get the remainder now
            rem_start = num_shards * num_files_per_shard
            rem_end = num_files

            # extract file lists
            # remainder
            data_rem = data_files[rem_start:rem_end]
            label_rem = label_files[rem_start:rem_end]

            # get the bulk
            data_files = data_files[shard_start:shard_end]
            label_files = label_files[shard_start:shard_end]

            # append remainder
            if shard_id < len(data_rem):
                data_files.append(data_rem[shard_id])
                label_files.append(label_rem[shard_id])

        # reset shard ids
        if self.shuffle_mode == "gpu":
            num_shards = 1
            shard_id = 0
        elif self.shuffle_mode == "node":
            num_shards = num_local_ranks
            shard_id = local_shard_id

        # set up pipeline
        self.pipeline = self.get_pipeline(data_files, label_files, num_shards, shard_id)

        # build pipes
        self.global_size = global_size
        self.pipeline.build()

        # init iterator
        self.iterator = ROCALNumpyIterator(self.pipeline,
                                           device="cuda" if not self.device == torch.device("cpu") else "cpu",
                                           device_id=self.device.index if self.device.index is not None else 0)


    def __init__(self, root_dir, prefix_data, prefix_label, statsfile,
                 batchsize, file_list_data=None, file_list_label=None,
                 num_threads=1, device=torch.device("cpu"),
                 num_shards=1, shard_id=0,
                 shuffle_mode=None, oversampling_factor=1,
                 is_validation=False,
                 lazy_init=False, transpose=True, augmentations=None,
                 use_mmap=True, use_odirect=False, read_gpu=False, seed=333):

        # read filenames first
        self.batchsize = batchsize
        self.num_threads = num_threads
        self.device = device
        self.shuffle_mode = shuffle_mode
        self.read_gpu = read_gpu
        self.pipeline = None
        self.iterator = None
        self.lazy_init = lazy_init
        self.transpose = transpose
        self.augmentations = augmentations or []
        if self.augmentations:
            raise NotImplementedError(
                "The rocAL NumPy loader does not currently support "
                f"data augmentations: {self.augmentations}"
            )
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.is_validation = is_validation
        self.seed = seed
        self.epoch_size = 0
        self.oversampling_factor = oversampling_factor

        assert(self.oversampling_factor == 1)

        # shuffle mode:
        if self.shuffle_mode is not None:
            self.shuffle = True
            # rocAL must remain on the rank's assigned shard. With
            # stick_to_shard=False, multiple ranks return overlapping samples.
            self.stick_to_shard = True
        else:
            self.shuffle = False
            self.stick_to_shard = True

        # init files
        self.init_files(root_dir, prefix_data, prefix_label,
                        statsfile, file_list_data, file_list_label)

        # compute epoch_size from file count
        self.epoch_size = self.global_size


    @property
    def shapes(self):
        return self.data_shape, self.label_shape


    def __iter__(self):
        self.iterator.reset()
        for batch in self.iterator:
            # ROCALNumpyIterator returns a list of output tensors
            data = batch[0]

            # rocAL returns DeepCAM data as NHWC. Normalize and transpose on
            # the GPU using the same min/max statistics as the baseline.
            mean = torch.as_tensor(
                self.data_mean_list,
                dtype=data.dtype,
                device=data.device,
            ).view(1, 1, 1, -1)
            stddev = torch.as_tensor(
                self.data_stddev_list,
                dtype=data.dtype,
                device=data.device,
            ).view(1, 1, 1, -1)

            data = (data - mean) / stddev
            if self.transpose:
                data = data.permute(0, 3, 1, 2).contiguous()

            label = batch[1].to(torch.int64)

            yield data, label, ""
