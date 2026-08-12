# DeepCAM rocAL NumPy Loader

This integration adds two NumPy data paths:

- `pytorch-numpy`: portable PyTorch reference loader.
- `rocal-numpy`: accelerated rocAL loader.

The rocAL path has been validated with separate DeepCAM NumPy files on one
eight-GPU AMD Instinct MI250 node.

## Dataset layout

The dataset root must contain:

```text
stats.h5
train/
  data-*.npy
  label-*.npy
validation/
  data-*.npy
  label-*.npy
```

Each data file must have shape `(768, 1152, 16)` in HWC layout. Each label
file must have shape `(768, 1152)`.

Data and label files are paired by their matching filename suffix.

## Requirements

The `pytorch-numpy` loader requires PyTorch and NumPy.

The `rocal-numpy` loader additionally requires a rocAL installation with its
Python module and shared libraries available through `PYTHONPATH` and
`LD_LIBRARY_PATH`.

NVIDIA DALI remains optional and is needed only when a `dali-*` data format is
selected.

## MI250 precision

Use BF16 automatic mixed precision on AMD Instinct MI250:

```text
--precision_mode amp-bf16
```

FP16 AMP produced non-finite BatchNorm running statistics in the tested
configuration. BF16 completed three epochs with finite validation loss and
accuracy.

## Example

Run DeepCAM with rocAL across eight GPUs:

```bash
mpirun --allow-run-as-root --bind-to none -np 8 bash -c '
  export HIP_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}
  export OMP_NUM_THREADS=8

  python3 src/deepCam/train.py \
    --wireup_method nccl-openmpi \
    --run_tag mi250_rocal_bf16 \
    --output_dir /workspace/results/mi250-rocal-bf16 \
    --data_dir_prefix /data/deepcam_npy_dataset \
    --data_format rocal-numpy \
    --data_num_threads 1 \
    --local_batch_size 1 \
    --local_batch_size_validation 1 \
    --optimizer Adam \
    --precision_mode amp-bf16 \
    --max_epochs 1 \
    --min_epochs 1 \
    --target_iou 2.0 \
    --logging_frequency 500 \
    --save_frequency 0 \
    --disable_tuning
'
```

Run the reference loader by changing:

```text
--data_format pytorch-numpy
```

## MI250 single-node results

One-epoch, eight-GPU BF16 measurements:

| Loader | Throughput | Validation time | Epoch time | Validation loss |
| --- | ---: | ---: | ---: | ---: |
| PyTorch NumPy | 43.95 samples/s | 325.59 s | 3084.60 s | 0.01802 |
| rocAL NumPy | 59.14 samples/s | 94.99 s | 2145.52 s | 0.02335 |

In this run, rocAL improved training throughput by 1.35x, validation time by
3.43x, and total epoch time by 1.44x.

These are single-node measurements rather than official benchmark results.
Multi-node correctness and scaling remain to be validated.
