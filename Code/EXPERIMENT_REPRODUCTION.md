# Experiment reproduction notes

This note documents the local compatibility work and the model files retained for the event-camera experiments. RAW recordings, annotations, manifests, cached images, and checkpoint binaries remain external inputs and are not committed to this repository.

## Runtime adaptation

The released implementation uses TensorFlow 1.x graph APIs. The experiment code uses `tf.compat.v1` for graph-mode variables and layers, and `tf_slim` for the former `tensorflow.contrib.framework` argument-scope helpers. The RAW inference entry point disables eager execution. The repository does not pin a complete TensorFlow, OpenCV, CUDA, or Metavision SDK environment, so record the environment used for any new result separately.

`ImageUtils.py` includes polarity visualization helpers used for RAW previews. The RAW inference and post-processing scripts expose sensor geometry, accumulation-window timing, response activation, thresholding, component filtering, and timestamp-anchor settings as command-line options.

## RAW inference

Install TensorFlow, NumPy, OpenCV, and the Prophesee Metavision Python bindings (`metavision_core`) in the same environment. From the `Code` directory, run the inference entry point with explicit local paths:

```powershell
python .\test_raw_ckpt.py `
  --raw_path "<path-to-recording.raw>" `
  --ckpt_path "..\DVSProp_single\CheckPoints\79model.ckpt" `
  --output_dir "<output-directory>" `
  --delta_t_us 50000 --width 1280 --height 720 `
  --vis_style prophesee --save_video 1 --fps 20
```

The command writes box predictions and, when enabled, a preview video under the selected output directory. `DVSProp/CheckPoints/49model.ckpt` is the alternate retained checkpoint prefix. The `run_raw_*.sh` files from the handoff snapshot contain machine-specific paths and are not part of this portable procedure.

## M30T bbox adaptation

`TrainM30TBBox.py` adapts the propeller-response model to full-UAV bbox occupancy targets; it is a task adaptation, not the original EVPropNet propeller-center objective. `EvaluateM30TBBox.py` evaluates the resulting checkpoint prefixes on a selected manifest partition. Both scripts require an external JSONL dataset manifest and image caches whose paths are supplied by the experiment owner. The train, validation, and held-out test partitions must remain recording-disjoint as defined by that manifest.

## Retained baseline checkpoints

Checkpoint binaries are ignored by Git. The following prefixes are the retained default models identified by the checkpoint state files in the handoff snapshot; this records selection metadata, not a claim that these are the best-performing models on every dataset.

| Checkpoint prefix | Component | Bytes | SHA-256 |
|---|---|---:|---|
| `DVSProp/CheckPoints/49model.ckpt` | `.data-00000-of-00001` | 32,587,796 | `bb96f713cc7c2abb83d2963f3bbef27beeed4f1cfc08f4e6f9ac2a984d0c6e7d` |
| `DVSProp/CheckPoints/49model.ckpt` | `.index` | 9,721 | `3b423ba02d18a962de2ed0cca1e109ce80e3ee79931933bc774be8078ba4df31` |
| `DVSProp/CheckPoints/49model.ckpt` | `.meta` | 936,637 | `6cc20f269c74d2c869d2a33de373edcea9bc1f355b190f00d183e643336d341f` |
| `DVSProp_single/CheckPoints/79model.ckpt` | `.data-00000-of-00001` | 32,587,796 | `216633d9e2968edb12d77b876878d86978bf20c5ce50a53f339813b27ec8612a` |
| `DVSProp_single/CheckPoints/79model.ckpt` | `.index` | 9,721 | `969e520e59bf06945d8251d0e4e6f2bd6f01533e329b90bc6b769a8f15933f68` |
| `DVSProp_single/CheckPoints/79model.ckpt` | `.meta` | 957,475 | `fe87fa336d28ccdb4aef8fe8b740b862381a0a2394dcf9f6f73363a427b3e3be` |

The C09 NJ-source P50 transfer experiment retained `evpropnet_nj_source_p50_epoch_3.ckpt`, selected on source-validation blocks at the frozen response threshold 0.75 (IoU@0.3 recall 0.74648). Its TensorFlow component hashes are:

| Component | Bytes | SHA-256 |
|---|---:|---|
| `.data-00000-of-00001` | 10,869,508 | `4e85ee6a1bf2d89c91f24ae04a16ff70550e57ed18e5dbfa056dd9e8ab17c2ed` |
| `.index` | 4,086 | `b0e2942898dae876f85d34fb65dabbbdb6123eeab896ca735c4d9ee4997b3690` |
| `.meta` | 995,413 | `9d7a91be1fbf89bec8cd59693c623b566772001681e4124abb4f556168f4e80b` |

This P50 transfer checkpoint is a separate protocol from EV-SpSegNet's P100 NJ-to-LX checkpoint. Target LX data were not used to select the EVPropNet epoch. Neither checkpoint binary nor its training data is included in Git.
