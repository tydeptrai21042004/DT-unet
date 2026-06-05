# Patch notes: datasets, fairness controls, and auto pretrained backbones

## Added dataset support

Added registry/loader support for:

- `isic2018`
- `busi`
- `drive`

The existing polyp dataset keys remain supported:

- `kvasir_seg`
- `cvc_clinicdb`
- `cvc_colondb`
- `etis`
- `cvc_300`
- `custom`

The generic loader now handles common mask suffixes:

- ISIC: `*_segmentation`
- BUSI: `*_mask`, `*_mask_1`
- DRIVE: `*_manual`, `*_manual1`

It also supports dataset-specific processed folders and split folders:

```text
data/processed/<dataset>/images_<size>
data/processed/<dataset>/masks_<size>
data/splits/<dataset>/train.txt
```

Legacy flat processed/split folders are still supported.

## Fixed paper-fair CSCA batch issue

`configs/paper_fair/csca_unet.yaml` now uses:

```yaml
data:
  batch_size: 2
train:
  gradient_accumulation_steps: 3
```

This keeps GPU memory safer while matching the effective batch size of 6 used by the other paper-fair configs.

## Added strict no-aux benchmark

Added:

```text
configs/strict_no_aux/
scripts/benchmark_strict_no_aux.py
```

These configs disable auxiliary side-output and boundary losses through:

```yaml
train:
  use_aux_outputs_loss: false
  use_boundary_loss: false
```

This gives a cleaner architecture-only comparison table.

## Added auto official checkpoint loading

`src/models/common/official_backbones.py` now includes default public checkpoint URLs for:

- Res2Net-50 v1b 26w 4s
- PVTv2-B2
- HarDNet-68

When an official adapter is created with `pretrained=True` and no explicit checkpoint path/URL, it automatically uses the default URL.

Added:

```text
configs/paper_fair_pretrained/
scripts/benchmark_paper_fair_pretrained.py
```

## Smoke verification performed

- Selected dataset/fairness/backbone tests passed.
- Tiny custom dataset train/eval smoke passed with CPU and `OMP_NUM_THREADS=1`.
- Full test suite is very slow in this container and timed out before completion; selected changed-area tests passed.
