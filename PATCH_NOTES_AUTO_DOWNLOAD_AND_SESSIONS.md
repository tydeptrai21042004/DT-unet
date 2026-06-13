# Automatic cross-domain downloads and balanced sessions

## Added automatic dataset sources

- `isic2018` downloads through KaggleHub handle
  `tschandl/isic2018-challenge-task1-data-segmentation`.
- `busi` downloads through KaggleHub handle
  `sabahesaraki/breast-ultrasound-images-dataset`.
- Local folders, zip archives, direct URLs, and explicit Kaggle handles remain
  supported and take precedence over registry defaults.

## CLI propagation

`--kaggle-handle` is supported by:

- `scripts/prepare_dataset.py`
- `scripts/benchmark_all.py`
- `scripts/benchmark_multi_seed.py`
- `scripts/run_hc_ablation.py`
- `run_hc_ablation.sh` through `KAGGLE_HANDLE`
- `run.sh` through `KAGGLE_HANDLE`

## Four independent runners

- `run_hc_session_1.sh`
- `run_hc_session_2.sh`
- `run_hc_session_3.sh`
- `run_hc_session_4.sh`
- shared implementation: `run_hc_balanced_session.sh`

Each session has 24 model-seed runs and downloads only ISIC 2018 or BUSI as
required. The sessions do not execute the HF proposal.
