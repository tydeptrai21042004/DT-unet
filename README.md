# HF-U-Net benchmark

## Paper-fair training recipe

Use `configs/paper_fair/` for strict paper comparisons. This preset aligns the training recipe across all models:

- same image size: 352
- same batch size: 6
- same augmentation: `strong`
- same optimizer / scheduler: AdamW + cosine
- same learning rate: 3e-4
- same weight decay: 1e-4
- same epochs: 30
- same gradient clipping: 1.0
- same segmentation loss: BCE+Dice
- same threshold: 0.5
- same mixed precision policy: disabled for every model
- no trainer-level auxiliary loss weighting for any model

The proposal model still differs structurally, but proposal-only training-side extras are disabled in `configs/paper_fair/` so the comparison is easier to defend in a paper.

### Run all models with the strict paper setup

```bash
CONFIG_DIR=configs/paper_fair OUTPUT_ROOT=outputs_paper_fair DEVICE=auto \
MODELS="unet,unet_cbam,unetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,proposal_hf_unet" \
bash run.sh benchmark
```

### Kaggle TLS workaround

If auto-download fails because of certificate validation on the dataset host:

```bash
ALLOW_INSECURE_DOWNLOAD=1 CONFIG_DIR=configs/paper_fair OUTPUT_ROOT=outputs_paper_fair DEVICE=auto bash run.sh benchmark
```


Added baseline: HSNet (adapted faithful implementation with CSA, HSC, and MSP modules).


## Official-backbone upgrade

This repo now includes benchmark adapters for public official/public backbone implementations:

- Res2Net for PraNet / ACSNet / CaraNet / HSNet
- PVTv2 for Polyp-PVT / HSNet
- HarDNet-68 for HarDNet-MSEG

### Quick notes

- Default configs are kept runnable on CPU/offline by using the new adapter layer without forcing checkpoint downloads.
- Stronger official-style configs are provided in `configs/official_faithful/`.
- To fetch public pretrained checkpoints, use:

```bash
python scripts/download_official_backbones.py --output-dir weights/official_backbones
```

Then point model checkpoint fields at the downloaded files in your YAML.

## Dataset support added

The generic binary segmentation loader now supports these dataset keys:

| Dataset key | Domain | Pairing support |
|---|---|---|
| `kvasir_seg` | polyp / colonoscopy | exact same image and mask stems |
| `cvc_clinicdb` | polyp / colonoscopy | exact same image and mask stems |
| `cvc_colondb` | polyp / colonoscopy | exact same image and mask stems |
| `etis` | polyp / colonoscopy | exact same image and mask stems |
| `cvc_300` | polyp / colonoscopy | exact same image and mask stems |
| `isic2018` | skin lesion / dermoscopy | `*_segmentation` mask suffix |
| `busi` | breast ultrasound | `*_mask` and `*_mask_1` suffixes, including shared class folders |
| `drive` | retinal vessel / fundus | `*_manual` / `*_manual1` mask suffixes |
| `custom` | any binary segmentation data | common image/mask folder names |

Recommended multi-dataset layout:

```text
data/
  raw/
    Kvasir-SEG/images, Kvasir-SEG/masks
    Dataset_BUSI_with_GT/benign/*.png
    ISIC2018/images, ISIC2018/masks
  processed/
    kvasir_seg/images_352, kvasir_seg/masks_352
    busi/images_352, busi/masks_352
    isic2018/images_352, isic2018/masks_352
  splits/
    kvasir_seg/train.txt, val.txt, test.txt
    busi/train.txt, val.txt, test.txt
```

Prepare and split examples:

```bash
# Public direct-download dataset
DATASET=kvasir_seg bash run.sh prepare
DATASET=kvasir_seg bash run.sh splits

# Automatic official-source download for cross-domain datasets
python scripts/prepare_dataset.py --dataset isic2018 --data-root data --image-size 352
python scripts/make_splits.py --dataset isic2018 --data-root data --image-size 352

python scripts/prepare_dataset.py --dataset busi --data-root data --image-size 352
python scripts/make_splits.py --dataset busi --data-root data --image-size 352

# Local source directories remain supported and override automatic official-source download.
python scripts/prepare_dataset.py --dataset isic2018 --source-dir /path/to/ISIC2018 --data-root data --image-size 352
python scripts/prepare_dataset.py --dataset busi --source-dir /path/to/Dataset_BUSI_with_GT --data-root data --image-size 352

python scripts/prepare_dataset.py --dataset drive --source-dir /path/to/DRIVE --data-root data --image-size 352
python scripts/make_splits.py --dataset drive --data-root data --image-size 352
```

Legacy flat folders such as `data/processed/images_352` and `data/splits/train.txt` are still supported for backward compatibility.

## Fairness controls added

### CSCA-U-Net effective batch-size fix

`configs/paper_fair/csca_unet.yaml` keeps physical `batch_size: 2` for memory safety but now uses:

```yaml
train:
  gradient_accumulation_steps: 3
```

So the effective batch size is `2 × 3 = 6`, matching the other paper-fair configs.

### Strict no-auxiliary-loss benchmark

Use this for an architecture-only table where models are compared with the same main-output BCE+Dice loss and no side-output/boundary loss advantage:

```bash
bash run.sh benchmark-strict-no-aux
# or explicitly:
CONFIG_DIR=configs/strict_no_aux OUTPUT_ROOT=outputs_strict_no_aux bash run.sh benchmark
```

The new training flags are:

```yaml
train:
  use_aux_outputs_loss: false
  use_boundary_loss: false
```

### Paper-fair pretrained-backbone benchmark

The official-style backbone adapters can now auto-download public ImageNet checkpoints when `backbone_pretrained: true` is set and no local checkpoint is provided.

```bash
bash run.sh benchmark-pretrained
# optional manual download cache
bash run.sh download-backbones --output-dir weights/official_backbones
```

Available default auto-download keys are:

```text
res2net50_v1b_26w_4s
pvt_v2_b2
hardnet68
```

Use `configs/paper_fair/` for from-scratch fair comparison, `configs/strict_no_aux/` for architecture-only fair comparison, and `configs/paper_fair_pretrained/` for paper-style pretrained-backbone comparison.

## Dedicated HC-U-Net ablation suite

The repository includes a separate HC-only ablation suite. It does not run
HF-U-Net variants or unrelated segmentation baselines.

| Variant | Purpose |
|---|---|
| `hc_reference` | Complete HC-U-Net no-gate reference configuration |
| `hc_without_hc_branch` | Sets `alpha=0`, disabling the HC residual contribution |
| `hc_shared_kernel` | Uses one shared height kernel and one shared width kernel across channels |
| `hc_learnable_h` | Learns a positive `h` instead of fixing `h=1` |
| `hc_kernel5` | Changes the HC axial kernel size from 3 to 5 |
| `hc_identity_projection` | Removes learned pre/post projections |
| `hc_no_channel_expansion` | Changes mixer expansion from 1.5 to 1.0 |

All seven variants use:

- the same U-Net encoder and decoder;
- the no-gate HC bottleneck;
- the same BCE-plus-Dice main loss;
- no auxiliary-output loss;
- no boundary loss;
- no proposal-only regularizer;
- no alpha warm-up.

### Run on Kvasir-SEG

```bash
python -m pytest -q \
  tests/test_hc_ablation_variants.py \
  tests/test_hc_operator_contracts.py

DATASET=kvasir_seg \
DEVICE=cuda \
SEEDS=42,1,2 \
OUTPUT_ROOT=outputs_hc_ablation_kvasir \
  bash run_hc_ablation.sh
```

Equivalent command through the main runner:

```bash
DATASET=kvasir_seg \
DEVICE=cuda \
SEEDS=42,1,2 \
OUTPUT_ROOT=outputs_hc_ablation_kvasir \
  bash run.sh hc-ablation --batch-size 6 --epochs 30 --num-workers 2
```

### Run on ISIC 2018 with automatic official-source download

```bash
DATASET=isic2018 \
DEVICE=cuda \
SEEDS=42,1,2 \
OUTPUT_ROOT=outputs_hc_ablation_isic2018 \
  bash run_hc_ablation.sh
```

### Run on BUSI with automatic official-source download

```bash
DATASET=busi \
DEVICE=cuda \
SEEDS=42,1,2 \
OUTPUT_ROOT=outputs_hc_ablation_busi \
  bash run_hc_ablation.sh
```

The aggregated table is saved at:

```text
<OUTPUT_ROOT>/results/tables/multi_seed_summary.csv
```

The complete configurations are stored in `configs/hc_ablation/` and can also
be trained individually with `scripts/train_one.py`.

## Four balanced independent Kaggle sessions

The repository includes four permanent session runners. Each session performs
24 model-seed runs and automatically downloads only the required cross-domain dataset from its official source:

| Session | Automatic dataset | Work allocation |
|---|---|---|
| 1 | ISIC 2018 | six comparison models, Kvasir HC proposal, HC reference |
| 2 | ISIC 2018 | five comparison models, ClinicDB HC proposal, two HC ablations |
| 3 | BUSI | six comparison models, ColonDB HC proposal, learnable-h ablation |
| 4 | BUSI | five comparison models, three remaining HC ablations |

Run one session per independent Kaggle notebook/runtime:

```bash
bash run_hc_session_1.sh
bash run_hc_session_2.sh
bash run_hc_session_3.sh
bash run_hc_session_4.sh
```

Useful overrides:

```bash
INSTALL_DEPS=0 RUN_TESTS=0 EPOCHS=30 BATCH_SIZE=6 DEVICE=cuda \
  bash run_hc_session_1.sh
```

ISIC 2018 is downloaded from the official ISIC Challenge archives:
`ISIC2018_Task1-2_Training_Input.zip` and
`ISIC2018_Task1_Training_GroundTruth.zip`. BUSI is downloaded from the
official Cairo University archive `https://scholar.cu.edu.eg/Dataset_BUSI.zip`.
A local `--source-dir`, zip, or explicit direct URL override remains available.

