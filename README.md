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

# Local / manually downloaded cross-domain datasets
python scripts/prepare_dataset.py --dataset isic2018 --source-dir /path/to/ISIC2018 --data-root data --image-size 352
python scripts/make_splits.py --dataset isic2018 --data-root data --image-size 352

python scripts/prepare_dataset.py --dataset busi --source-dir /path/to/Dataset_BUSI_with_GT --data-root data --image-size 352
python scripts/make_splits.py --dataset busi --data-root data --image-size 352

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
