# Patch notes: HF placement ablation

This patch refocuses the architecture-only ablation suite on the question:

> Where is the best place to insert the Hartley-Fourier (HF) block in U-Net?

## Added placement-only HF variants

The new variants keep the same HF block hyperparameters and training recipe, but move the HF block to different U-Net stages:

| Model name | HF insertion location | Feature scale for 352x352 input |
|---|---|---|
| `hf_unet_hf_at_encoder0` | after encoder stem / highest-resolution skip | 352x352 |
| `hf_unet_hf_at_encoder1` | after encoder stage 1 | 176x176 |
| `hf_unet_hf_at_encoder2` | after encoder stage 2 | 88x88 |
| `hf_unet_hf_at_encoder3` | after encoder stage 3 / pre-bottleneck | 44x44 |
| `hf_unet_hf_at_bottleneck` | deepest encoder bottleneck | 22x22 |
| `hf_unet_hf_at_decoder3` | after first decoder block | 44x44 |
| `hf_unet_hf_at_decoder2` | after second decoder block | 88x88 |
| `hf_unet_hf_at_decoder1` | after third decoder block | 176x176 |
| `hf_unet_hf_at_decoder0` | after final decoder block / before segmentation head | 352x352 |

## New focused config folder

A new folder was added:

```text
configs/placement_ablation/
```

This folder contains only:

- `unet`
- the nine placement variants above

This avoids mixing placement analysis with no-gate, SE, projection, or mixer-capacity ablations.

## New runner

```bash
python scripts/run_hf_placement_ablation.py \
  --dataset cvc_clinicdb \
  --data-root data \
  --image-size 352 \
  --batch-size 6 \
  --seed 42 \
  --device cuda \
  --output-root outputs_hf_placement_ablation_cvc_clinicdb
```

Equivalent `run.sh` command:

```bash
DATASET=cvc_clinicdb \
DATA_ROOT=data \
OUTPUT_ROOT=outputs_hf_placement_ablation_cvc_clinicdb \
DEVICE=cuda \
bash run.sh placement-ablation --batch-size 6
```

## Fairness settings

All placement configs are architecture-only:

- `batch_size: 6`
- `aux_loss_weight: 0.0`
- `use_hf_regularizer: false`
- `hf_alpha_start: hf_alpha`
- `hf_alpha_warmup_epochs: 0`
- same optimizer, scheduler, loss, seed, and training budget

## Tests added/updated

`tests/test_hf_ablation_variants.py` now verifies:

- all placement variants are registered
- all configs are buildable
- the placement-only runner matches `configs/placement_ablation/`
- every HF placement variant receives the expected feature shape using a forward hook
- every variant still performs a valid forward/backward pass
