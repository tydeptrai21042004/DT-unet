# Patch notes: architecture-only HF-U-Net ablation

This patch turns `configs/ablation/` into a strict architecture-only ablation suite and adds more HF controls.

## Fixed fairness issue

- `configs/ablation/unet.yaml` now uses `batch_size: 6`, matching all other ablation variants.
- All ablation configs now use the same core training recipe.
- Proposal-specific training helpers are disabled in this folder:
  - `use_hf_regularizer: false`
  - `hf_alpha_start: hf_alpha`
  - `hf_alpha_warmup_epochs: 0`
  - `aux_loss_weight: 0.0`
  - `aux_warmup_epochs: 0`
  - `aux_ramp_epochs: 0`

## Added architecture-only variants

The ablation suite now includes 15 variants:

1. `unet`
2. `unet_conv_bottleneck`
3. `unet_fft_bottleneck`
4. `proposal_hf_unet`
5. `hf_unet_wo_hartley`
6. `hf_unet_wo_fourier_kernel`
7. `hf_unet_wo_residual`
8. `hf_unet_encoder_stage4`
9. `hf_unet_decoder_stage`
10. `hf_unet_no_gate`
11. `hf_unet_with_se`
12. `hf_unet_identity_projection`
13. `hf_unet_conv_projection`
14. `hf_unet_low_rank_mixer`
15. `hf_unet_high_rank_mixer`

## Code changes

- `src/models/proposal/hf_ablation.py`
  - Added `hf_projection`, `hf_mixer_rank`, and `hf_mixer_init_hw` passthrough support.
  - Added registered wrappers for the new ablation variants.
- `scripts/run_compact_hf_ablation.py`
  - Updated to run all 15 architecture-only variants.
- `run.sh`
  - Added `bash run.sh ablation` shortcut.
- `tests/test_hf_ablation_variants.py`
  - Added registration, forward, backward, config-fairness, and script/config consistency checks.

## Recommended commands

Full architecture-only ablation:

```bash
python scripts/run_compact_hf_ablation.py \
  --dataset cvc_clinicdb \
  --data-root data \
  --image-size 352 \
  --batch-size 6 \
  --seed 42 \
  --device cuda \
  --output-root outputs_arch_only_ablation_cvc_clinicdb
```

Using `run.sh`:

```bash
CONFIG_DIR=configs/ablation \
DATASET=cvc_clinicdb \
DATA_ROOT=data \
OUTPUT_ROOT=outputs_arch_only_ablation_cvc_clinicdb \
DEVICE=cuda \
bash run.sh ablation --batch-size 6
```

Quick CPU smoke test:

```bash
python scripts/smoke_all_models.py \
  --models "unet,unet_conv_bottleneck,unet_fft_bottleneck,proposal_hf_unet,hf_unet_wo_hartley,hf_unet_wo_fourier_kernel,hf_unet_wo_residual,hf_unet_encoder_stage4,hf_unet_decoder_stage,hf_unet_no_gate,hf_unet_with_se,hf_unet_identity_projection,hf_unet_conv_projection,hf_unet_low_rank_mixer,hf_unet_high_rank_mixer" \
  --config-dir configs/ablation \
  --image-size 32 \
  --batch-size 1 \
  --device cpu
```
