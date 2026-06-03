# Architecture-only HF-U-Net ablation configs

This folder contains the full architecture-only ablation suite. It disables training-only helpers so that the table isolates architectural effects.

For the specific question **"where is the best place to insert the HF block?"**, use the smaller focused folder:

```text
configs/placement_ablation/
```

and run:

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

## Fairness rules used by every config here

- same image size, batch size, optimizer, scheduler, learning rate, and training budget
- `batch_size: 6` for all variants, including the plain U-Net control
- `aux_loss_weight: 0.0`
- `use_hf_regularizer: false` for every HF variant
- `hf_alpha_start: hf_alpha` and `hf_alpha_warmup_epochs: 0`
- no proposal-specific warmup or auxiliary regularization

## Placement-focused variants

These are the most important variants for finding the best HF insertion point:

1. `unet` — no HF block baseline
2. `hf_unet_hf_at_encoder0` — HF after encoder stem / highest-resolution skip
3. `hf_unet_hf_at_encoder1` — HF after encoder stage 1
4. `hf_unet_hf_at_encoder2` — HF after encoder stage 2
5. `hf_unet_hf_at_encoder3` — HF after encoder stage 3 / pre-bottleneck
6. `hf_unet_hf_at_bottleneck` — HF at deepest encoder bottleneck
7. `hf_unet_hf_at_decoder3` — HF after first decoder block
8. `hf_unet_hf_at_decoder2` — HF after second decoder block
9. `hf_unet_hf_at_decoder1` — HF after third decoder block
10. `hf_unet_hf_at_decoder0` — HF after final decoder block / before segmentation head

## Other architecture controls

These remain available for secondary ablation after the best placement is chosen:

- `unet_conv_bottleneck` — local convolution bottleneck control
- `unet_fft_bottleneck` — generic FFT/GFNet-like spectral bottleneck
- `proposal_hf_unet` — original proposal class, HF at deepest encoder feature
- `hf_unet_wo_hartley` — remove Hartley signal transform
- `hf_unet_wo_fourier_kernel` — remove learnable Fourier-kernel/frequency mixer
- `hf_unet_wo_residual` — remove residual identity path in the HF block
- `hf_unet_encoder_stage4` — legacy pre-bottleneck placement control
- `hf_unet_decoder_stage` — legacy post-decoder placement control
- `hf_unet_no_gate` — remove adaptive residual gate
- `hf_unet_with_se` — add squeeze-and-excitation inside the HF block
- `hf_unet_identity_projection` — use identity pre/post projection
- `hf_unet_conv_projection` — use legacy 3x3 conv pre/post projection
- `hf_unet_low_rank_mixer` — use lower-rank mixer capacity (`hf_expansion=1.0`)
- `hf_unet_high_rank_mixer` — use higher-rank mixer capacity (`hf_expansion=2.0`)

## Full architecture-only run

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
