# HF placement ablation configs

This folder focuses only on the question: **where should the HF block be inserted in U-Net?**

All configs keep the same architecture-only training recipe:

- same image size, batch size, optimizer, scheduler, learning rate, and training budget
- `batch_size: 6`
- `aux_loss_weight: 0.0`
- `use_hf_regularizer: false`
- `hf_alpha_start: hf_alpha`
- `hf_alpha_warmup_epochs: 0`

Placement variants:

1. `unet` — no HF block baseline
2. `hf_unet_hf_at_encoder0` — HF after encoder stem / highest-resolution skip
3. `hf_unet_hf_at_encoder1` — HF after encoder stage 1
4. `hf_unet_hf_at_encoder2` — HF after encoder stage 2
5. `hf_unet_hf_at_encoder3` — HF after encoder stage 3 / pre-bottleneck
6. `hf_unet_hf_at_bottleneck` — HF at deepest bottleneck
7. `hf_unet_hf_at_decoder3` — HF after first decoder block
8. `hf_unet_hf_at_decoder2` — HF after second decoder block
9. `hf_unet_hf_at_decoder1` — HF after third decoder block
10. `hf_unet_hf_at_decoder0` — HF after final decoder block / before segmentation head

Recommended run:

```bash
python scripts/run_hf_placement_ablation.py   --dataset cvc_clinicdb   --data-root data   --image-size 352   --batch-size 6   --seed 42   --device cuda   --output-root outputs_hf_placement_ablation_cvc_clinicdb
```

Quick CPU smoke run:

```bash
python scripts/run_hf_placement_ablation.py   --dataset custom   --data-root data_tiny   --image-size 64   --batch-size 2   --epochs 1   --num-workers 0   --device cpu   --output-root outputs_smoke_hf_placement_ablation
```
