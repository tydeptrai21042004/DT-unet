# Paper-fair baseline correction patch

This patch converts the main `configs/paper_fair/` comparison into a **paper-faithful but still training-fair** protocol.

## Main policy

- Keep the same dataset split, resize, augmentation, optimizer, scheduler, epoch budget, metric logic, and no test-time tricks.
- Keep `backbone_pretrained: false` in the main comparison so pretrained backbones do not get an unfair advantage over the proposal.
- Restore architecture-defining baseline behavior: full non-fast backbone variants, faithful auxiliary outputs, deep supervision, boundary loss, and paper-style attention where needed.

## Changed baseline behavior

| Baseline | Correction |
|---|---|
| Attention U-Net | Replaced borrowed ResUNet++ gate with a local additive Attention U-Net gate. |
| U-Net++ | Enabled deep supervision and faithful dict output. |
| PraNet | Uses full `res2net50_v1b_26w_4s`, faithful side outputs, and aux loss. |
| ACSNet | Uses full `res2net50_v1b_26w_4s`, faithful side outputs, and aux loss. |
| HarDNet-MSEG | Keeps official HarDNet-68 adapter without pretrained checkpoint in main fair run. |
| Polyp-PVT | Uses `pvt_v2_b2`, faithful coarse output, and aux loss. |
| CaraNet | Uses full `res2net50_v1b_26w_4s`, faithful side outputs, and aux loss. |
| CFA-Net | Enables boundary branch supervision and auxiliary outputs. |
| HSNet | Uses full Res2Net + `pvt_v2_b2`, faithful multi-scale outputs, and aux loss. |
| CSCA U-Net | Enables paper attention mode, deep supervision, faithful output, and removes logit clipping. |

## Config folders

- `configs/fair/`: lightweight fast-fair comparison.
- `configs/paper_fair/`: recommended main comparison, now paper-faithful architecture + fair training.
- `configs/paper_fair_faithful/`: copy of the corrected `paper_fair/` configs for explicit naming.
- `configs/official_faithful/`: closest official-style mode, allowed to use pretrained backbones/checkpoints.

## Dataset/threshold consistency

This patch also carries over the dataset-support fix:

- default download URLs are configured for all public dataset keys where the repo can auto-download the common public bundles;
- `eval_threshold_sweep.py` now uses the shared dataset factory, so it supports all registered datasets instead of only `kvasir_seg` and `custom`;
- dataset preparation/loader accepts common image-mask folder names such as `images/masks`, `Original/Ground Truth`, `GT`, `JPEGImages/SegmentationClass`, and `annotation/label`.

## Run

```bash
python scripts/benchmark_paper_fair.py --dataset kvasir_seg
```

or directly:

```bash
python scripts/benchmark_all.py \
  --config-dir configs/paper_fair \
  --models unet,attention_unet,unet_cbam,unetpp,resunetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,cfanet,hsnet,csca_unet,proposal_hf_unet
```

## Notes

`csca_unet` uses a smaller default batch size in paper-fair mode because paper-style CSCA spatial attention is much more memory-heavy than the efficient adapted mode. Override `--batch-size` if your GPU can handle a larger batch.
