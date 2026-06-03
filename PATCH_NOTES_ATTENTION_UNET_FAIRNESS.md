# Patch notes: Attention U-Net baseline and fairness/test corrections

This patch adds `attention_unet` as an additional architecture-only baseline. It reuses the same channel budget, data recipe, optimizer, scheduler, loss, image size, and epoch budget as the plain U-Net in the default, fair, paper-fair, and faithful config groups.

Main changes:

- Added `src/models/baselines/attention_unet.py`.
- Registered `attention_unet` in model imports and default benchmark scripts.
- Added `configs/*/attention_unet.yaml`.
- Fixed the ResUNet++ contract names by exposing `aspp_bridge` and `dec4` while keeping backward-compatible aliases.
- Exposed `ConvNormAct.conv` for architecture-contract checks.
- Made `configs/paper_fair/proposal_hf_unet.yaml` strictly architecture-fair by disabling HF warm-up and the proposal-only regularizer.
- Aligned `configs/faithful/csca_unet.yaml` batch size with the shared faithful recipe.
- Added tests for Attention U-Net registration/configuration, forward/backward behavior, attention-gated skip connections, gate conditioning, and ResUNet++ contract aliases.
