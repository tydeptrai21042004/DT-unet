from __future__ import annotations

from typing import Any, ClassVar, Dict

from ..registry import register_model
from .hc_unet import HCUNetNoGateProposal


class _HCUNetAblationVariant(HCUNetNoGateProposal):
    """Base class for one-factor HC-U-Net ablation variants.

    Each subclass fixes only the setting named by the ablation while preserving
    the same no-gate U-Net scaffold and all remaining HC reference settings from
    its YAML configuration.
    """

    ablation_name: ClassVar[str] = "hc_reference"
    forced_overrides: ClassVar[Dict[str, Any]] = {}

    def __init__(self, **kwargs: Any) -> None:
        kwargs.update(self.forced_overrides)
        super().__init__(**kwargs)
        self.hc_ablation_name = self.ablation_name
        self.hc_ablation_settings = dict(self.forced_overrides)


@register_model("hc_reference")
class HCReference(_HCUNetAblationVariant):
    """Complete HC-U-Net no-gate reference configuration."""

    ablation_name = "hc_reference"


@register_model("hc_without_hc_branch")
class HCWithoutHCBranch(_HCUNetAblationVariant):
    """Disable the HC residual contribution by fixing alpha to zero."""

    ablation_name = "hc_without_hc_branch"
    forced_overrides = {
        "hf_alpha": 0.0,
        "hf_alpha_start": 0.0,
        "hf_alpha_warmup_epochs": 0,
    }


@register_model("hc_shared_kernel")
class HCSharedKernel(_HCUNetAblationVariant):
    """Use one height kernel and one width kernel shared by all channels."""

    ablation_name = "hc_shared_kernel"
    forced_overrides = {"hc_shared_kernel": True}


@register_model("hc_learnable_h")
class HCLearnableH(_HCUNetAblationVariant):
    """Learn a positive h value instead of keeping h fixed."""

    ablation_name = "hc_learnable_h"
    forced_overrides = {"hc_learnable_h": True}


@register_model("hc_kernel5")
class HCKernel5(_HCUNetAblationVariant):
    """Use a five-tap HC axial kernel instead of the three-tap reference."""

    ablation_name = "hc_kernel5"
    forced_overrides = {"hc_kernel_size": 5}


@register_model("hc_identity_projection")
class HCIdentityProjection(_HCUNetAblationVariant):
    """Remove learned pre/post projections around the HC branch."""

    ablation_name = "hc_identity_projection"
    forced_overrides = {"hf_projection": "identity"}


@register_model("hc_no_channel_expansion")
class HCNoChannelExpansion(_HCUNetAblationVariant):
    """Use mixer expansion 1.0 instead of the 1.5 reference expansion."""

    ablation_name = "hc_no_channel_expansion"
    forced_overrides = {"hf_expansion": 1.0}


HC_ABLATION_MODEL_NAMES = [
    "hc_reference",
    "hc_without_hc_branch",
    "hc_shared_kernel",
    "hc_learnable_h",
    "hc_kernel5",
    "hc_identity_projection",
    "hc_no_channel_expansion",
]


__all__ = [
    "HCReference",
    "HCWithoutHCBranch",
    "HCSharedKernel",
    "HCLearnableH",
    "HCKernel5",
    "HCIdentityProjection",
    "HCNoChannelExpansion",
    "HC_ABLATION_MODEL_NAMES",
]
