from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..common.decoder import UNetDecoder
from ..common.encoder import PyramidEncoder
from ..common.utils import init_weights
from ..registry import register_model
from .hc_bottleneck import HCBottleneck
from .hf_unet import HFUNet


@register_model("proposal_hf_unet_no_gate")
class HFUNetNoGateProposal(HFUNet):
    """Old HF proposal kept unchanged, but with the residual gate disabled.

    This promotes the best ablation finding (`hf_unet_no_gate`) into an explicit
    proposal model name, without overwriting `proposal_hf_unet`.
    """

    def __init__(self, **kwargs) -> None:
        kwargs["use_gate"] = False
        super().__init__(**kwargs)


@register_model("proposal_hc_unet_no_gate")
class HCUNetNoGateProposal(nn.Module):
    """New no-gate proposal using the weighted h-Hartley-cosine convolution.

    This model is added beside the old HF proposal.  It uses the same U-Net
    encoder/decoder scaffold and inserts a residual HCBottleneck at the deepest
    encoder feature.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        hf_alpha: float = 0.5,
        hf_alpha_start: float = 0.5,
        hf_alpha_warmup_epochs: int = 0,
        hf_expansion: float = 1.5,
        hf_dropout: float = 0.0,
        use_hf_regularizer: bool = False,
        norm: str = "bn",
        act: str = "relu",
        hf_block_norm: str | None = None,
        hf_block_act: str | None = None,
        mixer_act: str = "gelu",
        use_se: bool = False,
        use_gate: bool = False,
        gate_init_bias: float = -2.0,
        decoder_use_cbam: bool = False,
        identity_init: bool = True,
        hf_projection: str = "linear",
        hf_mixer_rank: Optional[int] = None,
        hf_mixer_init_hw: Sequence[int] = (22, 22),
        hc_kernel_size: int = 3,
        hc_h: float = 1.0,
        hc_learnable_h: bool = False,
        hc_shared_kernel: bool = False,
    ) -> None:
        super().__init__()

        # Kept for config compatibility with HFUNet.  The HC proposal does not
        # use the old HF regularizer or mixer-rank parameters.
        _ = use_hf_regularizer, hf_mixer_rank, hf_mixer_init_hw

        self.encoder = PyramidEncoder(in_channels=in_channels, channels=channels, block="double", norm=norm, act=act)
        self.decoder = UNetDecoder(channels=channels, norm=norm, act=act, use_cbam=decoder_use_cbam)
        self.seg_head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

        self.hf_alpha_target = float(hf_alpha)
        self.hf_alpha_start = float(hf_alpha_start)
        self.hf_alpha_warmup_epochs = int(hf_alpha_warmup_epochs)

        block_norm = hf_block_norm or norm
        block_act = hf_block_act or act

        # The new proposal is intentionally no-gate.  The use_gate argument is
        # accepted for config compatibility, but is forced to False so the model
        # name and paper ablation remain unambiguous.
        use_gate = False

        self.hc_bottleneck = HCBottleneck(
            channels=channels[-1],
            expansion=hf_expansion,
            alpha=hf_alpha,
            dropout=hf_dropout,
            use_se=use_se,
            use_gate=use_gate,
            norm=block_norm,
            act=block_act,
            mixer_act=mixer_act,
            gate_init_bias=gate_init_bias,
            identity_init=identity_init,
            projection=hf_projection,
            hc_kernel_size=hc_kernel_size,
            hc_h=hc_h,
            hc_learnable_h=hc_learnable_h,
            hc_shared_kernel=hc_shared_kernel,
        )

        init_weights(self)
        if identity_init:
            self.hc_bottleneck._apply_identity_friendly_init(gate_init_bias=gate_init_bias)

        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        if self.hf_alpha_warmup_epochs <= 0:
            alpha = self.hf_alpha_target
        else:
            progress = min(max(float(epoch), 0.0) / float(self.hf_alpha_warmup_epochs), 1.0)
            alpha = self.hf_alpha_start + (self.hf_alpha_target - self.hf_alpha_start) * progress
        self.hc_bottleneck.set_alpha(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        feats[-1] = self.hc_bottleneck(feats[-1])
        dec = self.decoder(feats)
        return self.seg_head(dec)

    def auxiliary_regularization(self) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.zeros((), device=device)


__all__ = ["HFUNetNoGateProposal", "HCUNetNoGateProposal"]
