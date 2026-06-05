from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import ConvNormAct, SqueezeExcitation


@dataclass
class HCRegularizationTerms:
    """Lightweight regularization terms for the h-Hartley-cosine block.

    The existing HF regularizer is tied to HFBottleneck internals.  This class is
    intentionally local to the new proposal so the old proposal remains
    unchanged and the new model can expose diagnostics later if needed.
    """

    response_smoothness: torch.Tensor
    response_magnitude: torch.Tensor
    energy_penalty: torch.Tensor
    stability_penalty: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.response_smoothness + self.response_magnitude + self.energy_penalty + self.stability_penalty


class WeightedHHartleyCosineAxialConv(nn.Module):
    r"""Finite 2D axial implementation of the weighted h-Hartley-cosine convolution.

    The paper operator is 1D on :math:`\mathbb{T}_h`:

    .. math::

        (f *_\gamma g)(nh) = \frac{h}{2}\sum_m f(mh)
        [g(nh-mh-h)+g(nh-mh+h)+g(nh+mh+h)+g(nh+mh-h)].

    For feature maps, we use a finite, zero-boundary, depthwise axial extension.
    For each channel, a learnable finite filter f is applied along height and
    width.  The terms with ``n-m`` are standard convolution terms, while the
    terms with ``n+m`` are implemented by the flipped filter ``f(-m)``.  The
    ``±h`` shifts become one-pixel neighbor shifts on the chosen axis.

    This is deliberately not a drop-in Hartley transform.  It is a new spatial
    convolutional proposal block derived from the weighted h-Hartley-cosine
    convolution formula, added beside the old HF proposal.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        h: float = 1.0,
        learnable_h: bool = False,
        shared_kernel: bool = False,
        init_scale: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError("kernel_size must be an odd integer >= 3.")

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.shared_kernel = bool(shared_kernel)
        kernel_channels = 1 if self.shared_kernel else self.channels

        self.kernel_h = nn.Parameter(torch.empty(kernel_channels, 1, self.kernel_size, 1))
        self.kernel_w = nn.Parameter(torch.empty(kernel_channels, 1, 1, self.kernel_size))

        if learnable_h:
            # Softplus keeps h positive while allowing stable optimization.
            h_tensor = torch.tensor(float(h)).clamp_min(1.0e-6)
            self.raw_h = nn.Parameter(torch.log(torch.expm1(h_tensor)))
            self.register_buffer("fixed_h", torch.empty(0), persistent=False)
        else:
            self.raw_h = None
            self.register_buffer("fixed_h", torch.tensor(float(h)))

        self.reset_parameters(init_scale=init_scale)

    def reset_parameters(self, init_scale: float = 1.0e-3) -> None:
        nn.init.normal_(self.kernel_h, mean=0.0, std=init_scale)
        nn.init.normal_(self.kernel_w, mean=0.0, std=init_scale)

    @property
    def h_value(self) -> torch.Tensor:
        if self.raw_h is not None:
            return F.softplus(self.raw_h).clamp_min(1.0e-6)
        return self.fixed_h.clamp_min(1.0e-6)

    def _expanded_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        if self.shared_kernel:
            return kernel.expand(self.channels, -1, -1, -1)
        return kernel

    @staticmethod
    def _neighbor_sum(y: torch.Tensor, *, dim: int) -> torch.Tensor:
        """Return y[n-1] + y[n+1] with zero boundary padding."""
        out = torch.zeros_like(y)
        if dim == -2:
            out[..., 1:, :] = out[..., 1:, :] + y[..., :-1, :]
            out[..., :-1, :] = out[..., :-1, :] + y[..., 1:, :]
        elif dim == -1:
            out[..., :, 1:] = out[..., :, 1:] + y[..., :, :-1]
            out[..., :, :-1] = out[..., :, :-1] + y[..., :, 1:]
        else:
            raise ValueError("dim must be -2 or -1.")
        return out

    def _axis_operator(self, x: torch.Tensor, *, axis: str) -> torch.Tensor:
        if axis == "h":
            kernel = self._expanded_kernel(self.kernel_h).to(device=x.device, dtype=x.dtype)
            padding = (self.kernel_size // 2, 0)
            shift_dim = -2
            flip_dims = (-2,)
        elif axis == "w":
            kernel = self._expanded_kernel(self.kernel_w).to(device=x.device, dtype=x.dtype)
            padding = (0, self.kernel_size // 2)
            shift_dim = -1
            flip_dims = (-1,)
        else:
            raise ValueError("axis must be 'h' or 'w'.")

        # Standard n-m terms and reflected n+m terms.
        standard = F.conv2d(x, kernel, bias=None, stride=1, padding=padding, groups=self.channels)
        reflected = F.conv2d(x, torch.flip(kernel, dims=flip_dims), bias=None, stride=1, padding=padding, groups=self.channels)

        # Apply the ±h shifts from the definition as one-neighbor shifts.
        return 0.5 * self.h_value.to(device=x.device, dtype=x.dtype) * (
            self._neighbor_sum(standard, dim=shift_dim) + self._neighbor_sum(reflected, dim=shift_dim)
        )

    def regularization_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        kernels = [self.kernel_h, self.kernel_w]
        smooth_terms = []
        mag_terms = []
        for k in kernels:
            mag_terms.append(k.pow(2).mean())
            if k.shape[-2] > 1:
                smooth_terms.append((k[:, :, 1:, :] - k[:, :, :-1, :]).abs().mean())
            if k.shape[-1] > 1:
                smooth_terms.append((k[:, :, :, 1:] - k[:, :, :, :-1]).abs().mean())
        smooth = sum(smooth_terms) if smooth_terms else self.kernel_h.new_zeros(())
        magnitude = sum(mag_terms) / len(mag_terms)
        return smooth, magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average vertical and horizontal finite h-Hartley-cosine operators.
        return 0.5 * (self._axis_operator(x, axis="h") + self._axis_operator(x, axis="w"))


class HCBottleneck(nn.Module):
    """No-gate h-Hartley-cosine convolution bottleneck for U-Net features.

    This is the new proposal block requested by the user.  It keeps the residual
    U-Net bottleneck interface of HFBottleneck but replaces the Hartley/Fourier
    transform-mixer path with a finite weighted h-Hartley-cosine convolution.
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 1.5,
        alpha: float = 0.5,
        dropout: float = 0.0,
        use_se: bool = False,
        use_gate: bool = False,
        norm: str = "bn",
        act: str = "relu",
        mixer_act: str = "gelu",
        gate_init_bias: float = -2.0,
        identity_init: bool = True,
        projection: str = "linear",
        hc_kernel_size: int = 3,
        hc_h: float = 1.0,
        hc_learnable_h: bool = False,
        hc_shared_kernel: bool = False,
        hc_init_scale: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.base_alpha = float(alpha)
        self.alpha = float(alpha)
        self.use_gate = bool(use_gate)
        self.projection = projection.lower()

        self.pre = self._make_projection(self.channels, self.projection, norm=norm, act=act)
        self.hc_conv = WeightedHHartleyCosineAxialConv(
            channels=self.channels,
            kernel_size=hc_kernel_size,
            h=hc_h,
            learnable_h=hc_learnable_h,
            shared_kernel=hc_shared_kernel,
            init_scale=hc_init_scale,
        )

        hidden = max(int(self.channels * expansion), self.channels)
        self.mixer = nn.Sequential(
            nn.Conv2d(self.channels, hidden, kernel_size=1, bias=False),
            self._make_activation(mixer_act),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, self.channels, kernel_size=1, bias=False),
        )

        self.post = self._make_projection(self.channels, self.projection, norm=norm, act=act)
        self.se = SqueezeExcitation(self.channels) if use_se else nn.Identity()

        if self.use_gate:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.channels, self.channels, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.gate = nn.Identity()

        self._last_terms: Optional[HCRegularizationTerms] = None

        if identity_init:
            self._apply_identity_friendly_init(gate_init_bias=gate_init_bias)

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "identity":
            return nn.Identity()
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "gelu":
            return nn.GELU()
        if name == "silu":
            return nn.SiLU(inplace=True)
        raise ValueError(f"Unsupported mixer activation: {name}")

    @staticmethod
    def _make_projection(channels: int, mode: str, norm: str, act: str) -> nn.Module:
        mode = mode.lower()
        if mode == "identity":
            return nn.Identity()
        if mode == "linear":
            return nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        if mode == "conv":
            return ConvNormAct(channels, channels, kernel_size=3, norm=norm, act=act)
        raise ValueError("Unsupported HC projection mode. Use 'identity', 'linear', or 'conv'.")

    @staticmethod
    def _init_projection_as_identity(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            if module.kernel_size == (1, 1) and module.in_channels == module.out_channels:
                nn.init.dirac_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _apply_identity_friendly_init(self, gate_init_bias: float = -2.0) -> None:
        self._init_projection_as_identity(self.pre)
        self._init_projection_as_identity(self.post)

        # Keep the new convolutional residual initially small.
        self.hc_conv.reset_parameters(init_scale=1.0e-3)
        last = self.mixer[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.normal_(last.weight, mean=0.0, std=1.0e-3)

        if self.use_gate:
            gate_conv = self.gate[1]
            nn.init.zeros_(gate_conv.weight)
            if gate_conv.bias is not None:
                nn.init.constant_(gate_conv.bias, gate_init_bias)

    def regularization_terms(self) -> Optional[HCRegularizationTerms]:
        return self._last_terms

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        projected = self.pre(x)
        conv_response = self.hc_conv(projected)
        restored = self.mixer(conv_response)
        restored = self.post(restored)
        restored = self.se(restored)

        if self.use_gate:
            gate = self.gate(restored)
        else:
            gate = torch.ones_like(restored)

        residual = gate * restored
        out = identity + self.alpha * residual

        eps = 1.0e-6
        identity_energy = identity.pow(2).mean().detach() + eps
        restored_energy = restored.pow(2).mean()
        energy_penalty = (restored_energy / identity_energy - 1.0).abs()
        stability_penalty = torch.relu((self.alpha * residual).pow(2).mean() / identity_energy - 1.0)
        response_smoothness, response_magnitude = self.hc_conv.regularization_terms()

        self._last_terms = HCRegularizationTerms(
            response_smoothness=response_smoothness,
            response_magnitude=response_magnitude,
            energy_penalty=energy_penalty,
            stability_penalty=stability_penalty,
        )
        return out


__all__ = ["HCRegularizationTerms", "WeightedHHartleyCosineAxialConv", "HCBottleneck"]
