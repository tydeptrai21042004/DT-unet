from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.proposal.hc_bottleneck import WeightedHHartleyCosineAxialConv


torch.set_num_threads(1)


def test_hc_operator_rejects_invalid_kernel_sizes() -> None:
    with pytest.raises(ValueError):
        WeightedHHartleyCosineAxialConv(channels=2, kernel_size=2)
    with pytest.raises(ValueError):
        WeightedHHartleyCosineAxialConv(channels=2, kernel_size=1)


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_hc_operator_preserves_shape_and_finite_values(kernel_size: int) -> None:
    operator = WeightedHHartleyCosineAxialConv(channels=3, kernel_size=kernel_size)
    x = torch.randn(2, 3, 11, 13)
    y = operator(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_hc_operator_maps_zero_to_zero() -> None:
    operator = WeightedHHartleyCosineAxialConv(channels=4, kernel_size=3)
    x = torch.zeros(2, 4, 9, 7)
    y = operator(x)
    assert torch.equal(y, torch.zeros_like(y))


def test_fixed_h_scales_the_operator_linearly() -> None:
    op_h1 = WeightedHHartleyCosineAxialConv(channels=2, kernel_size=3, h=1.0)
    op_h2 = WeightedHHartleyCosineAxialConv(channels=2, kernel_size=3, h=2.0)
    with torch.no_grad():
        op_h2.kernel_h.copy_(op_h1.kernel_h)
        op_h2.kernel_w.copy_(op_h1.kernel_w)

    x = torch.randn(1, 2, 9, 9)
    y1 = op_h1(x)
    y2 = op_h2(x)
    assert torch.allclose(y2, 2.0 * y1, atol=1.0e-7, rtol=1.0e-5)


def test_shared_kernel_matches_repeated_per_channel_kernel() -> None:
    shared = WeightedHHartleyCosineAxialConv(channels=3, kernel_size=3, shared_kernel=True)
    per_channel = WeightedHHartleyCosineAxialConv(channels=3, kernel_size=3, shared_kernel=False)

    with torch.no_grad():
        per_channel.kernel_h.copy_(shared.kernel_h.expand_as(per_channel.kernel_h))
        per_channel.kernel_w.copy_(shared.kernel_w.expand_as(per_channel.kernel_w))

    x = torch.randn(2, 3, 10, 12)
    assert torch.allclose(shared(x), per_channel(x), atol=1.0e-7, rtol=1.0e-5)


def test_learnable_h_stays_positive_and_receives_gradient() -> None:
    operator = WeightedHHartleyCosineAxialConv(
        channels=2,
        kernel_size=3,
        h=1.0,
        learnable_h=True,
    )
    x = torch.randn(2, 2, 8, 8)
    loss = operator(x).pow(2).mean()
    loss.backward()

    assert operator.h_value.item() > 0.0
    assert operator.raw_h is not None
    assert operator.raw_h.grad is not None
    assert torch.isfinite(operator.raw_h.grad).all()


def test_hc_operator_kernel_gradients_are_finite() -> None:
    operator = WeightedHHartleyCosineAxialConv(channels=2, kernel_size=5)
    x = torch.randn(2, 2, 8, 8, requires_grad=True)
    loss = operator(x).abs().mean()
    loss.backward()

    assert operator.kernel_h.grad is not None
    assert operator.kernel_w.grad is not None
    assert torch.isfinite(operator.kernel_h.grad).all()
    assert torch.isfinite(operator.kernel_w.grad).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
