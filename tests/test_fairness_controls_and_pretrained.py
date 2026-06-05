from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.output_utils import compute_supervised_loss
from src.losses import BCEDiceLoss
from src.models.common.official_backbones import DEFAULT_CHECKPOINT_URLS


def test_compute_supervised_loss_can_disable_aux_and_boundary_losses():
    masks = torch.ones(1, 1, 8, 8)
    output = {
        "main": torch.zeros(1, 1, 8, 8),
        "aux": [torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8)],
        "boundary": torch.zeros(1, 1, 8, 8),
    }
    loss_fn = BCEDiceLoss()
    enabled_loss, enabled_logs, _ = compute_supervised_loss(
        output,
        masks,
        main_loss_fn=loss_fn,
        aux_loss_fn=loss_fn,
        aux_weights=[1.0, 1.0],
        boundary_loss_fn=loss_fn,
        boundary_weight=1.0,
    )
    disabled_loss, disabled_logs, _ = compute_supervised_loss(
        output,
        masks,
        main_loss_fn=loss_fn,
        aux_loss_fn=loss_fn,
        aux_weights=[1.0, 1.0],
        boundary_loss_fn=loss_fn,
        boundary_weight=1.0,
        use_aux_outputs=False,
        use_boundary_output=False,
    )
    assert enabled_loss > disabled_loss
    assert "aux_loss" in enabled_logs
    assert "boundary_loss" in enabled_logs
    assert "aux_loss" not in disabled_logs
    assert "boundary_loss" not in disabled_logs


def test_csca_paper_fair_uses_effective_batch_size_six():
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "paper_fair" / "csca_unet.yaml").read_text())
    assert int(cfg["data"]["batch_size"]) == 2
    assert int(cfg["train"]["gradient_accumulation_steps"]) == 3
    assert int(cfg["data"]["batch_size"]) * int(cfg["train"]["gradient_accumulation_steps"]) == 6


def test_strict_no_aux_configs_disable_side_losses():
    for path in (PROJECT_ROOT / "configs" / "strict_no_aux").glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text())
        assert cfg["train"]["use_aux_outputs_loss"] is False
        assert cfg["train"]["use_boundary_loss"] is False


def test_default_public_backbone_urls_are_available_for_auto_download():
    assert DEFAULT_CHECKPOINT_URLS["res2net50_v1b_26w_4s"].endswith("res2net50_v1b_26w_4s-3cf99910.pth")
    assert DEFAULT_CHECKPOINT_URLS["pvt_v2_b2"].endswith("pvt_v2_b2.pth")
    assert DEFAULT_CHECKPOINT_URLS["hardnet68"].endswith("hardnet68-5d684880.pth")
