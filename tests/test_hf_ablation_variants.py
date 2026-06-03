from __future__ import annotations

import ast
from pathlib import Path
import sys

import pytest
import torch
torch.set_num_threads(1)
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models  # noqa: F401  # side-effect registration
from src.models import build_model
from src.models.common.blocks import ConvNormAct, SqueezeExcitation
from src.models.proposal.hf_ablation import ConvBottleneck, FFTGFNetLikeBottleneck, IdentityTransform2d
from src.models.proposal.hf_bottleneck import FrequencyMixer, HFBottleneck
from src.models.registry import get_model_class

ABLATION_MODELS = [
    "unet",
    "unet_conv_bottleneck",
    "unet_fft_bottleneck",
    "proposal_hf_unet",
    "hf_unet_wo_hartley",
    "hf_unet_wo_fourier_kernel",
    "hf_unet_wo_residual",
    "hf_unet_encoder_stage4",
    "hf_unet_decoder_stage",
    "hf_unet_no_gate",
    "hf_unet_with_se",
    "hf_unet_identity_projection",
    "hf_unet_conv_projection",
    "hf_unet_low_rank_mixer",
    "hf_unet_high_rank_mixer",
    "hf_unet_hf_at_encoder0",
    "hf_unet_hf_at_encoder1",
    "hf_unet_hf_at_encoder2",
    "hf_unet_hf_at_encoder3",
    "hf_unet_hf_at_bottleneck",
    "hf_unet_hf_at_decoder3",
    "hf_unet_hf_at_decoder2",
    "hf_unet_hf_at_decoder1",
    "hf_unet_hf_at_decoder0",
]

PLACEMENT_MODELS = [
    "unet",
    "hf_unet_hf_at_encoder0",
    "hf_unet_hf_at_encoder1",
    "hf_unet_hf_at_encoder2",
    "hf_unet_hf_at_encoder3",
    "hf_unet_hf_at_bottleneck",
    "hf_unet_hf_at_decoder3",
    "hf_unet_hf_at_decoder2",
    "hf_unet_hf_at_decoder1",
    "hf_unet_hf_at_decoder0",
]


def _tiny_cfg(model_name: str) -> dict:
    if model_name == "unet":
        return {
            "in_channels": 3,
            "num_classes": 1,
            "channels": (1, 2, 4, 8, 16),
            "norm": "gn",
            "act": "relu",
        }
    return {
        "in_channels": 3,
        "num_classes": 1,
        "channels": (1, 2, 4, 8, 16),
        "norm": "gn",
        "act": "relu",
        "hf_block_norm": "gn",
        "hf_block_act": "gelu",
        "hf_expansion": 1.5,
        "hf_alpha": 0.25,
        "hf_alpha_start": 0.25,
        "hf_alpha_warmup_epochs": 0,
        "hf_projection": "linear",
        "use_gate": True,
        "use_hf_regularizer": False,
    }


@pytest.mark.parametrize("model_name", ABLATION_MODELS)
def test_compact_ablation_model_is_registered_and_outputs_correct_shape(model_name: str):
    get_model_class(model_name)
    model = build_model(model_name, config=_tiny_cfg(model_name)).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 1, 32, 32)
    assert torch.isfinite(y).all()


def test_compact_ablation_blocks_match_their_declared_purpose():
    conv = build_model("unet_conv_bottleneck", config=_tiny_cfg("unet_conv_bottleneck"))
    fft = build_model("unet_fft_bottleneck", config=_tiny_cfg("unet_fft_bottleneck"))
    no_hartley = build_model("hf_unet_wo_hartley", config=_tiny_cfg("hf_unet_wo_hartley"))
    no_kernel = build_model("hf_unet_wo_fourier_kernel", config=_tiny_cfg("hf_unet_wo_fourier_kernel"))
    no_residual = build_model("hf_unet_wo_residual", config=_tiny_cfg("hf_unet_wo_residual"))
    stage4 = build_model("hf_unet_encoder_stage4", config=_tiny_cfg("hf_unet_encoder_stage4"))
    decoder = build_model("hf_unet_decoder_stage", config=_tiny_cfg("hf_unet_decoder_stage"))
    no_gate = build_model("hf_unet_no_gate", config=_tiny_cfg("hf_unet_no_gate"))
    with_se = build_model("hf_unet_with_se", config=_tiny_cfg("hf_unet_with_se"))
    identity_projection = build_model("hf_unet_identity_projection", config=_tiny_cfg("hf_unet_identity_projection"))
    conv_projection = build_model("hf_unet_conv_projection", config=_tiny_cfg("hf_unet_conv_projection"))
    low_rank = build_model("hf_unet_low_rank_mixer", config=_tiny_cfg("hf_unet_low_rank_mixer"))
    high_rank = build_model("hf_unet_high_rank_mixer", config=_tiny_cfg("hf_unet_high_rank_mixer"))

    assert isinstance(conv.block, ConvBottleneck)
    assert isinstance(fft.block, FFTGFNetLikeBottleneck)
    assert isinstance(no_hartley.block, HFBottleneck)
    assert isinstance(no_hartley.block.hartley, IdentityTransform2d)
    assert isinstance(no_kernel.block, HFBottleneck)
    assert isinstance(no_kernel.block.mixer, torch.nn.Identity)
    assert isinstance(no_residual.block, HFBottleneck)
    assert no_residual.ablation == "wo_residual"
    assert stage4.placement_kind == "encoder"
    assert stage4.placement_label == "pre_bottleneck"
    assert decoder.placement_kind == "post_decoder"
    assert decoder.placement_label == "post_decoder"

    assert isinstance(no_gate.block, HFBottleneck)
    assert no_gate.block.use_gate is False
    assert isinstance(with_se.block.se, SqueezeExcitation)
    assert isinstance(identity_projection.block.pre, torch.nn.Identity)
    assert isinstance(identity_projection.block.post, torch.nn.Identity)
    assert isinstance(conv_projection.block.pre, ConvNormAct)
    assert isinstance(conv_projection.block.post, ConvNormAct)
    assert isinstance(low_rank.block.mixer, FrequencyMixer)
    assert low_rank.block.mixer.rank == 16
    assert isinstance(high_rank.block.mixer, FrequencyMixer)
    assert high_rank.block.mixer.rank == 32



def test_hf_placement_variants_target_expected_unet_locations():
    expected = {
        "hf_unet_hf_at_encoder0": ("encoder", "encoder0", 0, 2, 64),
        "hf_unet_hf_at_encoder1": ("encoder", "encoder1", 1, 4, 32),
        "hf_unet_hf_at_encoder2": ("encoder", "encoder2", 2, 8, 16),
        "hf_unet_hf_at_encoder3": ("encoder", "encoder3", 3, 16, 8),
        "hf_unet_hf_at_bottleneck": ("encoder", "bottleneck", 4, 32, 4),
        "hf_unet_hf_at_decoder3": ("decoder", "decoder3", 3, 16, 8),
        "hf_unet_hf_at_decoder2": ("decoder", "decoder2", 2, 8, 16),
        "hf_unet_hf_at_decoder1": ("decoder", "decoder1", 1, 4, 32),
        "hf_unet_hf_at_decoder0": ("decoder", "decoder0", 0, 2, 64),
    }
    for model_name, (kind, label, stage, channels, spatial) in expected.items():
        cfg = _tiny_cfg(model_name)
        cfg["channels"] = (2, 4, 8, 16, 32)
        model = build_model(model_name, config=cfg).eval()
        seen = []

        def hook(_module, inputs, _output):
            seen.append(tuple(inputs[0].shape))

        handle = model.block.register_forward_hook(hook)
        with torch.no_grad():
            y = model(torch.randn(1, 3, 64, 64))
        handle.remove()

        assert tuple(y.shape) == (1, 1, 64, 64)
        assert model.placement_kind == kind
        assert model.placement_label == label
        assert model.placement_stage == stage
        assert seen == [(1, channels, spatial, spatial)]


def test_ablation_configs_exist_build_and_are_architecture_only_fair():
    cfg_dir = PROJECT_ROOT / "configs" / "ablation"
    for model_name in ABLATION_MODELS:
        cfg_path = cfg_dir / f"{model_name}.yaml"
        assert cfg_path.exists(), model_name
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg["model"]["name"] == model_name

        # Strict architecture-only fairness: no training helpers or proposal-only boosts.
        assert cfg["data"]["batch_size"] == 6
        assert cfg["train"]["aux_loss_weight"] == 0.0
        assert cfg["train"].get("aux_warmup_epochs", 0) == 0
        assert cfg["train"].get("aux_ramp_epochs", 0) == 0
        if model_name != "unet":
            assert cfg["model"].get("use_hf_regularizer") is False
            assert cfg["model"].get("hf_alpha_start") == cfg["model"].get("hf_alpha")
            assert cfg["model"].get("hf_alpha_warmup_epochs") == 0

        model_cfg = {k: v for k, v in cfg["model"].items() if k != "name"}
        # Use tiny channels for speed while preserving the model's ablation key.
        model_cfg.update(_tiny_cfg(model_name))
        build_model(model_name, config=model_cfg)



def test_placement_ablation_configs_and_runner_are_focused_on_hf_location_only():
    cfg_dir = PROJECT_ROOT / "configs" / "placement_ablation"
    cfg_models = sorted(p.stem for p in cfg_dir.glob("*.yaml"))
    assert sorted(PLACEMENT_MODELS) == cfg_models

    for model_name in PLACEMENT_MODELS:
        cfg = yaml.safe_load((cfg_dir / f"{model_name}.yaml").read_text(encoding="utf-8"))
        assert cfg["model"]["name"] == model_name
        assert cfg["data"]["batch_size"] == 6
        assert cfg["train"]["aux_loss_weight"] == 0.0
        if model_name != "unet":
            assert cfg["model"]["use_hf_regularizer"] is False
            assert cfg["model"]["hf_alpha_start"] == cfg["model"]["hf_alpha"]
            assert cfg["model"]["hf_alpha_warmup_epochs"] == 0

    script_path = PROJECT_ROOT / "scripts" / "run_hf_placement_ablation.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    script_models = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PLACEMENT_MODELS":
                    script_models = ast.literal_eval(node.value)
    assert script_models == PLACEMENT_MODELS


def test_run_compact_ablation_script_matches_config_set():
    script_path = PROJECT_ROOT / "scripts" / "run_compact_hf_ablation.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    script_models = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ABLATION_MODELS":
                    script_models = ast.literal_eval(node.value)
    assert script_models == ABLATION_MODELS

    cfg_models = sorted(p.stem for p in (PROJECT_ROOT / "configs" / "ablation").glob("*.yaml"))
    assert sorted(script_models) == cfg_models


def test_hf_ablation_backward_smoke():
    model = build_model("hf_unet_identity_projection", config=_tiny_cfg("hf_unet_identity_projection"))
    model.train()
    x = torch.randn(2, 3, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
