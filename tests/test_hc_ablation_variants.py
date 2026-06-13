from __future__ import annotations

import ast
from pathlib import Path
import sys

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models  # noqa: F401  # side-effect registration
from src.models import build_model
from src.models.proposal.hc_ablation import HC_ABLATION_MODEL_NAMES
from src.models.proposal.hc_bottleneck import WeightedHHartleyCosineAxialConv
from src.models.registry import get_model_class


torch.set_num_threads(1)

EXPECTED_VARIANTS = [
    "hc_reference",
    "hc_without_hc_branch",
    "hc_shared_kernel",
    "hc_learnable_h",
    "hc_kernel5",
    "hc_identity_projection",
    "hc_no_channel_expansion",
]


def _tiny_config() -> dict:
    return {
        "in_channels": 3,
        "num_classes": 1,
        "channels": (2, 4, 8, 16, 32),
        "norm": "gn",
        "act": "relu",
        "hf_alpha": 0.5,
        "hf_alpha_start": 0.5,
        "hf_alpha_warmup_epochs": 0,
        "hf_expansion": 1.5,
        "hf_dropout": 0.0,
        "use_hf_regularizer": False,
        "hf_block_norm": "gn",
        "hf_block_act": "gelu",
        "mixer_act": "gelu",
        "use_se": False,
        "use_gate": False,
        "decoder_use_cbam": False,
        "identity_init": True,
        "hf_projection": "linear",
        "hc_kernel_size": 3,
        "hc_h": 1.0,
        "hc_learnable_h": False,
        "hc_shared_kernel": False,
    }


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def test_hc_ablation_manifest_is_exact_and_registered() -> None:
    assert HC_ABLATION_MODEL_NAMES == EXPECTED_VARIANTS
    for model_name in EXPECTED_VARIANTS:
        cls = get_model_class(model_name)
        assert cls.__name__


@pytest.mark.parametrize("model_name", EXPECTED_VARIANTS)
def test_every_hc_ablation_builds_and_preserves_segmentation_shape(model_name: str) -> None:
    model = build_model(model_name, config=_tiny_config()).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 1, 32, 32)
    assert torch.isfinite(y).all()
    assert model.hc_bottleneck.use_gate is False
    assert model.hc_ablation_name == model_name


def test_hc_ablation_variants_match_their_declared_purpose() -> None:
    reference = build_model("hc_reference", config=_tiny_config())
    without_branch = build_model("hc_without_hc_branch", config=_tiny_config())
    shared = build_model("hc_shared_kernel", config=_tiny_config())
    learnable_h = build_model("hc_learnable_h", config=_tiny_config())
    kernel5 = build_model("hc_kernel5", config=_tiny_config())
    identity_projection = build_model("hc_identity_projection", config=_tiny_config())
    no_expansion = build_model("hc_no_channel_expansion", config=_tiny_config())

    assert reference.hc_bottleneck.alpha == pytest.approx(0.5)
    assert reference.hc_bottleneck.hc_conv.kernel_size == 3
    assert reference.hc_bottleneck.hc_conv.shared_kernel is False
    assert reference.hc_bottleneck.hc_conv.raw_h is None
    assert isinstance(reference.hc_bottleneck.pre, torch.nn.Conv2d)
    assert isinstance(reference.hc_bottleneck.post, torch.nn.Conv2d)
    assert reference.hc_bottleneck.mixer[0].out_channels == 48

    assert without_branch.hc_bottleneck.alpha == pytest.approx(0.0)
    assert without_branch.hf_alpha_target == pytest.approx(0.0)
    assert without_branch.hf_alpha_start == pytest.approx(0.0)

    assert shared.hc_bottleneck.hc_conv.shared_kernel is True
    assert shared.hc_bottleneck.hc_conv.kernel_h.shape[0] == 1
    assert shared.hc_bottleneck.hc_conv.kernel_w.shape[0] == 1

    assert isinstance(learnable_h.hc_bottleneck.hc_conv.raw_h, torch.nn.Parameter)
    assert learnable_h.hc_bottleneck.hc_conv.h_value.item() > 0.0

    assert kernel5.hc_bottleneck.hc_conv.kernel_size == 5
    assert kernel5.hc_bottleneck.hc_conv.kernel_h.shape[-2] == 5
    assert kernel5.hc_bottleneck.hc_conv.kernel_w.shape[-1] == 5

    assert isinstance(identity_projection.hc_bottleneck.pre, torch.nn.Identity)
    assert isinstance(identity_projection.hc_bottleneck.post, torch.nn.Identity)

    assert no_expansion.hc_bottleneck.mixer[0].in_channels == 32
    assert no_expansion.hc_bottleneck.mixer[0].out_channels == 32
    assert no_expansion.hc_bottleneck.mixer[-1].in_channels == 32


def test_hc_without_branch_bottleneck_is_exact_identity() -> None:
    model = build_model("hc_without_hc_branch", config=_tiny_config()).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = model.hc_bottleneck(x)
    assert torch.equal(y, x)


def test_hc_ablation_parameter_budgets_change_in_expected_direction() -> None:
    counts = {
        name: _parameter_count(build_model(name, config=_tiny_config()))
        for name in EXPECTED_VARIANTS
    }
    assert counts["hc_without_hc_branch"] == counts["hc_reference"]
    assert counts["hc_shared_kernel"] < counts["hc_reference"]
    assert counts["hc_learnable_h"] == counts["hc_reference"] + 1
    assert counts["hc_kernel5"] > counts["hc_reference"]
    assert counts["hc_identity_projection"] < counts["hc_reference"]
    assert counts["hc_no_channel_expansion"] < counts["hc_reference"]


@pytest.mark.parametrize("model_name", EXPECTED_VARIANTS)
def test_every_hc_ablation_supports_backward(model_name: str) -> None:
    model = build_model(model_name, config=_tiny_config()).train()
    x = torch.randn(2, 3, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    finite_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None and torch.isfinite(parameter.grad).all()
    ]
    assert finite_gradients, model_name


def test_hc_ablation_configs_are_complete_and_architecture_only() -> None:
    config_dir = PROJECT_ROOT / "configs" / "hc_ablation"
    assert sorted(path.stem for path in config_dir.glob("*.yaml")) == sorted(EXPECTED_VARIANTS)

    for model_name in EXPECTED_VARIANTS:
        path = config_dir / f"{model_name}.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["model"]["name"] == model_name
        assert cfg["experiment"]["name"] == model_name
        assert cfg["ablation"]["suite"] == "hc_unet"
        assert cfg["ablation"]["variant"] == model_name
        assert cfg["ablation"]["purpose"]
        assert cfg["data"]["batch_size"] == 6
        assert cfg["train"]["aux_loss_weight"] == 0.0
        assert cfg["train"]["use_aux_outputs_loss"] is False
        assert cfg["train"]["use_boundary_loss"] is False
        assert cfg["eval"]["include_aux_loss"] is False
        assert cfg["model"]["use_hf_regularizer"] is False
        assert cfg["model"]["use_gate"] is False
        assert cfg["model"]["hf_alpha_warmup_epochs"] == 0

        model_cfg = {key: value for key, value in cfg["model"].items() if key != "name"}
        model_cfg.update(_tiny_config())
        build_model(model_name, config=model_cfg)


def test_hc_ablation_runner_matches_config_set() -> None:
    script = PROJECT_ROOT / "scripts" / "run_hc_ablation.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    script_models = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "HC_ABLATION_MODELS":
                    script_models = ast.literal_eval(node.value)
    assert script_models == EXPECTED_VARIANTS

    shell_runner = (PROJECT_ROOT / "run_hc_ablation.sh").read_text(encoding="utf-8")
    assert "scripts/run_hc_ablation.py" in shell_runner
    assert "python - <<" not in shell_runner
    assert "proposal_hf_unet" not in shell_runner
