#!/usr/bin/env python3
"""Run the HF placement-only ablation suite.

This runner answers one question: where should the HF block be inserted?
It intentionally excludes non-placement controls such as no-gate, SE, projection,
and mixer-capacity variants.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HF placement-only ablation training/evaluation.")
    parser.add_argument("--dataset", type=str, default="cvc_clinicdb")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-root", type=str, default="outputs_hf_placement_ablation")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    models = ",".join(PLACEMENT_MODELS)
    common = [
        "--models", models,
        "--dataset", args.dataset,
        "--config-dir", "configs/placement_ablation",
        "--data-root", args.data_root,
        "--image-size", str(args.image_size),
        "--seed", str(args.seed),
        "--device", args.device,
        "--output-root", args.output_root,
    ]
    for flag, value in [("--batch-size", args.batch_size), ("--epochs", args.epochs), ("--num-workers", args.num_workers)]:
        if value is not None:
            common += [flag, str(value)]

    run([sys.executable, str(PROJECT_ROOT / "scripts" / "train_all.py"), *common])
    if not args.skip_eval:
        run([sys.executable, str(PROJECT_ROOT / "scripts" / "eval_all.py"), *common])
        run([sys.executable, str(PROJECT_ROOT / "scripts" / "export_results.py"), "--output-root", args.output_root])


if __name__ == "__main__":
    main()
