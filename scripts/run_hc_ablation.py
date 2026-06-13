#!/usr/bin/env python3
"""Run the dedicated architecture-only HC-U-Net ablation suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HC_ABLATION_MODELS = [
    "hc_reference",
    "hc_without_hc_branch",
    "hc_shared_kernel",
    "hc_learnable_h",
    "hc_kernel5",
    "hc_identity_projection",
    "hc_no_channel_expansion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train, evaluate, and aggregate all dedicated HC-U-Net ablation "
            "variants without running HF-U-Net or unrelated baselines."
        )
    )
    parser.add_argument("--dataset", type=str, default="kvasir_seg")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument("--zip-path", type=str, default=None)
    parser.add_argument("--download-url", type=str, default=None)
    parser.add_argument("--download-dst", type=str, default=None)
    parser.add_argument("--kaggle-handle", type=str, default=None, help="Optional Kaggle dataset handle override.")
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-root", type=str, default="outputs_hc_ablation")
    parser.add_argument("--seeds", type=str, default="42,1,2")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--allow-insecure-download", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "benchmark_multi_seed.py"),
        "--models",
        ",".join(HC_ABLATION_MODELS),
        "--dataset",
        args.dataset,
        "--config-dir",
        "configs/hc_ablation",
        "--data-root",
        args.data_root,
        "--image-size",
        str(args.image_size),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--output-root",
        args.output_root,
        "--seeds",
        args.seeds,
    ]

    optional_values = [
        ("--source-dir", args.source_dir),
        ("--zip-path", args.zip_path),
        ("--download-url", args.download_url),
        ("--download-dst", args.download_dst),
        ("--kaggle-handle", args.kaggle_handle),
        ("--lr", args.lr),
    ]
    for flag, value in optional_values:
        if value is not None:
            cmd.extend([flag, str(value)])

    if args.save_predictions:
        cmd.append("--save-predictions")
    if args.save_visualizations:
        cmd.append("--save-visualizations")
    if args.allow_insecure_download:
        cmd.append("--allow-insecure-download")

    run(cmd)

    summary = PROJECT_ROOT / args.output_root / "results" / "tables" / "multi_seed_summary.csv"
    print(f"HC ablation completed. Summary: {summary}")


if __name__ == "__main__":
    main()
