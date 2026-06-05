#!/usr/bin/env python3
"""Run the paper-fair benchmark variant with official ImageNet backbones enabled.

Only models that have official external encoders use auto-downloaded public
ImageNet checkpoints. U-Net-family and proposal models have no external official
classification backbone in this repo and remain randomly initialized.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    "unet",
    "attention_unet",
    "unet_cbam",
    "unetpp",
    "resunetpp",
    "pranet",
    "acsnet",
    "hardnet_mseg",
    "polyp_pvt",
    "caranet",
    "cfanet",
    "hsnet",
    "csca_unet",
    "proposal_hf_unet",
]


def main() -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "benchmark_all.py"),
        "--config-dir",
        "configs/paper_fair_pretrained",
        "--models",
        ",".join(MODELS),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
