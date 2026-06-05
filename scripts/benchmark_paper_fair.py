#!/usr/bin/env python3
"""Run the recommended paper-faithful but still training-fair benchmark.

This mode keeps the same training/evaluation recipe for all methods and avoids
pretrained backbone advantages, while enabling the paper-defining architecture
parts of each baseline: full non-fast backbone variants, faithful side outputs,
deep supervision, and boundary loss where applicable.
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
        "configs/paper_fair",
        "--models",
        ",".join(MODELS),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
