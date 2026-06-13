# Dedicated HC-U-Net ablation update

This update adds a permanent, code-level HC-U-Net ablation suite. No runtime
Python patching or generated YAML is required.

## Added model variants

- `hc_reference`
- `hc_without_hc_branch`
- `hc_shared_kernel`
- `hc_learnable_h`
- `hc_kernel5`
- `hc_identity_projection`
- `hc_no_channel_expansion`

Each variant is explicitly registered in
`src/models/proposal/hc_ablation.py`. The variant class enforces its defining
setting even when a caller provides a conflicting value.

## Added configurations

Seven complete YAML files were added under `configs/hc_ablation/`. All use an
architecture-only protocol with auxiliary supervision, boundary loss, HC/HF
regularization, and alpha warm-up disabled.

## Added runners

- `scripts/run_hc_ablation.py`
- `run_hc_ablation.sh`
- `bash run.sh hc-ablation`

The runner trains and evaluates only the seven HC variants and aggregates
multiple seeds into `multi_seed_summary.csv`.

## Added tests

`tests/test_hc_ablation_variants.py` validates:

- registry coverage;
- output shape and finite outputs;
- exact structural meaning of each ablation;
- exact identity behavior when alpha is zero;
- expected parameter-count changes;
- backward propagation for every variant;
- YAML completeness and fairness controls;
- runner/config synchronization;
- absence of inline Python patching in the shell runner.

`tests/test_hc_operator_contracts.py` validates:

- kernel-size constraints;
- shape preservation for kernels 3, 5, and 7;
- zero-input behavior;
- linear scaling with fixed h;
- equivalence between a shared kernel and repeated channel kernels;
- positivity and gradient flow of learnable h;
- finite kernel and input gradients.
