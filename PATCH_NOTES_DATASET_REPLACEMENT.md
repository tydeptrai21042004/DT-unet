# Dataset replacement patch

Removed registry/session support for BUSI, DRIVE, and the generic custom entry.

Added:

- `kvasir_instrument`: official Simula direct ZIP.
- `hyper_kvasir_seg`: only the official segmentation subset ZIP; the full HyperKvasir archive is never downloaded.
- `montgomery_lung`: official NLM directory download with automatic left/right lung-mask union.

Session allocation:

- Session 3: Kvasir-Instrument + CVC-ColonDB HC comparison.
- Session 4: Montgomery Lung + HyperKvasir-SEG HC comparison.

HyperKvasir and Kvasir-SEG are never merged automatically. If both are present, filename overlap is checked and written to `data/processed/hyper_kvasir_seg/overlap_with_kvasir_seg.txt`.
