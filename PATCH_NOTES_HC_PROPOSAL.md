# New proposal variants: no-gate HF and h-Hartley-cosine convolution

This patch adds two proposal variants beside the old `proposal_hf_unet`:

1. `proposal_hf_unet_no_gate`
   - Same HF bottleneck as the original proposal.
   - Residual gate disabled (`use_gate: false`).
   - This promotes the best small-subset ablation finding into an explicit proposal model.

2. `proposal_hc_unet_no_gate`
   - Replaces the HF transform/mixer path with a finite axial implementation of the weighted discrete h-Hartley-cosine convolution

     (f *^gamma g)(nh) = h/2 sum_m f(mh)[g(nh-mh-h)+g(nh-mh+h)+g(nh+mh+h)+g(nh+mh-h)].

   - Uses no residual gate by default.
   - Keeps the original `proposal_hf_unet` unchanged.

Run with:

```bash
python scripts/train_one.py --model proposal_hc_unet_no_gate --config configs/proposal_hc_unet_no_gate.yaml --dataset kvasir_seg --data-root data_kvasir --device cuda
```

Architecture ablation now includes the two new proposal variants in `scripts/run_compact_hf_ablation.py`.
