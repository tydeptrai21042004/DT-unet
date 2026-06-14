# Kaggle disk-space fix

The balanced HC sessions are disk-safe by default.

- `DELETE_CHECKPOINTS_AFTER_EVAL=1`: removes each seed's checkpoint directories only after training and evaluation finish. CSV, JSON, LaTeX, histories, summaries, and aggregate results remain.
- `CLEAN_DOWNLOAD_ARCHIVES=1`: removes downloaded archives and temporary extraction directories after processed images/masks and split files have been created.
- Dependency installation uses `pip --no-cache-dir` and clears the pip cache.

To keep checkpoints intentionally, run:

```bash
DELETE_CHECKPOINTS_AFTER_EVAL=0 bash run_hc_session_3.sh
```

This is not recommended on Kaggle for multi-model, multi-seed sessions.
