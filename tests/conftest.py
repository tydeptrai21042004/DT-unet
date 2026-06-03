from __future__ import annotations

import torch

# Keep CPU-only CI/local smoke tests fast and deterministic. The benchmark
# scripts still use the user's normal PyTorch threading unless they run pytest.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
