"""Retain evidence for the server's legacy OpenBLAS kernel workaround."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
probe = r'''
import json
import numpy as np
import torch
from threadpoolctl import threadpool_info
torch.set_num_threads(1)
generator = torch.Generator().manual_seed(20260930)
a = torch.randn(40, 40, dtype=torch.float64, generator=generator)
matrix = a.T @ a + torch.eye(40, dtype=torch.float64)
reference = torch.linalg.pinv(matrix).numpy()
actual = np.linalg.pinv(matrix.numpy())
print(json.dumps({"blas": threadpool_info(),
    "pinv_identity_max_abs_error": float(np.abs(np.linalg.pinv(np.eye(40)) - np.eye(40)).max()),
    "pinv_spd_vs_torch_max_abs_error": float(np.abs(actual - reference).max())}))
'''
records = []
for coretype in (None, "Haswell"):
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    environment.pop("OPENBLAS_CORETYPE", None)
    if coretype:
        environment["OPENBLAS_CORETYPE"] = coretype
    result = subprocess.check_output([sys.executable, "-c", probe], env=environment, text=True)
    record = {"forced_coretype": coretype, **json.loads(result)}
    records.append(record)
fixed = records[1]
assert fixed["pinv_identity_max_abs_error"] < 1e-12
assert fixed["pinv_spd_vs_torch_max_abs_error"] < 1e-10
(ROOT / "reports/blas-diagnostic.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records, indent=2))
