"""Run with torchrun --standalone --nproc_per_node=2 from the model venv."""
import json
import os
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", timeout=timedelta(seconds=60))
x = torch.tensor([dist.get_rank() + 1.0], device="cuda")
dist.all_reduce(x)
assert x.item() == 3.0, x
dist.barrier()
if dist.get_rank() == 0:
    root = Path.home() / "DriveWAM_Lightweight_Action_Readout"
    report = {"backend": dist.get_backend(), "world_size": dist.get_world_size(),
              "all_reduce_result": x.item(), "passed": True}
    (root / "reports/nccl-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
dist.destroy_process_group()
