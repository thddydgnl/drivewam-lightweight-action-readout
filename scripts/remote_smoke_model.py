"""Verify the frozen runtime on both GPUs; this is not a research benchmark."""
import importlib.metadata
import json
from pathlib import Path
import sys
import runpy
from datetime import datetime, timezone
import torch
from torch.nn import functional as F
from flash_attn import flash_attn_func

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "work/DriveWAM/src"))
from modules.navsim_model import WanTransformer3DNavSimModel
from modules.utils import load_vae, load_text_encoder, load_tokenizer
from dataset.navsim_dataset import NavSimEpisodeDataset
from configs import VA_CONFIGS
from utils.scheduler import FlowMatchScheduler
evaluation = runpy.run_path(str(ROOT / "work/DriveWAM/src/navsim/eval.py"),
                           run_name="drivewam_environment_smoke")

report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
          "torch": torch.__version__, "torch_cuda": torch.version.cuda,
          "torch_cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
          "packages": {p: importlib.metadata.version(p) for p in
                       ("torchvision", "diffusers", "transformers", "flash-attn", "numpy")},
          "source_imports": "passed", "evaluator_import": "passed",
          "pdm_available_in_model_environment": evaluation["_PDM_AVAILABLE"], "gpus": []}
assert torch.cuda.device_count() == 2
for index in range(torch.cuda.device_count()):
    device = torch.device("cuda", index)
    torch.manual_seed(20260930)
    with torch.cuda.device(index):
        assert torch.cuda.is_bf16_supported(including_emulation=False)
        q = torch.randn(2, 112, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        k, v = [torch.randn(2, 512, 4, 128, device=device, dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(2)]
        actual = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        reference = F.scaled_dot_product_attention(q.float().transpose(1, 2),
                    k.float().transpose(1, 2), v.float().transpose(1, 2)).transpose(1, 2)
        error = (actual.float() - reference).abs().max().item()
        torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
        actual.float().square().mean().backward()
        assert all(torch.isfinite(t.grad).all() for t in (q, k, v))
        head = torch.nn.Linear(3072, 24, device=device, dtype=torch.bfloat16)
        head(torch.randn(8, 3072, device=device, dtype=torch.bfloat16)).float().square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in head.parameters())
        torch.cuda.synchronize(index)
        report["gpus"].append({"index": index, "name": torch.cuda.get_device_name(index),
                               "capability": torch.cuda.get_device_capability(index),
                               "attention_test": {"q_tokens": 112, "kv_tokens": 512, "head_dim": 128},
                               "fa2_vs_fp32_sdpa_max_abs_error": error,
                               "fa2_backward_finite": True, "bf16_head_backward_finite": True})
        del q, k, v, actual, reference, head
        torch.cuda.empty_cache()

scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
scheduler.set_timesteps(3)
assert len(scheduler.timesteps) == 3
report["flow_scheduler_import_and_initialization"] = "passed"
(ROOT / "reports/model-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
