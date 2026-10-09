"""Precompute the four prompts and run B10 on synthetic input on one 24GB GPU.

This validates runtime/memory, not driving quality, latency, or real-scene PDMS.
It calls the pinned evaluator unchanged, using a test-only initializer so that
UMT5 is released after prompt encoding rather than kept on the policy GPU.
"""
import gc
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
from datetime import datetime, timezone
import torch
from diffusers import AutoencoderKLWan
from transformers import UMT5EncoderModel

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "work/DriveWAM/src"))
evaluation = runpy.run_path(str(ROOT / "work/DriveWAM/src/navsim/eval.py"),
                           run_name="drivewam_checkpoint_smoke")
Evaluator = evaluation["NavSimEvaluator"]
from configs import VA_CONFIGS
from modules.navsim_model import WanTransformer3DNavSimModel
from modules.utils import load_tokenizer
from utils.scheduler import FlowMatchScheduler

status_path = ROOT / "reports/checkpoint-smoke-status.json"
report = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
          "input": "synthetic constant RGB / zero ego history; not a benchmark",
          "source_modified": False, "initializer": "test-only initializer with precomputed prompts"}

def stage(name):
    status_path.write_text(json.dumps({"stage": name}, indent=2) + "\n")
    print(name, flush=True)

def verify_info(label, info):
    report[label + "_loading_info"] = info
    assert not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), info

torch.set_num_threads(8)
base = ROOT / "checkpoints/lingbot-va-base"
config = VA_CONFIGS["navsim_cfg"]
config.wan22_pretrained_model_name_or_path = str(base)
stage("encoding_four_prompts_on_gpu1")
encoder, info = UMT5EncoderModel.from_pretrained(base / "text_encoder",
                         torch_dtype=torch.bfloat16, output_loading_info=True, local_files_only=True)
verify_info("text_encoder", info)
encoder = encoder.eval().requires_grad_(False).to("cuda:1")
prompt_evaluator = Evaluator.__new__(Evaluator)
prompt_evaluator.config = config
prompt_evaluator.device = torch.device("cuda:1")
prompt_evaluator.dtype = torch.bfloat16
prompt_evaluator.tokenizer = load_tokenizer(str(base / "tokenizer"))
prompt_evaluator.text_encoder = encoder
prompt_evaluator._prompt_cache = {}
with torch.inference_mode():
    prompt_evaluator._encode_prompt(config.command_prompts)
cache = {p: value.cpu() for p, value in prompt_evaluator._prompt_cache.items()}
assert len(cache) == 4 and all(t.shape == (512, 4096) and torch.isfinite(t).all() for t in cache.values())
cache_path = ROOT / "caches/prompt/navsim-four-prompts.pt"
torch.save(cache, cache_path)
report["prompt_cache"] = {"path": str(cache_path), "prompts": list(cache),
                         "shape_each": [512, 4096], "dtype": "bfloat16",
                         "sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                         "encoder_peak_allocated_bytes": torch.cuda.max_memory_allocated(1)}
del prompt_evaluator, encoder
gc.collect()
torch.cuda.empty_cache()

stage("loading_frozen_transformer_and_vae_on_gpu0")
torch.cuda.set_device(0)
torch.cuda.reset_peak_memory_stats(0)
transformer = WanTransformer3DNavSimModel.from_pretrained(
    ROOT / "checkpoints/drivewam-navsim/transformer", torch_dtype=torch.bfloat16,
    attn_mode="flashattn", local_files_only=True, ignore_mismatched_sizes=False)
# The pinned override returns the model only and retains these two diagnostics.
info = {"missing_keys": getattr(transformer, "_missing_keys", []),
        "unexpected_keys": getattr(transformer, "_unexpected_keys", []),
        "ignore_mismatched_sizes": False}
verify_info("transformer", info)
transformer = transformer.eval().requires_grad_(False).to("cuda:0")
vae, info = AutoencoderKLWan.from_pretrained(base / "vae", torch_dtype=torch.bfloat16,
                                           output_loading_info=True, local_files_only=True)
verify_info("vae", info)
vae = vae.eval().requires_grad_(False).to("cuda:0")
report["transformer_parameters"] = sum(p.numel() for p in transformer.parameters())
report["vae_parameters"] = sum(p.numel() for p in vae.parameters())
report["resident_allocated_bytes"] = torch.cuda.memory_allocated(0)

policy = Evaluator.__new__(Evaluator)
policy.config = config
policy.device = torch.device("cuda:0")
policy.dtype = torch.bfloat16
policy.patch_size = config.patch_size
policy.cache_name = "environment_synthetic_b10"
policy.transformer = transformer
policy.vae = vae
policy.tokenizer = None
policy.text_encoder = None
policy._prompt_cache = {p: t.to("cuda:0") for p, t in cache.items()}
policy.video_scheduler = FlowMatchScheduler(shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
policy.action_scheduler = FlowMatchScheduler(shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
actions = torch.zeros(1, 30, 2, 8, 1)
mask = torch.zeros_like(actions, dtype=torch.bool)
mask[:, :3, 1] = True
batch = {"images": torch.full((1, 5, 3, 256, 448), 127, dtype=torch.uint8),
         "prompt": [config.command_prompts[1]], "ego_velocity_acc": torch.zeros(1, 4),
         "history_poses": torch.zeros(1, 4, 3), "command_id": torch.tensor([1]),
         "actions": actions, "actions_mask": mask}
calls = []
def count_forward(module, args, kwargs):
    calls.append({"action_mode": kwargs.get("action_mode", False),
                  "update_cache": kwargs.get("update_cache", 0)})
hook = transformer.register_forward_pre_hook(count_forward, with_kwargs=True)
stage("running_synthetic_b10")
start = time.perf_counter()
with torch.inference_mode():
    predicted_actions, predicted_latents = policy.predict(batch, num_video_steps=3,
           num_action_steps=10, seed=20260930, video_denoise_fraction=0.6)
torch.cuda.synchronize(0)
hook.remove()
assert torch.isfinite(predicted_actions).all() and torch.isfinite(predicted_latents).all()
assert predicted_actions.shape == (1, 30, 2, 8, 1)
assert len(calls) == 17, calls
np_trajectory = policy.actions_to_trajectory(predicted_actions)
report["synthetic_b10"] = {"num_video_steps": 3, "num_action_steps": 10,
    "video_denoise_fraction": 0.6, "seed": 20260930, "forward_calls": calls,
    "actions_shape": list(predicted_actions.shape), "latents_shape": list(predicted_latents.shape),
    "trajectory_shape": list(np_trajectory.shape), "outputs_finite": True,
    "elapsed_unwarmed_seconds": time.perf_counter() - start,
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
    "latency_benchmark": False, "driving_quality_evaluated": False}
torch.save({"actions": predicted_actions.cpu(), "latents": predicted_latents.cpu()},
           ROOT / "reports/synthetic-b10-output.pt")
report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
(ROOT / "reports/checkpoint-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
stage("passed")
print(json.dumps(report, indent=2))
