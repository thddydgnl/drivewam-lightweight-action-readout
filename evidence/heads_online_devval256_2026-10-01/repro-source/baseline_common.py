"""Shared frozen DriveWAM initialization for observation-only baseline runs."""
import hashlib
from pathlib import Path
import runpy
import sys

import torch
from diffusers import AutoencoderKLWan

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "work/DriveWAM/src"))
evaluation = runpy.run_path(str(ROOT / "work/DriveWAM/src/navsim/eval.py"), run_name="baseline_frozen_evaluator")
from configs import VA_CONFIGS
from modules.navsim_model import WanTransformer3DNavSimModel
from utils.scheduler import FlowMatchScheduler

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

def sample_seed(token, generation_seed):
    # Explicit delimiter, digest slice and endianness are part of the protocol.
    return int.from_bytes(hashlib.sha256(f"{token}:{generation_seed}".encode()).digest()[:4], "big")

def load_policy():
    config = VA_CONFIGS["navsim_cfg"]
    base = ROOT / "checkpoints/lingbot-va-base"
    config.wan22_pretrained_model_name_or_path = str(base)
    transformer = WanTransformer3DNavSimModel.from_pretrained(
        ROOT / "checkpoints/drivewam-navsim/transformer", torch_dtype=torch.bfloat16,
        attn_mode="flashattn", local_files_only=True, ignore_mismatched_sizes=False)
    info = {"missing_keys": getattr(transformer, "_missing_keys", []),
            "unexpected_keys": getattr(transformer, "_unexpected_keys", [])}
    assert not any(info.values()), info
    transformer = transformer.eval().requires_grad_(False).to("cuda:0")
    assert len(transformer.blocks) == 30
    assert transformer.config.num_attention_heads * transformer.config.attention_head_dim == 3072
    vae, vae_info = AutoencoderKLWan.from_pretrained(base / "vae", torch_dtype=torch.bfloat16,
                  output_loading_info=True, local_files_only=True)
    assert not any(vae_info.get(k) for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]), vae_info
    vae = vae.eval().requires_grad_(False).to("cuda:0")
    cache_path = ROOT / "caches/prompt/navsim-four-prompts.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    assert set(cache) == set(config.command_prompts)
    assert all(t.shape == (512, 4096) and t.dtype == torch.bfloat16 and torch.isfinite(t).all() for t in cache.values())
    policy = evaluation["NavSimEvaluator"].__new__(evaluation["NavSimEvaluator"])
    policy.config, policy.device, policy.dtype = config, torch.device("cuda:0"), torch.bfloat16
    policy.patch_size, policy.cache_name = config.patch_size, "b10_b05_baseline"
    policy.transformer, policy.vae = transformer, vae
    policy.tokenizer = policy.text_encoder = None
    policy._prompt_cache = {p: t.to("cuda:0") for p, t in cache.items()}
    policy.video_scheduler = FlowMatchScheduler(shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
    policy.action_scheduler = FlowMatchScheduler(shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
    return policy, {"transformer": info, "vae": vae_info, "prompt_cache_sha256": file_sha(cache_path),
                    "transformer_parameters": sum(p.numel() for p in transformer.parameters()),
                    "vae_parameters": sum(p.numel() for p in vae.parameters())}
