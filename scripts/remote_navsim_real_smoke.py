"""Run frozen original B10 on one real navtrain observation; verify GT invariance."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from navsim_current_dataset import load_policy_input, build_rollout_batch, NavsimCurrentDataset

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "work/DriveWAM/src"))
evaluation = runpy.run_path(str(ROOT / "work/DriveWAM/src/navsim/eval.py"), run_name="navsim_real_data_smoke")
Evaluator = evaluation["NavSimEvaluator"]
from configs import VA_CONFIGS
from dataset.navsim_dataset import NavSimEpisodeDataset
from modules.navsim_model import WanTransformer3DNavSimModel
from utils.scheduler import FlowMatchScheduler

def check_info(info):
    assert not any(info.get(key) for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]), info

if __name__ == "__main__":
    torch.set_num_threads(8)
    status_path = ROOT / "reports/navsim-real-b10-status.json"
    def status(stage):
        status_path.write_text(json.dumps({"stage": stage}, indent=2) + "\n")
        print(stage, flush=True)
    try:
        config = VA_CONFIGS["navsim_cfg"]
        config.wan22_pretrained_model_name_or_path = str(ROOT / "checkpoints/lingbot-va-base")
        token = (ROOT / "splits/smoke32-tokens.txt").read_text().splitlines()[0]
        rows = [json.loads(line) for line in (ROOT / "splits/pilot-manifest.jsonl").read_text().splitlines()]
        row = next(row for row in rows if row["scene_token"] == token)
        assert row["split"] == "head_train" and row["input_available"]
        assert len(NavsimCurrentDataset(ROOT / "splits/pilot-manifest.jsonl", "head_train")) == 4000
        assert len(NavsimCurrentDataset(ROOT / "splits/pilot-manifest.jsonl", "head_val")) == 1000
        observation = load_policy_input(row["input_path"])
        batch = build_rollout_batch(observation, config)
        # Reference path deliberately includes GT, solely to check equivalence.
        config.dataset_path = str(ROOT / "processed/reference_smoke")
        original_dataset = NavSimEpisodeDataset(config)
        assert len(original_dataset) == 1
        original = original_dataset[0]
        reference_batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
                           for key, value in original.items()}
        for key in ["history_poses", "ego_velocity_acc", "command_id", "actions_mask"]:
            assert torch.equal(batch[key], reference_batch[key]), key
        assert torch.equal(batch["images"][:, 0], reference_batch["images"][:, 0])
        assert batch["images"].shape == reference_batch["images"].shape
        cache_path = ROOT / "caches/prompt/navsim-four-prompts.pt"
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        assert set(cache) == set(config.command_prompts)
        status("loading_frozen_models")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        transformer = WanTransformer3DNavSimModel.from_pretrained(
            ROOT / "checkpoints/drivewam-navsim/transformer", torch_dtype=torch.bfloat16,
            attn_mode="flashattn", local_files_only=True, ignore_mismatched_sizes=False)
        info = {"missing_keys": getattr(transformer, "_missing_keys", []),
                "unexpected_keys": getattr(transformer, "_unexpected_keys", [])}
        check_info(info)
        transformer = transformer.eval().requires_grad_(False).to("cuda:0")
        vae, vae_info = AutoencoderKLWan.from_pretrained(ROOT / "checkpoints/lingbot-va-base/vae",
             torch_dtype=torch.bfloat16, output_loading_info=True, local_files_only=True)
        check_info(vae_info)
        vae = vae.eval().requires_grad_(False).to("cuda:0")
        policy = Evaluator.__new__(Evaluator)
        policy.config, policy.device, policy.dtype = config, torch.device("cuda:0"), torch.bfloat16
        policy.patch_size, policy.cache_name = config.patch_size, "real_navtrain_data_smoke"
        policy.transformer, policy.vae = transformer, vae
        policy.tokenizer = policy.text_encoder = None
        policy._prompt_cache = {prompt: embedding.to("cuda:0") for prompt, embedding in cache.items()}
        policy.video_scheduler = FlowMatchScheduler(shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
        policy.action_scheduler = FlowMatchScheduler(shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        calls = []
        hook = transformer.register_forward_pre_hook(lambda module, args, kwargs: calls.append(
            {"action_mode": kwargs.get("action_mode", False), "update_cache": kwargs.get("update_cache", 0)}), with_kwargs=True)
        status("real_observation_b10")
        start = time.perf_counter()
        with torch.inference_mode():
            actions, latents = policy.predict(batch, 3, 10, 20260930, video_denoise_fraction=0.6)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert len(calls) == 17 and torch.isfinite(actions).all() and torch.isfinite(latents).all()
        observation_calls = list(calls)
        calls.clear()
        status("reference_gt_future_invariance")
        with torch.inference_mode():
            ref_actions, ref_latents = policy.predict(reference_batch, 3, 10, 20260930, video_denoise_fraction=0.6)
        torch.cuda.synchronize()
        hook.remove()
        assert len(calls) == 17
        assert torch.equal(actions, ref_actions) and torch.equal(latents, ref_latents)
        trajectory = policy.pred_traj_to_poses(policy.actions_to_trajectory(actions))
        assert trajectory.shape == (8, 3) and np.isfinite(trajectory).all()
        with np.load(row["label_path"], allow_pickle=False) as target:
            gt = target["future_poses_4s"].copy()
        normalized_gt_roundtrip = policy.pred_traj_to_poses(policy.actions_to_trajectory(reference_batch["actions"]))
        roundtrip_error = float(np.max(np.abs(normalized_gt_roundtrip - gt)))
        assert roundtrip_error < 3e-5
        np.savez(ROOT / "reports/navsim-real-b10-output.npz", actions=actions.float().cpu().numpy(),
                 trajectory=trajectory, scene_token=np.asarray(token))
        report = {"scene_token": token, "log_name": row["log_name"], "split": "head_train",
            "input_path": row["input_path"], "input_sha256": row["input_sha256"],
            "prompt_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            "initializer": "test-only initializer with precomputed prompts; original predict unchanged",
            "transformer_loading_info": info, "vae_loading_info": vae_info,
            "num_video_steps": 3, "num_action_steps": 10, "video_denoise_fraction": 0.6, "seed": 20260930,
            "forward_calls": observation_calls, "actions_shape": list(actions.shape),
            "latents_shape": list(latents.shape), "trajectory_shape": list(trajectory.shape), "outputs_finite": True,
            "policy_input_contains_future_pixels_or_gt": False,
            "reference_with_real_future_rgb_and_gt_actions_has_bitwise_identical_predictions": True,
            "gt_normalization_denormalization_max_abs_error": roundtrip_error,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0), "elapsed_unwarmed_seconds": elapsed,
            "latency_benchmark": False, "driving_quality_evaluated": False, "navtest_used": False,
            "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        (ROOT / "reports/navsim-real-b10-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
        status("verified")
        print(json.dumps(report, indent=2))
    except Exception as error:
        status("failed: " + repr(error))
        raise
