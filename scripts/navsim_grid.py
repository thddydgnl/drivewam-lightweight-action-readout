"""Several opt action-step counts from one upstream NAVSIM `predict` (pre-registration 2026-11-01 §1.1).

`predict_multi` follows `NavSimEvaluator.predict` line by line (VAE encoding, cache priming, one
future latent of `video_steps` denoising steps + terminal cache pass), then runs the opt action
chunk once per requested step count. The opt chunk never writes the cache, and each count restarts
from the RNG state left by the video passes, so every output equals a separate
`predict(batch, video_steps, k, seed)` with the opt action chunk. Callers check this bitwise.
"""
import re

import torch

def parse(models):
    """'V0-A1,V0-A2,V1-A1' -> {0: [1, 2], 1: [1]} (video steps -> action steps, input order)."""
    plan = {}
    for name in models:
        m = re.fullmatch(r"V(\d)-A(\d+)", name)
        assert m, name
        plan.setdefault(int(m.group(1)), []).append(int(m.group(2)))
    return plan

@torch.no_grad()
def predict_multi(policy, opt_action, batch, video_steps, action_steps, seed, video_denoise_fraction=0.6):
    images = batch["images"][:1].to(policy.device)
    prompt = batch["prompt"][:1] if isinstance(batch["prompt"], list) else [batch["prompt"][0]]
    ego_cond = {"velocity_acc": batch["ego_velocity_acc"][:1].to(policy.device),
                "history_poses": batch["history_poses"][:1].to(policy.device),
                "command_ids": batch["command_id"][:1].to(policy.device)}
    first_frame_latent = policy._encode_images(images[:, :1]).to(policy.dtype)
    prompt_emb = policy._encode_prompt(prompt)
    action_cond = torch.zeros_like(batch["actions"][:1, :, 0:1], device=policy.device, dtype=policy.dtype)
    future_action_mask = batch["actions_mask"][:1, :, 1:2].bool().to(policy.device)
    devices = policy._rng_devices()
    out = {}
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        policy._prime_rollout_cache(first_frame_latent, action_cond, prompt_emb, ego_cond)
        try:
            shape = (first_frame_latent.shape[0], first_frame_latent.shape[1], 1,
                     first_frame_latent.shape[-2], first_frame_latent.shape[-1])
            num_future_latents = (images.shape[1] - 1) // 4
            assert num_future_latents == 1
            for frame_index in range(1, 1 + num_future_latents):
                policy._rollout_video_frame(frame_index=frame_index, latent_shape=shape, prompt_emb=prompt_emb,
                                            ego_cond=ego_cond, num_steps=video_steps,
                                            denoise_fraction=video_denoise_fraction)
            cpu_state = torch.get_rng_state()
            cuda_states = [torch.cuda.get_rng_state(d) for d in devices]
            for k in action_steps:
                torch.set_rng_state(cpu_state)
                for d, s in zip(devices, cuda_states):
                    torch.cuda.set_rng_state(s, d)
                future_action = opt_action(chunk_index=1,
                    action_shape=(action_cond.shape[0], action_cond.shape[1], 1, action_cond.shape[-2], action_cond.shape[-1]),
                    action_mask=future_action_mask, prompt_emb=prompt_emb, ego_cond=ego_cond, num_steps=k)
                out[k] = torch.cat([action_cond, future_action], dim=2)
        finally:
            policy.transformer.clear_cache(policy.cache_name)
    return out
