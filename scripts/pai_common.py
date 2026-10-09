"""Shared PhysicalAI-AV runtime (pre-registration 사전등록_PhysicalAI_2026-10-04).

* Upstream `PhysicalAIEvaluator` / `PhysicalAITestDataset` from the pinned DriveWAM tree are
  reused. Only two things change: frames come from `frames_1fps.npz` written by
  `remote_pai_process.py` (same decode path), and prompt embeddings come from a precomputed
  cache built with the upstream `_encode_single_prompt` (UMT5 is never co-resident).
* `rollout` re-implements `predict_autoregressive` step by step so the same loop can run
  baselines (raw / opt) and student early exits; `B10`/`B05` parity with the upstream function
  is checked bitwise by `remote_pai_parity.py`. Students consume RNG exactly like the
  baseline (video noise, then action noise per chunk) so their first-pass noise is identical.
"""
import hashlib
import json
from pathlib import Path
import pickle
import runpy
import sys

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
SRC = ROOT / "work/DriveWAM/src"
sys.path.insert(0, str(SRC))
UP = runpy.run_path(str(SRC / "physicalai/eval.py"), run_name="pai_frozen_evaluator")
from configs import VA_CONFIGS                                       # noqa: E402
from dataset.physicalai_dataset import PhysicalAITestDataset         # noqa: E402
from modules.physicalai_model import WanTransformer3DPhysicalAIModel  # noqa: E402
from modules.utils import load_vae                                   # noqa: E402

PROCESSED = ROOT / "processed/physicalai-v1"
LAYERS = [9, 19, 29]
CKPT_SHA = "ad90feae1a26007d13743ecb7896887ae7f9906168e5652d57b5406073fa0d98"

def prompt_key(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def config():
    cfg = VA_CONFIGS["physicalai_cfg"]
    cfg.wan22_pretrained_model_name_or_path = str(ROOT / "checkpoints/lingbot-va-base")
    cfg.dataset_path = str(PROCESSED)
    cfg.split = "test"
    return cfg

class ClipData(PhysicalAITestDataset):
    """Upstream __getitem__ verbatim; frames from the stored 1 fps decode (same values)."""
    def __init__(self, cfg, clip_ids, prompts_store=None, prompts_dir=None):
        self.config, self.data_root, self.clip_ids = cfg, PROCESSED, list(clip_ids)
        self.video_fps, self.target_video_fps = 10, int(getattr(cfg, "target_video_fps", 1))
        self.frame_skip = self.video_fps // self.target_video_fps
        self.action_chunk_steps = int(cfg.action_chunk_steps)
        self.image_height, self.image_width = int(cfg.height), int(cfg.width)
        self.action_channel_ids = list(cfg.used_action_channel_ids)
        self.actions_q01 = torch.tensor(cfg.norm_stat["q01"], dtype=torch.float32)
        self.actions_q99 = torch.tensor(cfg.norm_stat["q99"], dtype=torch.float32)
        self.prompt = str(cfg.prompt)
        self.prompts_store, self.prompts_dir = prompts_store, (Path(prompts_dir) if prompts_dir else None)
    def _load_ego_data(self, clip_id, max_retries=3, retry_delay=1.0):
        with open(PROCESSED / clip_id / "camera_front_wide_120fov_ego.pkl", "rb") as f:
            return pickle.load(f)
    def _load_video_frames(self, clip_id, frame_indices):
        with np.load(PROCESSED / clip_id / "frames_1fps.npz") as d:
            stored, n_frames, stride = d["frames"], int(d["n_frames"]), int(d["stride"])
        frames = []
        for fi in frame_indices:                                     # upstream semantics
            if 0 <= fi < n_frames:
                assert fi % stride == 0, (clip_id, fi)
                frames.append(stored[fi // stride])
            elif frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8))
        return np.stack(frames, axis=0)

def collate(item):
    batch = {}
    for k, v in item.items():
        if k == "prompts":
            batch[k] = [[p] for p in v]
        elif k == "scene_token":
            batch[k] = [v]
        else:
            batch[k] = v[None]
    return batch

class PromptCache:
    def __init__(self, files):
        self.index = {}
        for f in files:
            meta = json.loads(Path(str(f) + ".json").read_text())
            for key in meta["keys"]:
                self.index[key] = str(f)
    def tensor(self, prompt, length=512, dtype=torch.bfloat16):
        key = prompt_key(prompt)
        with safe_open(self.index[key], framework="pt", device="cpu") as s:
            valid = s.get_tensor(key)
        return torch.cat([valid, valid.new_zeros(length - valid.shape[0], valid.shape[1])]).to(dtype)

def load_policy(device="cuda:0", prompt_cache=None):
    cfg = config()
    ev = UP["PhysicalAIEvaluator"].__new__(UP["PhysicalAIEvaluator"])
    ev.config, ev.device, ev.dtype = cfg, torch.device(device), cfg.param_dtype
    ev.patch_size, ev.cache_name = cfg.patch_size, "pai_rollout"
    ev.q01 = torch.tensor(cfg.norm_stat["q01"][:2], dtype=torch.float32)
    ev.q99 = torch.tensor(cfg.norm_stat["q99"][:2], dtype=torch.float32)
    ev.action_chunk_steps = cfg.action_chunk_steps
    ev.transformer = WanTransformer3DPhysicalAIModel.from_pretrained(
        str(ROOT / "checkpoints/drivewam-physicalai/transformer"), torch_dtype=ev.dtype, attn_mode="flashattn").to(ev.device).eval()
    ev.transformer.requires_grad_(False)
    ev.vae = load_vae(str(ROOT / "checkpoints/lingbot-va-base/vae"), torch_dtype=ev.dtype, torch_device=ev.device).eval()
    ev.vae.requires_grad_(False)
    ev.tokenizer = ev.text_encoder = None
    ev.video_scheduler = UP["FlowMatchScheduler"](shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True)
    ev.action_scheduler = UP["FlowMatchScheduler"](shift=cfg.action_snr_shift, sigma_min=0.0, extra_one_step=True)
    ev._prompt_cache = {}
    ev._prompt_source = prompt_cache
    original = ev._encode_single_prompt
    def cached(prompt, _ev=ev):
        if prompt not in _ev._prompt_cache:
            assert _ev._prompt_source is not None, "prompt cache required (UMT5 not loaded)"
            _ev._prompt_cache[prompt] = _ev._prompt_source.tensor(prompt).to(_ev.device)
        return _ev._prompt_cache[prompt]
    ev._encode_single_prompt = cached
    ev._upstream_encode_single_prompt = original
    return ev

class Tap:
    """Capture selected block outputs for a tagged forward; never alters outputs."""
    def __init__(self, ev, layers=LAYERS):
        self.ev, self.layers, self.tag, self.store = ev, list(layers), None, {}
        self.handles = [ev.transformer.blocks[i].register_forward_hook(
            lambda m, a, out, i=i: self._grab(i, out)) for i in self.layers]
    def _grab(self, i, out):
        if self.tag is not None:
            o = out[0] if isinstance(out, tuple) else out
            self.store.setdefault(self.tag, {})[i] = o.detach().clone()
    def take(self, tag):
        d = self.store.pop(tag)
        return torch.cat([d[i] for i in self.layers], dim=0)          # [L, N, 3072]
    def close(self):
        for h in self.handles:
            h.remove()

def rollout(ev, batch, mode="B", video_steps=3, action_steps=10, skip_terminal_action=False, seed=0,
            video_denoise_fraction=0.6, tap=None, collect=(), head=None, stop_video_after_first=False,
            clock=None, gt_latents=None):
    """Mirror of upstream predict_autoregressive. mode 'B' = baseline action denoising;
    mode 'S' = no action denoising (action noise still drawn). `collect` phases
    ('current','first','terminal') are captured per chunk through `tap`. If `head` is given,
    it maps (features dict, chunk_index, batch) -> normalized action chunk [1,30,1,40,1]."""
    preds, latents, feats, calls = _rollout(ev, batch, mode, video_steps, [("main", action_steps, skip_terminal_action)],
                                            seed, video_denoise_fraction, tap, collect, head, stop_video_after_first,
                                            clock, gt_latents)
    return preds["main"], latents, feats, calls

def rollout_multi(ev, batch, specs, video_steps=3, seed=0, video_denoise_fraction=0.6):
    """Several baselines from one rollout: video passes, noise and GT registration are shared;
    each spec (name, action_steps, skip_terminal) denoises its own copy of the same action noise.
    Opt specs never write the cache (update_cache=0 slots are restored), so outputs are bitwise
    those of separate runs (checked by remote_pai_parity.py). At most one raw spec, placed last."""
    raw = [i for i, s in enumerate(specs) if not s[2]]
    assert len(raw) <= 1 and (not raw or raw[0] == len(specs) - 1), specs
    preds, latents, _, calls = _rollout(ev, batch, "B", video_steps, specs, seed, video_denoise_fraction)
    return preds, latents, calls

@torch.no_grad()
def _rollout(ev, batch, mode, video_steps, specs, seed=0, video_denoise_fraction=0.6, tap=None, collect=(),
             head=None, stop_video_after_first=False, clock=None, gt_latents=None):
    """Shared core. `clock(event, frame_idx)` (optional) is called at chunk start (right after the
    previous GT registration), at decision (action chunk ready) and after the GT registration.
    `gt_latents` may be passed precomputed (latency runs exclude the VAE)."""
    from einops import rearrange
    tick = clock if clock is not None else (lambda event, idx: None)
    ev._prompt_cache.clear()          # upstream keeps every prompt on the GPU (4 MB each); per clip here
    F = torch.nn.functional
    images = batch["images"][:1].to(ev.device)
    actions = batch["actions"][:1].to(ev.device, ev.dtype)
    actions_mask = batch["actions_mask"][:1].to(ev.device)
    ego_vel_acc, ego_curv = batch["ego_velocity_acc"][:1], batch["curvature"][:1]
    prompts_per_chunk = batch["prompts"]
    def chunk_prompt(ci):
        p = prompts_per_chunk[ci] if ci < len(prompts_per_chunk) else prompts_per_chunk[0]
        return ev._encode_prompt([p[0] if isinstance(p, (list, tuple)) else p])
    cond_prompt_emb = chunk_prompt(0)
    if gt_latents is None:
        gt_latents = ev._encode_images(images).to(ev.dtype)
    T_lat = gt_latents.shape[2]
    _, C_lat, _, H_lat, W_lat = gt_latents.shape
    action_dim, steps_per_chunk = actions.shape[1], ev.action_chunk_steps
    latent_timesteps = ev._prepare_video_timesteps(video_steps, video_denoise_fraction)
    schedulers = {}
    for _, n, _ in specs:
        sch = UP["FlowMatchScheduler"](shift=ev.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        sch.set_timesteps(n)
        schedulers[n] = (sch, F.pad(sch.timesteps, (0, 1), mode="constant", value=0))
    ph, pw = ev.patch_size[1], ev.patch_size[2]
    action_cond = torch.zeros(1, action_dim, 1, steps_per_chunk, 1, device=ev.device, dtype=ev.dtype)
    out_chunks, latents_out, feats = {s[0]: [] for s in specs}, [], []
    calls = {"video": 0, "action": 0, "register": 0, "action_by_model": {s[0]: 0 for s in specs}}
    def fwd(inp, update_cache, action_mode, tag=None, kind="video"):
        if tap is not None:
            tap.tag = tag
        try:
            y = ev.transformer(inp, update_cache=update_cache, cache_name=ev.cache_name, action_mode=action_mode)
        finally:
            if tap is not None:
                tap.tag = None
        calls[kind] += 1
        return y
    with torch.random.fork_rng(devices=ev._rng_devices()):
        torch.manual_seed(seed)
        ev.transformer.create_empty_cache(ev.cache_name, ev.config.attn_window, (H_lat // ph) * (W_lat // pw),
                                          steps_per_chunk, ev.device, ev.dtype, batch_size=1)
        try:
            empty = ev._gt_ego_cond(ego_vel_acc, ego_curv, 0)
            zero_t = torch.zeros((1, 1), device=ev.device, dtype=torch.float32)
            fwd(ev._make_video_input(gt_latents[:, :, 0:1], zero_t, cond_prompt_emb, empty, frame_index=0), 2, False,
                tag="current" if "current" in collect else None, kind="register")
            fwd(ev._make_action_input(action_cond, zero_t, cond_prompt_emb, empty, chunk_index=0), 2, True, kind="register")
            for frame_idx in range(1, T_lat):
                tick("start", frame_idx)
                ego = ev._gt_ego_cond(ego_vel_acc, ego_curv, frame_idx)
                prompt_emb = chunk_prompt(frame_idx)
                chunk_feats = {}
                if "current" in collect:
                    chunk_feats["current"] = tap.take("current")
                chunk_latents = torch.randn(1, C_lat, 1, H_lat, W_lat, device=ev.device, dtype=ev.dtype)
                for i, t in enumerate(latent_timesteps):
                    last = i == len(latent_timesteps) - 1
                    tag = "first" if (i == 0 and "first" in collect) else ("terminal" if (last and "terminal" in collect) else None)
                    tt = torch.full((1, 1), float(t.item()), device=ev.device, dtype=torch.float32)
                    pred = fwd(ev._make_video_input(chunk_latents, tt, prompt_emb, ego, frame_index=frame_idx),
                               1 if last else 0, False, tag=tag)
                    if tag is not None:
                        chunk_feats[tag] = tap.take(tag)
                    if stop_video_after_first and i == 0:
                        break
                    if last:
                        continue
                    pred = UP["data_seq_to_patch"](ev.patch_size, pred, 1, H_lat, W_lat, batch_size=1)
                    chunk_latents = ev.video_scheduler.step(pred, t, chunk_latents)
                latents_out.append(chunk_latents)
                mask = actions_mask[:, :, frame_idx:frame_idx + 1].bool()
                noise = torch.randn(1, action_dim, 1, steps_per_chunk, 1, device=ev.device, dtype=ev.dtype)
                noise = torch.where(mask, noise, torch.zeros_like(noise))
                for name, n_steps, skip_terminal in specs:
                    chunk_actions = noise.clone()
                    if mode == "B":
                        action_scheduler, action_timesteps = schedulers[n_steps]
                        for i, t in enumerate(action_timesteps):
                            last = i == len(action_timesteps) - 1
                            if last and skip_terminal:
                                break
                            tt = torch.full((1, 1), float(t.item()), device=ev.device, dtype=torch.float32)
                            pred = fwd(ev._make_action_input(chunk_actions, tt, prompt_emb, ego, chunk_index=frame_idx),
                                       1 if last else 0, True, kind="action")
                            calls["action_by_model"][name] += 1
                            if last:
                                continue
                            pred = rearrange(pred, "b (f n) c -> b c f n 1", f=1)
                            chunk_actions = action_scheduler.step(pred, t, chunk_actions)
                            chunk_actions = torch.where(mask, chunk_actions, torch.zeros_like(chunk_actions))
                    elif head is not None:
                        chunk_actions = head(chunk_feats, frame_idx, batch)
                    out_chunks[name].append(chunk_actions)
                tick("decision", frame_idx)
                feats.append(chunk_feats)
                ev.transformer.clear_pred_cache(ev.cache_name)
                fwd(ev._make_video_input(gt_latents[:, :, frame_idx:frame_idx + 1].to(ev.dtype), zero_t, prompt_emb, ego,
                                         frame_index=frame_idx), 2, False,
                    tag="current" if ("current" in collect and frame_idx + 1 < T_lat) else None, kind="register")
                fwd(ev._make_action_input(actions[:, :, frame_idx:frame_idx + 1], zero_t, prompt_emb, ego,
                                          chunk_index=frame_idx), 2, True, kind="register")
                tick("registered", frame_idx)
        finally:
            ev.transformer.clear_cache(ev.cache_name)
            if tap is not None:
                tap.store.clear()
    preds = {}
    for name, chunks in out_chunks.items():
        dtype = torch.float32 if any(c.dtype == torch.float32 for c in chunks) else action_cond.dtype
        preds[name] = torch.cat([c.to(dtype) for c in [action_cond] + chunks], dim=2)  # FP32 only for heads
    pred_latents = torch.cat([gt_latents[:, :, 0:1].to(ev.dtype)] + latents_out, dim=2)
    return preds, pred_latents, feats, calls

def metrics(ev, pred_actions, gt_actions):
    """Upstream ADE/FDE plus per-chunk final errors (same reset convention)."""
    ade, fde = ev.compute_ade_fde(pred_actions, gt_actions)
    pred, gt = ev.actions_to_trajectory_with_reset(pred_actions, gt_actions)
    l2 = np.sqrt(((pred[0] - gt[0]) ** 2).sum(axis=0))
    k = ev.action_chunk_steps
    chunk_fde = [float(l2[(c + 1) * k - 1]) for c in range(l2.shape[0] // k)]
    chunk_ade = [float(l2[c * k:(c + 1) * k].mean()) for c in range(l2.shape[0] // k)]
    return {"ade": ade, "fde": fde, "chunk_fde": chunk_fde, "chunk_ade": chunk_ade}
