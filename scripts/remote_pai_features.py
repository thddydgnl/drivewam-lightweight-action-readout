"""PhysicalAI student features: per clip-chunk block outputs [3 layers, 112, 3072] (bf16).

Runs the upstream-mirrored rollout in student mode (no action denoising; action noise still
drawn so RNG matches B10) with GT history registration exactly as the official protocol.
Variants (pre-registration §5, diagnostics at 4k):
  normal     : collect current/first(/terminal) with full GT history and chunk guidance
  nohist     : cache reset before every chunk; only the latest GT frame + zero action registered
  fixedprompt: config default prompt for every chunk (no scene-evolving guidance)
Per row we also store the head inputs (ego vel/acc 4 + curvature 1) and the normalized target
chunk (40 x 3), which are read from the batch only after the rollout. Resumable shards.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import save_file

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "scripts"))
import pai_common as pc                                          # noqa: E402

def stores(name):
    if name == "final":
        return json.loads((ROOT / "work/DriveWAM/src/physicalai/eval_data/prompts_test_sample_1k.json").read_text())
    if name == "dev":
        return json.loads((ROOT / "data/physicalai/dev_guidance/scene_evolving_guidance_store.json").read_text())
    return None                                                  # train: per-clip files in nav_prompts

@torch.no_grad()
def nohist_rollout(ev, batch, tap, collect):
    """Diagnostic: every chunk starts from an empty cache holding only the latest GT frame."""
    F = torch.nn.functional
    ev._prompt_cache.clear()
    images = batch["images"][:1].to(ev.device)
    ego_va, ego_c = batch["ego_velocity_acc"][:1], batch["curvature"][:1]
    prompts = batch["prompts"]
    emb = lambda ci: ev._encode_prompt([prompts[ci][0] if isinstance(prompts[ci], (list, tuple)) else prompts[ci]])
    gt = ev._encode_images(images).to(ev.dtype)
    T_lat = gt.shape[2]; _, C, _, H, W = gt.shape
    steps = ev.action_chunk_steps
    ts = ev._prepare_video_timesteps(3, 0.6)
    zero_t = torch.zeros((1, 1), device=ev.device, dtype=torch.float32)
    feats = []
    with torch.random.fork_rng(devices=ev._rng_devices()):
        torch.manual_seed(0)
        for k in range(1, T_lat):
            ev.transformer.create_empty_cache(ev.cache_name, ev.config.attn_window, (H // 2) * (W // 2), steps,
                                              ev.device, ev.dtype, batch_size=1)
            try:
                ego_prev = ev._gt_ego_cond(ego_va, ego_c, k - 1)
                p_prev = emb(k - 1)
                tap.tag = "current" if "current" in collect else None
                ev.transformer(ev._make_video_input(gt[:, :, k - 1:k], zero_t, p_prev, ego_prev, frame_index=k - 1),
                               update_cache=2, cache_name=ev.cache_name, action_mode=False)
                tap.tag = None
                ev.transformer(ev._make_action_input(torch.zeros(1, 30, 1, steps, 1, device=ev.device, dtype=ev.dtype),
                               zero_t, p_prev, ego_prev, chunk_index=k - 1), update_cache=2, cache_name=ev.cache_name, action_mode=True)
                chunk = {}
                if "current" in collect:
                    chunk["current"] = tap.take("current")
                noise = torch.randn(1, C, 1, H, W, device=ev.device, dtype=ev.dtype)
                tap.tag = "first"
                ev.transformer(ev._make_video_input(noise, torch.full((1, 1), float(ts[0].item()), device=ev.device,
                               dtype=torch.float32), emb(k), ev._gt_ego_cond(ego_va, ego_c, k), frame_index=k),
                               update_cache=0, cache_name=ev.cache_name, action_mode=False)
                tap.tag = None
                chunk["first"] = tap.take("first")
                torch.randn(1, 30, 1, steps, 1, device=ev.device, dtype=ev.dtype)   # keep RNG order
                feats.append(chunk)
            finally:
                tap.tag = None
                ev.transformer.clear_cache(ev.cache_name)
    return feats

def wait_for(clip, poll=30):
    """Block until the clip is processed (True) or its processing failed (False)."""
    while True:
        if (pc.PROCESSED / clip / "receipt.json").exists():
            return True
        if (ROOT / "logs/pai-process-errors" / (clip + ".txt")).exists():
            return False
        time.sleep(poll)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True)                          # train4k | train20k | dev | final
    p.add_argument("--variant", default="normal", choices=["normal", "nohist", "fixedprompt"])
    p.add_argument("--phases", default="first")                     # comma list of current,first,terminal
    p.add_argument("--clips", default=None, help="optional clip list file (defaults to split file)")
    p.add_argument("--tag", default=None, help="output folder suffix (defaults to the variant)")
    p.add_argument("--wait", action="store_true", help="wait for clips still being processed")
    p.add_argument("--shard", type=int, required=True); p.add_argument("--num-shards", type=int, default=2)
    a = p.parse_args()
    assert torch.cuda.device_count() == 1
    assert json.loads((ROOT / "reports/pai-parity.json").read_text())["passed"]
    phases = tuple(a.phases.split(","))
    split_file = a.clips or str(ROOT / f"splits/pai-{a.set.replace('train', 'train-') if a.set.startswith('train') else a.set + '-1k'}.txt")
    clips = Path(split_file).read_text().split()[a.shard::a.num_shards]
    prompt_set = "train20k" if a.set.startswith("train") else a.set
    cache = pc.PromptCache(sorted((ROOT / "caches/pai-prompts" / prompt_set).glob("*.safetensors")))
    ev = pc.load_policy("cuda:0", cache)
    cfg = pc.config()
    store = None if a.variant == "fixedprompt" else stores(a.set)
    pdir = None if (a.variant == "fixedprompt" or not a.set.startswith("train")) else ROOT / "data/physicalai/nav_prompts"
    data = pc.ClipData(cfg, clips, prompts_store=store, prompts_dir=pdir)
    tap = pc.Tap(ev)
    out = ROOT / "caches/pai-features" / f"{a.set}-{a.tag or a.variant}" / f"shard{a.shard}"
    out.mkdir(parents=True, exist_ok=True)
    status = out.parent / f"status-shard{a.shard}.json"
    start = time.monotonic()
    buf = {ph: [] for ph in phases}; meta = []; ego_rows = []; targets = []
    number = len(list(out.glob("*.safetensors")))
    done = set()
    for m in out.glob("*.safetensors.json"):
        done.update(json.loads(m.read_text())["clips"])
    skipped = []
    pending_clips = []
    def flush():
        global number, buf, meta, ego_rows, targets, pending_clips
        if not pending_clips:
            return
        tensors = {f"dense_{ph}": torch.stack(buf[ph]) for ph in phases}
        tensors["ego"] = torch.stack(ego_rows); tensors["target"] = torch.stack(targets)
        path = out / f"{number:05d}.safetensors"; tmp = path.with_suffix(".tmp")
        save_file(tensors, str(tmp)); tmp.replace(path)
        Path(str(path) + ".json").write_text(json.dumps({"clips": pending_clips, "rows": meta, "phases": list(phases),
            "variant": a.variant, "set": a.set}) + "\n")
        number += 1
        buf = {ph: [] for ph in phases}; meta, ego_rows, targets, pending_clips = [], [], [], []
    for i, clip in enumerate(clips):
        if clip in done:
            continue
        if not (wait_for(clip) if a.wait else (pc.PROCESSED / clip / "receipt.json").exists()):
            skipped.append(clip)
            continue
        item = data[i]; batch = pc.collate(item)
        if a.variant == "nohist":
            feats = nohist_rollout(ev, batch, tap, phases)
        else:
            _, _, feats, calls = pc.rollout(ev, batch, "S", 3, 10, tap=tap, collect=phases,
                                            stop_video_after_first=("terminal" not in phases))
        K = len(feats)
        for k in range(K):
            for ph in phases:
                f = feats[k][ph]
                assert f.shape == (3, 112, 3072), (clip, k, ph, tuple(f.shape))
                buf[ph].append(f.cpu())
            ego_rows.append(torch.cat([item["ego_velocity_acc"][k], item["curvature"][k:k + 1]]).float())
            targets.append(item["actions"][:3, k + 1, :, 0].T.contiguous().float())          # [40, 3] normalized
            meta.append({"clip": clip, "chunk": k + 1})
        pending_clips.append(clip)
        if len(pending_clips) == 64:
            flush()
            status.write_text(json.dumps({"done": i + 1, "total": len(clips), "skipped": len(skipped),
                "elapsed_s": time.monotonic() - start, "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    flush()
    tap.close()
    status.write_text(json.dumps({"stage": "verified", "total": len(clips), "skipped": skipped,
        "elapsed_s": time.monotonic() - start}) + "\n")
