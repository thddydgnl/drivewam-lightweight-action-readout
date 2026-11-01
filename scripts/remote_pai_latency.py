"""Interleaved decision-latency measurement on the fixed dev-256 clips (pre-registration §4).

Per clip and repetition, every model runs the full official rollout (GT registration included)
in a rotated order; the clock brackets each chunk's decision (start = right after the previous GT
registration, end = action chunk ready) and the following GT registration. VAE latents are
computed once per clip and excluded. Baselines are opt models (the terminal action pass of raw
B10 comes after the decision). The student = first video pass + head; its online predictions are
saved to be compared bitwise with the offline (cached-feature) predictions.
Grid models `V{v}-A{k}` (pre-registration 2026-11-01 §1.3) run v video denoising steps + k opt action steps.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import torch
from safetensors.torch import load_file

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "scripts"))
import pai_common as pc                                          # noqa: E402
from pai_heads import PaiHead, heading_before, to_clip_frame     # noqa: E402

BASE = {"B10-opt": 10, "B05-opt": 5, "B03-opt": 3, "B02-opt": 2, "B01-opt": 1}

def steps(model):
    """(video steps, action steps) of an opt baseline or grid model."""
    m = re.fullmatch(r"V(\d)-A(\d+)", model)
    return (int(m.group(1)), int(m.group(2))) if m else (3, BASE[model])

class Student:
    def __init__(self, folder, device):
        self.model = PaiHead().to(device).eval()
        self.model.load_state_dict(load_file(str(Path(folder) / "epoch-040.safetensors")))
        st = load_file(str(Path(folder) / "stats.safetensors"))
        self.mu, self.sd, self.device = st["ego_mean"], st["ego_std"], device
        self.frame = json.loads((Path(folder) / "summary.json").read_text()).get("frame", "ego")
    @torch.no_grad()
    def __call__(self, feats, frame_idx, batch):
        w = feats["first"][None]
        e = torch.cat([batch["ego_velocity_acc"][0, frame_idx - 1], batch["curvature"][0, frame_idx - 1:frame_idx]]).float()
        e = ((e - self.mu) / self.sd)[None].to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(w, e)                                   # [1, 40, 3] FP32
        if self.frame == "ego":                                      # heading from registered GT actions
            prev = [batch["actions"][0, :3, j, :, 0].T.numpy() for j in range(1, frame_idx)]
            th = torch.tensor([heading_before(prev)], dtype=torch.float64).float().to(self.device)
            out = to_clip_frame(out, th)
        act = torch.zeros(1, 30, 1, 40, 1, device=self.device, dtype=torch.float32)
        act[0, :3, 0, :, 0] = out[0].T
        return act

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="B10-opt,B05-opt,B03-opt,B02-opt,B01-opt,S-first")
    p.add_argument("--head", required=True, help="trained head folder used for the student path")
    p.add_argument("--clips", type=int, default=256); p.add_argument("--reps", type=int, default=3)
    p.add_argument("--shard", type=int, required=True); p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--run-id", required=True)
    a = p.parse_args()
    assert torch.cuda.device_count() == 1
    models = a.models.split(",")
    clips = (ROOT / "splits/pai-dev-1k.txt").read_text().split()[:a.clips][a.shard::a.num_shards]
    out = ROOT / "runs" / a.run_id; out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"shard{a.shard}.jsonl"
    store = json.loads((ROOT / "data/physicalai/dev_guidance/scene_evolving_guidance_store.json").read_text())
    ev = pc.load_policy("cuda:0", pc.PromptCache(sorted((ROOT / "caches/pai-prompts/dev").glob("*.safetensors"))))
    data = pc.ClipData(pc.config(), clips, prompts_store=store)
    student = Student(a.head, ev.device)
    tap = pc.Tap(ev)
    stamps = []
    def clock(event, idx):
        torch.cuda.synchronize()
        stamps.append((event, idx, time.perf_counter()))
    def run(model, batch, latents):
        stamps.clear()
        if model == "S-first":
            pred, _, _, _ = pc.rollout(ev, batch, "S", 3, 10, tap=tap, collect=("first",), head=student,
                                       stop_video_after_first=True, clock=clock, gt_latents=latents)
        else:
            pred, _, _, _ = pc.rollout(ev, batch, "B", *steps(model), skip_terminal_action=True, clock=clock,
                                       gt_latents=latents)
        t = {(e, i): s for e, i, s in stamps}
        chunks = sorted({i for _, i, _ in stamps})
        return pred, [{"chunk": i, "decision_ms": 1e3 * (t[("decision", i)] - t[("start", i)]),
                       "register_ms": 1e3 * (t[("registered", i)] - t[("decision", i)])} for i in chunks]
    # warm-up (not recorded)
    for i in range(2):
        batch = pc.collate(data[i]); lat = ev._encode_images(batch["images"][:1].to(ev.device)).to(ev.dtype)
        for m in models:
            run(m, batch, lat)
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line); done.add((r["clip"], r["rep"]))
    online = {}
    start = time.monotonic()
    with rows_path.open("a") as sink:
        for i, clip in enumerate(clips):
            batch = pc.collate(data[i])
            lat = ev._encode_images(batch["images"][:1].to(ev.device)).to(ev.dtype)
            for rep in range(a.reps):
                if (clip, rep) in done:
                    continue
                k = (i * a.reps + rep) % len(models)
                order = models[k:] + models[:k]                  # rotate the model order
                recs = []
                for m in order:
                    pred, chunks = run(m, batch, lat)
                    if m == "S-first" and rep == 0:
                        online[clip] = pred[0, :3, 1:, :, 0].float().cpu()
                    recs += [{"clip": clip, "rep": rep, "model": m, "position": order.index(m), **c} for c in chunks]
                sink.write("".join(json.dumps(r) + "\n" for r in recs)); sink.flush()
            if (i + 1) % 16 == 0:
                print(json.dumps({"done": i + 1, "total": len(clips), "elapsed_s": time.monotonic() - start}), flush=True)
    tap.close()
    if online:
        torch.save(online, out / f"student-online-shard{a.shard}.pt")
    (out / f"status-shard{a.shard}.json").write_text(json.dumps({"stage": "verified", "clips": len(clips), "reps": a.reps,
        "models": models, "head": a.head, "gpu": torch.cuda.get_device_name(0),
        "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
