"""PhysicalAI baseline inference (quality only) on a fixed clip set; resumable, sharded by GPU.

Opt models (B10/B05/B03/B02/B01-opt) share one rollout (`rollout_multi`, bitwise equal to
separate runs per the parity gate). For the first `--raw-check` clips of each shard, raw B10 is
added last and must equal B10-opt bitwise (pre-registration §4). Saves normalized action chunks
(channels 0-2, chunks 1..K) and upstream ADE/FDE + per-chunk errors per clip.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "scripts"))
import pai_common as pc                                          # noqa: E402

MODELS = {"B10": (10, False), "B05": (5, False), "B10-opt": (10, True), "B05-opt": (5, True),
          "B03-opt": (3, True), "B02-opt": (2, True), "B01-opt": (1, True)}

def prompt_store(name):
    if name == "final":
        return json.loads((ROOT / "work/DriveWAM/src/physicalai/eval_data/prompts_test_sample_1k.json").read_text())
    if name == "dev":
        return json.loads((ROOT / "data/physicalai/dev_guidance/scene_evolving_guidance_store.json").read_text())
    raise ValueError(name)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True, choices=["final", "dev"])
    p.add_argument("--models", required=True)
    p.add_argument("--shard", type=int, required=True); p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--run-id", required=True)
    p.add_argument("--raw-check", type=int, default=16)
    p.add_argument("--seed", type=int, default=0, help="per-clip seed (official protocol: 0)")
    a = p.parse_args()
    assert torch.cuda.device_count() == 1
    assert json.loads((ROOT / "reports/pai-parity.json").read_text())["passed"], "parity gates must pass first"
    models = a.models.split(",")
    assert all(MODELS[m][1] for m in models), "only opt models here; raw B10 is the --raw-check reference"
    clips = (ROOT / f"splits/pai-{a.set}-1k.txt").read_text().split()[a.shard::a.num_shards]
    out = ROOT / "runs" / a.run_id / f"shard{a.shard}"; out.mkdir(parents=True, exist_ok=True)
    cache = pc.PromptCache(sorted((ROOT / "caches/pai-prompts" / a.set).glob("*.safetensors")))
    ev = pc.load_policy("cuda:0", cache)
    data = pc.ClipData(pc.config(), clips, prompts_store=prompt_store(a.set))
    status = out.parent / f"status-shard{a.shard}.json"
    start = time.monotonic()
    for i, clip in enumerate(clips):
        path = out / f"{clip}.json"
        if path.exists():
            continue
        batch = pc.collate(data[i])
        record = {"clip": clip, "seed": a.seed, "chunk_count": int(batch["actions"].shape[2] - 1), "models": {}}
        specs = [(m, *MODELS[m]) for m in models]
        raw_check = i < a.raw_check and "B10-opt" in models
        if raw_check:
            specs.append(("B10", 10, False))
        preds, _, calls = pc.rollout_multi(ev, batch, specs, seed=a.seed)
        if raw_check:
            same = torch.equal(preds["B10"], preds["B10-opt"])
            record["raw_check"] = {"B10_equals_B10-opt": bool(same)}
            assert same, clip
        gt = batch["actions"][:1].to(ev.device)
        for m in models:
            pred = preds[m]
            met = pc.metrics(ev, pred, gt)
            record["models"][m] = {**met, "action_calls": calls["action_by_model"][m],
                "pred_norm": pred[0, :3, 1:, :, 0].float().cpu().numpy().round(6).tolist()}
        record["shared_calls"] = {k: calls[k] for k in ["video", "register"]}
        tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(record) + "\n"); tmp.replace(path)
        if (i + 1) % 20 == 0 or i + 1 == len(clips):
            status.write_text(json.dumps({"done": i + 1, "total": len(clips), "elapsed_s": time.monotonic() - start,
                                          "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    status.write_text(json.dumps({"stage": "verified", "done": len(clips), "total": len(clips),
                                  "elapsed_s": time.monotonic() - start}) + "\n")
