"""Round-3 X1 dense features: world-only rollout, layers [10,20,30], 3 phases.

  --shard i      extract tokens[i::2] on the visible GPU (resumable receipts)
  --finalize     verify both halves and write FEATURE_LOCK.json
The first 8 X1 tokens are re-extracted in a fresh call and compared bitwise.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch
from safetensors.torch import save_file
from baseline_common import ROOT, file_sha, sample_seed, load_policy
from navsim_current_dataset import load_policy_input, build_rollout_batch
from world_feature_tap import WorldFeatureTap

OUT = ROOT / "caches/features/x1_features_2026-10-03"
LAYERS = [9, 19, 29]

def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, indent=2) + "\n"); tmp.replace(path)

def ego_vector(obs):
    return torch.cat([obs["history_poses"].flatten(), obs["ego_velocity_acc"],
                      torch.nn.functional.one_hot(obs["command_id"], num_classes=4).float()])

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--shard", type=int); p.add_argument("--finalize", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tokens = (ROOT / "splits/x1-head-train-tokens.txt").read_text().splitlines()
    assert len(tokens) == len(set(tokens)) == 16000
    if a.finalize:
        shards = []
        for receipt in sorted(OUT.glob("shard*/*.json")):
            r = json.loads(receipt.read_text())
            assert file_sha(r["path"]) == r["sha256"]
            shards.append(r)
        covered = [t for r in shards for t in r["tokens"]]
        assert sorted(covered) == sorted(tokens) and len(covered) == len(set(covered))
        write(OUT / "FEATURE_LOCK.json", {"status": "x1_dense_features_verified", "scenes": 16000, "layers_zero_based": LAYERS,
            "phases": ["current_primer", "future_first_preupdate", "future_cache_terminal"], "generation_seed": 0,
            "shards": shards, "bytes": sum(r["bytes"] for r in shards), "created_at_utc": datetime.now(timezone.utc).isoformat()})
        print("locked", len(shards)); raise SystemExit
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    rows = {r["scene_token"]: r for r in map(json.loads, (ROOT / "splits/x1-manifest.jsonl").read_text().splitlines())}
    mine = tokens[a.shard::2]
    folder = OUT / f"shard{a.shard}"; folder.mkdir(exist_ok=True)
    status = OUT / f"shard{a.shard}-status.json"
    policy, loading = load_policy()
    tap = WorldFeatureTap(policy, LAYERS)
    def extract(token):
        obs = load_policy_input(rows[token]["input_path"])
        r = tap.extract(build_rollout_batch(obs, policy.config), sample_seed(token, 0))
        assert r["forward_calls"] == 6 and r["dense"].shape == (3, 3, 112, 3072)
        return r["dense"].cpu(), r["mean"].cpu(), ego_vector(obs)
    start, done = time.monotonic(), 0
    for off in range(0, len(mine), 128):
        chosen = mine[off:off + 128]
        path = folder / f"{off:05d}.safetensors"; receipt = path.with_suffix(".json")
        if receipt.exists():
            done += len(chosen); continue
        dense, mean, ego = [], [], []
        for t in chosen:
            assert file_sha(rows[t]["input_path"]) == rows[t]["input_sha256"]
            d, m, e = extract(t); dense.append(d); mean.append(m); ego.append(e)
        if off == 0:                                                    # determinism check
            for i, t in enumerate(chosen[:8]):
                assert torch.equal(extract(t)[0], dense[i]), t
        tensors = {"dense": torch.stack(dense), "mean": torch.stack(mean), "ego": torch.stack(ego)}
        tmp = path.with_suffix(".tmp"); save_file(tensors, str(tmp)); tmp.replace(path)
        write(receipt, {"path": str(path), "tokens": chosen, "sha256": file_sha(path), "bytes": path.stat().st_size})
        done += len(chosen)
        write(status, {"stage": "extracting", "completed": done, "total": len(mine), "elapsed_seconds": time.monotonic() - start})
    tap.close()
    write(status, {"stage": "verified", "completed": done, "total": len(mine), "loading": loading})
