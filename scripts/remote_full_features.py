"""Round-4 first-phase features for streamed head-train scenes (pre-registration §7).

  --shard i     poll processed fragments; extract tokens with sha256(token)%2==i in 128-row
                shards, early-stopping after the first future forward (3 backbone calls)
  --finalize    verify coverage of full-extra-tokens.txt and write FEATURE_LOCK.json
First shard re-extracts 8 X1 tokens and compares bitwise with the X1 cache (phase 1).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from baseline_common import ROOT, file_sha, sample_seed, load_policy
from navsim_current_dataset import load_policy_input, build_rollout_batch
from world_feature_tap import WorldFeatureTap, PHASES

OUT = ROOT / "caches/features/full_first_2026-10-03"
LAYERS = [9, 19, 29]
FIRST = PHASES[1]

def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, indent=2) + "\n"); tmp.replace(path)

def ego_vector(obs):
    return torch.cat([obs["history_poses"].flatten(), obs["ego_velocity_acc"],
                      torch.nn.functional.one_hot(obs["command_id"], num_classes=4).float()])

def mine(token, shard):
    return int(hashlib.sha256(token.encode()).hexdigest(), 16) % 2 == shard

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--shard", type=int); p.add_argument("--finalize", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.finalize:
        tokens = (ROOT / "splits/full-extra-tokens.txt").read_text().splitlines()
        shards = [json.loads(r.read_text()) for r in sorted(OUT.glob("shard*/*.json"))]
        for r in shards:
            assert file_sha(r["path"]) == r["sha256"]
        covered = [t for r in shards for t in r["tokens"]]
        assert len(covered) == len(set(covered)) and set(covered) == set(tokens), (len(covered), len(tokens))
        write(OUT / "FEATURE_LOCK.json", {"status": "full_first_features_verified", "scenes": len(tokens), "phase": FIRST,
              "layers_zero_based": LAYERS, "generation_seed": 0, "shards": shards, "bytes": sum(r["bytes"] for r in shards),
              "created_at_utc": datetime.now(timezone.utc).isoformat()})
        print("locked", len(shards), len(tokens)); raise SystemExit
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    folder = OUT / f"shard{a.shard}"; folder.mkdir(exist_ok=True)
    status_path = OUT / f"shard{a.shard}-status.json"
    policy, loading = load_policy()
    tap = WorldFeatureTap(policy, LAYERS, collect_phases=[FIRST], capture_latents=False, record_metadata=False)
    def extract(obs, token):
        r = tap.extract(build_rollout_batch(obs, policy.config), sample_seed(token, 0), stop_at=FIRST, include_mean=False)
        assert r["forward_calls"] == 3 and r["dense"].shape == (1, 3, 112, 3072)
        return r["dense"][0].cpu()
    # Parity with the X1 cache (full rollout, phase index 1) on 8 tokens.
    x1 = json.loads((ROOT / "caches/features/x1_features_2026-10-03/FEATURE_LOCK.json").read_text())["shards"][a.shard]
    x1_rows = {r["scene_token"]: r for r in map(json.loads, (ROOT / "splits/x1-manifest.jsonl").read_text().splitlines())}
    with safe_open(x1["path"], framework="pt", device="cpu") as f:
        reference = f.get_slice("dense")[:8, 1:2][:, 0]
    for i, t in enumerate(x1["tokens"][:8]):
        assert torch.equal(extract(load_policy_input(x1_rows[t]["input_path"]), t), reference[i]), ("early-stop parity", t)
    done = {t for r in folder.glob("*.json") for t in json.loads(r.read_text())["tokens"]}
    start, number = time.monotonic(), len(list(folder.glob("*.json")))
    while True:
        rows = {}
        for frag in sorted((ROOT / "reports/full-processed").glob("*.jsonl")):
            for line in frag.read_text().splitlines():
                r = json.loads(line)
                if mine(r["scene_token"], a.shard) and r["scene_token"] not in done:
                    rows[r["scene_token"]] = r
        stream = json.loads((ROOT / "reports/full-stream-status.json").read_text()).get("stage")
        pending = sorted(rows)
        if not pending and stream in ("streamed", "verified", "excluded_over_limit"):
            break
        if len(pending) < 128 and stream not in ("streamed", "verified", "excluded_over_limit"):
            time.sleep(60); continue
        if stream == "failed":
            raise RuntimeError("stream failed")
        chosen = pending[:128]
        dense, ego = [], []
        for t in chosen:
            assert file_sha(rows[t]["input_path"]) == rows[t]["input_sha256"]
            obs = load_policy_input(rows[t]["input_path"])
            dense.append(extract(obs, t)); ego.append(ego_vector(obs))
        path = folder / f"{number:05d}.safetensors"; number += 1
        tmp = path.with_suffix(".tmp"); save_file({"dense": torch.stack(dense), "ego": torch.stack(ego)}, str(tmp)); tmp.replace(path)
        write(path.with_suffix(".json"), {"path": str(path), "tokens": chosen, "sha256": file_sha(path), "bytes": path.stat().st_size})
        done.update(chosen)
        write(status_path, {"stage": "extracting", "completed": len(done), "elapsed_seconds": time.monotonic() - start, "stream": stream})
    tap.close()
    write(status_path, {"stage": "verified", "completed": len(done), "loading": loading})
