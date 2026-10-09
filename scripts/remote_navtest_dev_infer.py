"""Quality-only inference on the pre-registered navtest-dev split (no timing).

Per scene, one hooked original B10 rollout yields the B10 trajectory and the
current/first/terminal dense features (hook parity was verified bitwise in the
pilot). B05/B03/B02/B01-opt then run with the same sample seed. Targets are
never opened. Shards are resumable; two GPUs process disjoint token halves.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from safetensors.torch import save_file
from baseline_common import ROOT, file_sha, sample_seed, load_policy
from navsim_current_dataset import load_policy_input, build_rollout_batch
from navsim_action_opt import make_skip_terminal_action
from world_feature_tap import WorldFeatureTap, PHASES, LAYERS

OPT = {"B05-opt": 5, "B03-opt": 3, "B02-opt": 2, "B01-opt": 1}
MODELS = ["B10"] + list(OPT)
SEED_TOKENS = 32

def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)

def ego_vector(obs):
    return torch.cat([obs["history_poses"].flatten(), obs["ego_velocity_acc"],
                      torch.nn.functional.one_hot(obs["command_id"], num_classes=4).float()])

def rank(token):
    return hashlib.sha256(f"17:seed-sensitivity:{token}".encode()).hexdigest()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="navtest_dev_quality_2026-10-02")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=128)
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    (out / f"shard{args.shard}").mkdir(parents=True, exist_ok=True)
    status_path = out / f"inference-shard{args.shard}-status.json"
    def status(stage, **extra):
        write_json(status_path, {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra})
        print(json.dumps({"stage": stage, **extra}), flush=True)
    try:
        torch.set_num_threads(8)
        assert torch.cuda.device_count() == 1
        manifest_path = ROOT / "splits/navtest-dev-manifest.jsonl"
        rows = {r["scene_token"]: r for r in map(json.loads, manifest_path.read_text().splitlines())}
        tokens_all = (ROOT / "splits/navtest-dev-tokens.txt").read_text().splitlines()
        assert len(tokens_all) == len(set(tokens_all)) == 3250 and set(tokens_all) == set(rows)
        assert all(r["split"] == "navtest_dev" and r["input_available"] for r in rows.values())
        tokens = tokens_all[args.shard::args.num_shards]
        seed_tokens = set(sorted(tokens_all, key=rank)[:SEED_TOKENS])
        source = ROOT / "work/DriveWAM"
        assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == "abbefb407c9c553f221522fc26077d307620adb4"
        assert subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True) == ""
        weight_sha = file_sha(ROOT / "checkpoints/drivewam/navsim/diffusion_pytorch_model.safetensors")
        assert weight_sha == "27e4846b0a25e22fc299f7aca90c01d802072d2b297741c85ed074b689ef26a9"
        protocol = {"run_id": args.run_id, "split": "navtest_dev", "preregistration": "reports/사전등록_navtest-dev_2026-10-02.md",
            "manifest_sha256": file_sha(manifest_path), "tokens_sha256": file_sha(ROOT / "splits/navtest-dev-tokens.txt"),
            "checkpoint_sha256": weight_sha, "source_commit": "abbefb407c9c553f221522fc26077d307620adb4",
            "models": {"B10": 10, **OPT}, "skip_terminal_action_cache": {m: m.endswith("-opt") for m in MODELS},
            "video_steps": 3, "video_denoise_fraction": 0.6, "generation_seed": 0,
            "sample_seed": "big-endian first 4 SHA256 bytes of UTF-8 scene_token:generation_seed",
            "seed_sensitivity": {"model": "B10", "generation_seeds": [1, 2], "tokens": sorted(seed_tokens)},
            "features": {"phases": PHASES, "layer_indices_zero_based": LAYERS, "dense_dtype": "bfloat16",
                         "source": "hooked original B10 rollout (bitwise parity with world-only extraction verified in pilot)"},
            "timing": "not measured; quality-only", "targets_opened": False, "navtest_final_used": False,
            "scripts": {n: file_sha(ROOT / "scripts" / n) for n in ["remote_navtest_dev_infer.py", "world_feature_tap.py",
                        "navsim_action_opt.py", "baseline_common.py", "navsim_current_dataset.py"]}}
        ppath = out / "protocol.json"
        if ppath.exists():
            assert json.loads(ppath.read_text()) == protocol, "Protocol mismatch on resume"
        else:
            write_json(ppath, protocol)
        status("loading_model", scenes=len(tokens))
        policy, loading = load_policy()
        raw_action, opt_action, _ = make_skip_terminal_action(policy)
        policy._skip_terminal_action_cache = False
        policy._rollout_action_chunk = lambda *a, **k: (opt_action if policy._skip_terminal_action_cache else raw_action)(*a, **k)
        tap = WorldFeatureTap(policy)
        def trajectory_of(actions):
            t = policy.pred_traj_to_poses(policy.actions_to_trajectory(actions))
            assert t.shape == (8, 3) and np.isfinite(t).all()
            return t.astype(np.float32)
        start = time.monotonic()
        done = 0
        for offset in range(0, len(tokens), args.chunk):
            chosen = tokens[offset:offset + args.chunk]
            stem = out / f"shard{args.shard}" / f"{offset:05d}"
            receipt = stem.with_suffix(".json")
            if receipt.exists():
                report = json.loads(receipt.read_text())
                assert report["tokens"] == chosen and file_sha(report["features_path"]) == report["features_sha256"]
                done += len(chosen)
                continue
            dense, mean, ego = [], [], []
            traj = {m: [] for m in MODELS}
            seeds = {}
            for token in chosen:
                row = rows[token]
                assert file_sha(row["input_path"]) == row["input_sha256"]
                obs = load_policy_input(row["input_path"])
                batch = build_rollout_batch(obs, policy.config)
                seed = sample_seed(token, 0)
                policy._skip_terminal_action_cache = False
                full = tap.extract(batch, seed, skip_actions=False)
                assert full["forward_calls"] == 17
                traj["B10"].append(trajectory_of(full["original_outputs"][0]))
                dense.append(full["dense"].cpu()); mean.append(full["mean"].cpu()); ego.append(ego_vector(obs))
                del full
                for name, steps in OPT.items():
                    policy._skip_terminal_action_cache = True
                    with torch.inference_mode():
                        actions, _ = policy.predict(build_rollout_batch(obs, policy.config), 3, steps, seed, video_denoise_fraction=0.6)
                    traj[name].append(trajectory_of(actions))
                if token in seed_tokens:
                    policy._skip_terminal_action_cache = False
                    for g in [1, 2]:
                        with torch.inference_mode():
                            actions, _ = policy.predict(build_rollout_batch(obs, policy.config), 3, 10, sample_seed(token, g), video_denoise_fraction=0.6)
                        seeds[f"{token}:{g}"] = trajectory_of(actions)
            tensors = {"dense": torch.stack(dense), "mean": torch.stack(mean), "ego": torch.stack(ego)}
            for value in tensors.values():
                assert torch.isfinite(value).all()
            fpath = stem.with_suffix(".safetensors")
            tmp = fpath.with_suffix(".tmp")
            save_file(tensors, str(tmp)); tmp.replace(fpath)
            tpath = stem.with_suffix(".npz")
            np.savez(tpath, scene_tokens=np.asarray(chosen), **{m.replace("-", "_"): np.stack(v) for m, v in traj.items()},
                     seed_keys=np.asarray(list(seeds)), seed_trajectories=np.stack(list(seeds.values())) if seeds else np.zeros((0, 8, 3), np.float32))
            write_json(receipt, {"tokens": chosen, "features_path": str(fpath), "features_sha256": file_sha(fpath),
                                 "trajectories_path": str(tpath), "trajectories_sha256": file_sha(tpath)})
            done += len(chosen)
            status("inferring", completed=done, total=len(tokens), elapsed_seconds=time.monotonic() - start)
        tap.close()
        status("verified", completed=done, total=len(tokens), loading=loading)
    except Exception as error:
        status("failed", error=repr(error))
        raise
