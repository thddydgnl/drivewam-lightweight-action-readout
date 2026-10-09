"""navtest-final, one pass (pre-registration §8). Quality only; latency is separate.

Per scene: hooked original B10 rollout -> B10 trajectory + first-phase [10,20,30]
features; B02-opt (B*) with the same sample seed; frozen student heads (seeds 17,
29) at batch one on the first-phase features. Targets are never opened here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from safetensors.torch import load_file
from baseline_common import ROOT, file_sha, sample_seed, load_policy
from navsim_current_dataset import load_policy_input, build_rollout_batch
from navsim_action_opt import make_skip_terminal_action
from world_feature_tap import WorldFeatureTap, PHASES
from heads_v3 import RegressionHead

FROZEN = {17: "7af22cee1759676c1c078f6f5ce4d0e24e6ffb0fabf62fc447d22edc03b72c90",
          29: "0c5ad62f132d0fcf2c47995c5a02d2badcf9e9db88aed9a646ae0179d8ebd6b4"}
STATS_SHA = "a4c4b471b2ae35276a9780456ba301749f79a1bc4454d87c5a08fb48af1867d8"
V3 = ROOT / "runs/heads_v3_2026-10-03"

def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, indent=2, default=str) + "\n"); tmp.replace(path)

def ego_vector(obs):
    return torch.cat([obs["history_poses"].flatten(), obs["ego_velocity_acc"],
                      torch.nn.functional.one_hot(obs["command_id"], num_classes=4).float()])

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--shard", type=int, required=True); p.add_argument("--run-id", default="navtest_final_2026-10-03")
    a = p.parse_args()
    out = ROOT / "runs" / a.run_id; (out / f"shard{a.shard}").mkdir(parents=True, exist_ok=True)
    status = out / f"inference-shard{a.shard}-status.json"
    try:
        assert torch.cuda.device_count() == 1
        torch.set_num_threads(8)
        rows = {r["scene_token"]: r for r in map(json.loads, (ROOT / "splits/navtest-final-manifest.jsonl").read_text().splitlines())}
        tokens_all = (ROOT / "splits/navtest-final-tokens.txt").read_text().splitlines()
        assert len(tokens_all) == len(set(tokens_all)) == 8896 and set(tokens_all) == set(rows)
        assert all(r["split"] == "navtest_final" and r["input_available"] for r in rows.values())
        tokens = tokens_all[a.shard::2]
        assert subprocess.check_output(["git", "-C", str(ROOT / "work/DriveWAM"), "status", "--porcelain"], text=True) == ""
        assert file_sha(V3 / "stats-full-first.safetensors") == STATS_SHA
        stats = load_file(str(V3 / "stats-full-first.safetensors"))
        heads = {}
        for seed, digest in FROZEN.items():
            path = V3 / f"full-R-first-seed{seed}" / "epoch-040.safetensors"
            assert file_sha(path) == digest, f"frozen checkpoint changed: seed {seed}"
            model = RegressionHead(); model.load_state_dict(load_file(str(path)), strict=True)
            heads[seed] = model.to("cuda:0").eval()
        ppath = out / "protocol.json"
        if a.shard == 0 and not ppath.exists():
            write(ppath, {"run_id": a.run_id, "split": "navtest_final", "preregistration": "reports/사전등록_navtest-dev_2026-10-02.final.md (§8)",
                "models": ["B10", "B02-opt", "student-seed17", "student-seed29"], "frozen_checkpoints": FROZEN, "stats_sha256": STATS_SHA,
                "manifest_sha256": file_sha(ROOT / "splits/navtest-final-manifest.jsonl"), "generation_seed": 0,
                "student_features": "first-phase hook on original B10 rollout, layers [9,19,29]", "student_batch": 1,
                "targets_opened": False, "created_at_utc": datetime.now(timezone.utc).isoformat()})
        policy, loading = load_policy()
        raw, opt, _ = make_skip_terminal_action(policy)
        policy._skip_terminal_action_cache = False
        policy._rollout_action_chunk = lambda *x, **k: (opt if policy._skip_terminal_action_cache else raw)(*x, **k)
        tap = WorldFeatureTap(policy, [9, 19, 29], collect_phases=[PHASES[1]], capture_latents=False)
        to_traj = lambda act: policy.pred_traj_to_poses(policy.actions_to_trajectory(act)).astype(np.float32)
        # Safety: this exact path must reproduce the scored navtest-dev student predictions.
        dev_rows = {r["scene_token"]: r for r in map(json.loads, (ROOT / "splits/navtest-dev-manifest.jsonl").read_text().splitlines())}
        with np.load(V3 / "full-R-first-seed17/navtest-dev-predictions.npz") as d:
            reference = dict(zip(d["scene_tokens"].astype(str).tolist(), d["trajectories"]))
        for t in sorted(reference)[a.shard * 4:(a.shard + 1) * 4]:
            obs = load_policy_input(dev_rows[t]["input_path"])
            full = tap.extract(build_rollout_batch(obs, policy.config), sample_seed(t, 0), skip_actions=False, include_mean=False)
            ego = ((ego_vector(obs) - stats["ego_mean"]) / stats["ego_std"])[None].to("cuda:0")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pose = heads[17](full["dense"][0][None].to("cuda:0"), ego)["poses"].float().cpu()[0].numpy()
            assert np.max(np.abs(pose - reference[t])) <= 1e-5, ("dev reproduction failed", t, float(np.max(np.abs(pose - reference[t]))))
        start, done = time.monotonic(), 0
        for off in range(0, len(tokens), 128):
            chosen = tokens[off:off + 128]
            path = out / f"shard{a.shard}" / f"{off:05d}.npz"
            if path.exists():
                done += len(chosen); continue
            result = {"B10": [], "B02-opt": [], "student-seed17": [], "student-seed29": []}
            for t in chosen:
                assert file_sha(rows[t]["input_path"]) == rows[t]["input_sha256"]
                obs = load_policy_input(rows[t]["input_path"]); seed = sample_seed(t, 0)
                policy._skip_terminal_action_cache = False
                full = tap.extract(build_rollout_batch(obs, policy.config), seed, skip_actions=False, include_mean=False)
                assert full["forward_calls"] == 17
                result["B10"].append(to_traj(full["original_outputs"][0]))
                world = full["dense"][0][None].to("cuda:0")                      # [1,3,112,3072]
                ego = ((ego_vector(obs) - stats["ego_mean"]) / stats["ego_std"])[None].to("cuda:0")
                with torch.inference_mode():
                    for s, m in heads.items():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pose = m(world, ego)["poses"].float().cpu()[0].numpy()
                        assert np.isfinite(pose).all(); result[f"student-seed{s}"].append(pose)
                    policy._skip_terminal_action_cache = True
                    act, _ = policy.predict(build_rollout_batch(obs, policy.config), 3, 2, seed, video_denoise_fraction=.6)
                result["B02-opt"].append(to_traj(act))
                del full
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, scene_tokens=np.asarray(chosen), **{k.replace("-", "_"): np.stack(v) for k, v in result.items()}); tmp.replace(path)
            done += len(chosen)
            write(status, {"stage": "inferring", "completed": done, "total": len(tokens), "elapsed_seconds": time.monotonic() - start})
        tap.close()
        write(status, {"stage": "verified", "completed": done, "total": len(tokens), "loading": loading})
    except Exception as error:
        write(status, {"stage": "failed", "error": repr(error)}); raise
