"""Pass-1 noise sensitivity on navtest-dev (pre-registration 2026-11-01 §3.2).

Per scene and generation seed g (default 1, 2): one hooked original B10 rollout with
`sample_seed(token, g)` gives the B10(g) trajectory and the first-phase [10,20,30] features, on
which the frozen 92.6k student heads (seeds 17, 29) run at batch one, exactly as in
remote_navtest_final_infer.py. The first `--check` scenes of a shard also run g = 0 and must
reproduce the stored navtest-dev B10 and student trajectories within 1e-5 (regenerated inputs and
the online path). Quality only, targets never opened, resumable, two GPUs take token halves.
"""
import argparse
from datetime import datetime, timezone
import json
import time

import numpy as np
import torch
from safetensors.torch import load_file
from baseline_common import ROOT, file_sha, sample_seed, load_policy
from navsim_current_dataset import load_policy_input, build_rollout_batch
from navsim_action_opt import make_skip_terminal_action
from world_feature_tap import WorldFeatureTap, PHASES
from heads_v3 import RegressionHead
from remote_navtest_final_infer import FROZEN, STATS_SHA, V3, ego_vector

def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1,2")
    p.add_argument("--run-id", default="r5_navsim_noise_seed_2026-11-01")
    p.add_argument("--shard", type=int, required=True); p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--check", type=int, default=4)
    p.add_argument("--chunk", type=int, default=128)
    a = p.parse_args()
    gens = [int(g) for g in a.seeds.split(",")]
    out = ROOT / "runs" / a.run_id
    (out / f"shard{a.shard}").mkdir(parents=True, exist_ok=True)
    status_path = out / f"inference-shard{a.shard}-status.json"
    def status(stage, **extra):
        write_json(status_path, {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra})
        print(json.dumps({"stage": stage, **extra}), flush=True)
    try:
        torch.set_num_threads(8)
        assert torch.cuda.device_count() == 1
        manifest = ROOT / "splits/navtest-dev-manifest.regen-2026-11-01.jsonl"
        rows = {r["scene_token"]: r for r in map(json.loads, manifest.read_text().splitlines())}
        tokens_all = (ROOT / "splits/navtest-dev-tokens.txt").read_text().splitlines()
        assert len(tokens_all) == len(set(tokens_all)) == 3250 and set(tokens_all) == set(rows)
        tokens = tokens_all[a.shard::a.num_shards]
        assert file_sha(V3 / "stats-full-first.safetensors") == STATS_SHA
        stats = load_file(str(V3 / "stats-full-first.safetensors"))
        heads = {}
        for seed, digest in FROZEN.items():
            path = V3 / f"full-R-first-seed{seed}" / "epoch-040.safetensors"
            assert file_sha(path) == digest, f"frozen checkpoint changed: seed {seed}"
            model = RegressionHead(); model.load_state_dict(load_file(str(path)), strict=True)
            heads[seed] = model.to("cuda:0").eval()
        reference = {}
        with np.load(ROOT / "runs/navtest_dev_quality_2026-10-02/predictions/B10.npz") as d:
            reference["B10"] = dict(zip(d["scene_tokens"].astype(str).tolist(), d["trajectories"]))
        for seed in FROZEN:
            with np.load(V3 / f"full-R-first-seed{seed}/navtest-dev-predictions.npz") as d:
                reference[f"student-seed{seed}"] = dict(zip(d["scene_tokens"].astype(str).tolist(), d["trajectories"]))
        protocol = {"run_id": a.run_id, "split": "navtest_dev", "preregistration": "reports/사전등록_보완실험_2026-11-01.md",
            "generation_seeds": gens, "sample_seed": "sample_seed(token, g)", "frozen_checkpoints": {str(k): v for k, v in FROZEN.items()},
            "stats_sha256": STATS_SHA, "manifest": str(manifest), "manifest_sha256": file_sha(manifest),
            "student_features": "first-phase hook on original B10 rollout, layers [9,19,29]", "student_batch": 1,
            "check_scenes_per_shard": a.check, "targets_opened": False,
            "scripts": {n: file_sha(ROOT / "scripts" / n) for n in ["remote_r5_navsim_seed.py", "world_feature_tap.py",
                        "heads_v3.py", "remote_navtest_final_infer.py", "baseline_common.py", "navsim_current_dataset.py"]}}
        ppath = out / f"protocol-shard{a.shard}.json"
        if ppath.exists():
            assert json.loads(ppath.read_text()) == protocol, "Protocol mismatch on resume"
        else:
            write_json(ppath, protocol)
        status("loading_model", scenes=len(tokens))
        policy, loading = load_policy()
        raw, opt, _ = make_skip_terminal_action(policy)
        policy._skip_terminal_action_cache = False
        policy._rollout_action_chunk = lambda *x, **k: (opt if policy._skip_terminal_action_cache else raw)(*x, **k)
        tap = WorldFeatureTap(policy, [9, 19, 29], collect_phases=[PHASES[1]], capture_latents=False)
        to_traj = lambda act: policy.pred_traj_to_poses(policy.actions_to_trajectory(act)).astype(np.float32)
        def run(obs, seed):
            full = tap.extract(build_rollout_batch(obs, policy.config), seed, skip_actions=False, include_mean=False)
            assert full["forward_calls"] == 17
            res = {"B10": to_traj(full["original_outputs"][0])}
            world = full["dense"][0][None].to("cuda:0")
            ego = ((ego_vector(obs) - stats["ego_mean"]) / stats["ego_std"])[None].to("cuda:0")
            with torch.inference_mode():
                for s, m in heads.items():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pose = m(world, ego)["poses"].float().cpu()[0].numpy()
                    assert np.isfinite(pose).all(); res[f"student-seed{s}"] = pose
            del full
            return res
        start, done = time.monotonic(), 0
        for offset in range(0, len(tokens), a.chunk):
            chosen = tokens[offset:offset + a.chunk]
            path = out / f"shard{a.shard}" / f"{offset:05d}.npz"
            if path.exists():
                done += len(chosen); continue
            result = {}
            for j, t in enumerate(chosen):
                assert file_sha(rows[t]["input_path"]) == rows[t]["input_sha256"]
                obs = load_policy_input(rows[t]["input_path"])
                if offset + j < a.check:
                    base = run(obs, sample_seed(t, 0))
                    for k, v in base.items():
                        err = float(np.max(np.abs(v - reference[k][t])))
                        assert err <= 1e-5, ("seed-0 reproduction failed", k, t, err)
                    print(json.dumps({"seed0_reproduced": t}), flush=True)
                for g in gens:
                    for k, v in run(obs, sample_seed(t, g)).items():
                        result.setdefault(f"{k}@g{g}", []).append(v)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, scene_tokens=np.asarray(chosen),
                     **{k.replace("-", "_").replace("@", "__"): np.stack(v) for k, v in result.items()})
            tmp.replace(path)
            done += len(chosen)
            status("inferring", completed=done, total=len(tokens), elapsed_seconds=time.monotonic() - start)
        tap.close()
        status("verified", completed=done, total=len(tokens), loading=loading)
    except Exception as error:
        status("failed", error=repr(error))
        raise
