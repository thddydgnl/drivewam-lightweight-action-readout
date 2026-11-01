"""Training-free grid on navtest (pre-registration 2026-11-01 §1): opt variants V{v}-A{k}.

Per scene and video-step count v, one `navsim_grid.predict_multi` call yields every requested
action-step count k with the sample seed of the official protocol (`sample_seed(token, 0)`).
For the first `--parity` scenes of a shard every model is also run with a separate upstream
`predict` call (must be bitwise equal), and V3-A2 (= B02-opt) must reproduce the stored B02-opt
trajectory within 1e-5, which checks the regenerated inputs. Quality only (no timing), targets
never opened. Resumable; two GPUs take disjoint token halves. `--split final` is only for a
re-selected B* (pre-registration §1.4).
"""
import argparse
from datetime import datetime, timezone
import json
import time

import numpy as np
import torch
from baseline_common import ROOT, file_sha, sample_seed, load_policy
from navsim_current_dataset import load_policy_input, build_rollout_batch
from navsim_action_opt import make_skip_terminal_action
from navsim_grid import parse, predict_multi

def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["dev", "final"])
    p.add_argument("--models", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--shard", type=int, required=True); p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--parity", type=int, default=4)
    p.add_argument("--chunk", type=int, default=128)
    a = p.parse_args()
    out = ROOT / "runs" / a.run_id
    (out / f"shard{a.shard}").mkdir(parents=True, exist_ok=True)
    status_path = out / f"inference-shard{a.shard}-status.json"
    def status(stage, **extra):
        write_json(status_path, {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra})
        print(json.dumps({"stage": stage, **extra}), flush=True)
    try:
        torch.set_num_threads(8)
        assert torch.cuda.device_count() == 1
        models = a.models.split(",")
        plan = parse(models)
        name = f"navtest-{a.split}"
        manifest = ROOT / f"splits/{name}-manifest.regen-2026-11-01.jsonl"
        rows = {r["scene_token"]: r for r in map(json.loads, manifest.read_text().splitlines())}
        tokens_all = (ROOT / f"splits/{name}-tokens.txt").read_text().splitlines()
        assert len(tokens_all) == len(set(tokens_all)) == {"dev": 3250, "final": 8896}[a.split] and set(tokens_all) == set(rows)
        tokens = tokens_all[a.shard::a.num_shards]
        ref_path = ROOT / ("runs/navtest_dev_quality_2026-10-02/predictions/B02-opt.npz" if a.split == "dev"
                           else "runs/navtest_final_2026-10-03/predictions/B02-opt.npz")
        with np.load(ref_path) as d:
            reference = dict(zip(d["scene_tokens"].astype(str).tolist(), d["trajectories"]))
        protocol = {"run_id": a.run_id, "split": name, "preregistration": "reports/사전등록_보완실험_2026-11-01.md",
            "models": models, "plan": {str(v): ks for v, ks in plan.items()}, "skip_terminal_action_cache": True,
            "video_denoise_fraction": 0.6, "generation_seed": 0, "manifest": str(manifest), "manifest_sha256": file_sha(manifest),
            "checkpoint_sha256": file_sha(ROOT / "checkpoints/drivewam/navsim/diffusion_pytorch_model.safetensors"),
            "parity_scenes_per_shard": a.parity, "input_check": f"V3-A2 reproduces {ref_path.name} within 1e-5",
            "targets_opened": False,
            "scripts": {n: file_sha(ROOT / "scripts" / n) for n in ["remote_r5_navsim_grid.py", "navsim_grid.py",
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
        policy._rollout_action_chunk = lambda *x, **k: (opt_action if policy._skip_terminal_action_cache else raw_action)(*x, **k)
        def trajectory_of(actions):
            t = policy.pred_traj_to_poses(policy.actions_to_trajectory(actions))
            assert t.shape == (8, 3) and np.isfinite(t).all()
            return t.astype(np.float32)
        start, done = time.monotonic(), 0
        for offset in range(0, len(tokens), a.chunk):
            chosen = tokens[offset:offset + a.chunk]
            path = out / f"shard{a.shard}" / f"{offset:05d}.npz"
            if path.exists():
                done += len(chosen); continue
            traj = {m: [] for m in models}
            for j, token in enumerate(chosen):
                row = rows[token]
                assert file_sha(row["input_path"]) == row["input_sha256"]
                obs = load_policy_input(row["input_path"])
                seed = sample_seed(token, 0)
                got = {}
                for v, ks in plan.items():
                    multi = predict_multi(policy, opt_action, build_rollout_batch(obs, policy.config), v, ks, seed)
                    for k in ks:
                        got[f"V{v}-A{k}"] = multi[k]
                if offset + j < a.parity:
                    policy._skip_terminal_action_cache = True
                    for m, act in got.items():
                        v, k = map(int, m[1:].split("-A"))
                        with torch.inference_mode():
                            alone, _ = policy.predict(build_rollout_batch(obs, policy.config), v, k, seed, video_denoise_fraction=0.6)
                        assert torch.equal(alone, act), ("shared predict differs from a separate predict", token, m)
                    check = predict_multi(policy, opt_action, build_rollout_batch(obs, policy.config), 3, [2], seed)[2]
                    err = float(np.max(np.abs(trajectory_of(check) - reference[token])))
                    assert err <= 1e-5, ("regenerated input does not reproduce B02-opt", token, err)
                    print(json.dumps({"parity_ok": token, "b02_reproduction_max_abs": err}), flush=True)
                for m in models:
                    traj[m].append(trajectory_of(got[m]))
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, scene_tokens=np.asarray(chosen), **{m.replace("-", "_"): np.stack(v) for m, v in traj.items()})
            tmp.replace(path)
            done += len(chosen)
            status("inferring", completed=done, total=len(tokens), elapsed_seconds=time.monotonic() - start)
        status("verified", completed=done, total=len(tokens), loading=loading)
    except Exception as error:
        status("failed", error=repr(error))
        raise
