"""navtest-final latency (pre-registration §8): B10, B02-opt and the frozen student
(native early exit after the first future forward + SF3 head), interleaved per scene,
3 repeats on 256 fixed final scenes. Online outputs are checked against the quality run.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file
from baseline_common import ROOT, sample_seed, load_policy
from navsim_current_dataset import load_policy_input, build_rollout_batch
from navsim_action_opt import make_skip_terminal_action
from world_feature_tap import WorldFeatureTap, PHASES
from heads_v3 import RegressionHead
from remote_navtest_final_infer import ego_vector

OUT = ROOT / "runs/navtest_final_latency_2026-10-03"
V3 = ROOT / "runs/heads_v3_2026-10-03"

if __name__ == "__main__":
    assert not OUT.exists(); OUT.mkdir(parents=True)
    torch.set_num_threads(8)
    rows = {r["scene_token"]: r for r in map(json.loads, (ROOT / "splits/navtest-final-manifest.jsonl").read_text().splitlines())}
    tokens = sorted(rows, key=lambda t: hashlib.sha256(f"17:final-latency:{t}".encode()).hexdigest())[:256]
    reference = {}
    for f in sorted((ROOT / "runs/navtest_final_2026-10-03").glob("shard*/*.npz")):
        with np.load(f) as d:
            for i, t in enumerate(d["scene_tokens"].astype(str)):
                reference[t] = {"B10": d["B10"][i], "B02-opt": d["B02_opt"][i], "student": d["student_seed17"][i]}
    inputs = {t: load_policy_input(rows[t]["input_path"]) for t in tokens}
    policy, _ = load_policy()
    raw, opt, _ = make_skip_terminal_action(policy)
    policy._skip_terminal_action_cache = False
    policy._rollout_action_chunk = lambda *x, **k: (opt if policy._skip_terminal_action_cache else raw)(*x, **k)
    tap = WorldFeatureTap(policy, [9, 19, 29], collect_phases=[PHASES[1]], capture_latents=False, record_metadata=False)
    stats = load_file(str(V3 / "stats-full-first.safetensors"))
    head = RegressionHead(); head.load_state_dict(load_file(str(V3 / "full-R-first-seed17/epoch-040.safetensors"))); head = head.to("cuda:0").eval()
    def run(name, t):
        torch.cuda.synchronize(); load = float(Path("/proc/loadavg").read_text().split()[0]); s = time.perf_counter()
        if name == "student":
            r = tap.extract(build_rollout_batch(inputs[t], policy.config), sample_seed(t, 0), stop_at=PHASES[1], include_mean=False)
            ego = ((ego_vector(inputs[t]) - stats["ego_mean"]) / stats["ego_std"])[None].to("cuda:0")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                traj = head(r["dense"][0][None].to("cuda:0"), ego)["poses"].float().cpu()[0].numpy()
            assert r["forward_calls"] == 3
        else:
            policy._skip_terminal_action_cache = name == "B02-opt"
            with torch.inference_mode():
                act, _ = policy.predict(build_rollout_batch(inputs[t], policy.config), 3, 10 if name == "B10" else 2, sample_seed(t, 0), video_denoise_fraction=.6)
                traj = policy.pred_traj_to_poses(policy.actions_to_trajectory(act))
        torch.cuda.synchronize(); ms = (time.perf_counter() - s) * 1000
        assert np.max(np.abs(np.asarray(traj, np.float32) - reference[t][name])) <= 1e-5, (name, t)
        return {"model": name, "scene_token": t, "policy_ms": ms, "host_load1": load, "sm_clock_mhz": 0}
    models = ["B10", "B02-opt", "student"]
    for t in tokens[:16]:
        for m in models: run(m, t)
    with (OUT / "timings.jsonl").open("w") as stream:
        for rep in range(3):
            for i, t in enumerate(tokens):
                k = (i + rep) % 3; order = models[k:] + models[:k]
                if rep % 2: order.reverse()
                for m in order:
                    rec = run(m, t); rec["repeat"] = rep; stream.write(json.dumps(rec) + "\n")
    (OUT / "status.json").write_text(json.dumps({"stage": "verified", "samples": 3 * 256 * 3, "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    print("latency done")
