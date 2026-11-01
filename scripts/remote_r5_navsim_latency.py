"""NAVSIM policy-latency re-measurement on an idle host (pre-registration 2026-11-01 §1.3).

Same protocol as remote_final_latency.py (256 fixed navtest-final scenes, 16 warm-up scenes,
3 repeats, model order rotated per scene and reversed on odd repeats, raw RGB -> trajectory)
for any set of models: B10 (official), Bk-opt, grid V{v}-A{k} (opt) and the frozen student
(seed 17). B10, B02-opt and the student must reproduce the 10/3 final-run trajectories (<= 1e-5).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from safetensors.torch import load_file
from baseline_common import ROOT, file_sha, sample_seed, load_policy
from navsim_current_dataset import load_policy_input, build_rollout_batch
from navsim_action_opt import make_skip_terminal_action
from world_feature_tap import WorldFeatureTap, PHASES
from heads_v3 import RegressionHead
from remote_navtest_final_infer import V3, ego_vector

def steps(name):
    """(video steps, action steps, opt) of a baseline name."""
    if name == "B10":
        return 3, 10, False
    m = re.fullmatch(r"V(\d)-A(\d+)", name) or re.fullmatch(r"B(\d+)-opt", name)
    return (int(m.group(1)), int(m.group(2)), True) if name.startswith("V") else (3, int(m.group(1)), True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--run-id", default="r5_navsim_latency_2026-11-01")
    a = p.parse_args()
    models = a.models.split(",")
    out = ROOT / "runs" / a.run_id
    assert not out.exists(); out.mkdir(parents=True)
    torch.set_num_threads(8)
    manifest = ROOT / "splits/navtest-final-manifest.regen-2026-11-01-latency256.jsonl"
    rows = {r["scene_token"]: r for r in map(json.loads, manifest.read_text().splitlines())}
    final_rows = [json.loads(line)["scene_token"] for line in (ROOT / "splits/navtest-final-manifest.jsonl").read_text().splitlines()]
    tokens = sorted(final_rows, key=lambda t: hashlib.sha256(f"17:final-latency:{t}".encode()).hexdigest())[:256]
    assert set(tokens) == set(rows)
    for t in tokens:
        assert file_sha(rows[t]["input_path"]) == rows[t]["input_sha256"]
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
    (out / "protocol.json").write_text(json.dumps({"run_id": a.run_id, "models": models, "scenes": 256, "repeats": 3,
        "warmup_scenes": 16, "preregistration": "reports/사전등록_보완실험_2026-11-01.md", "manifest": str(manifest),
        "script_sha256": file_sha(ROOT / "scripts/remote_r5_navsim_latency.py"),
        "gpu": torch.cuda.get_device_name(0), "created_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    def run(name, t):
        torch.cuda.synchronize(); load = float(Path("/proc/loadavg").read_text().split()[0]); s = time.perf_counter()
        if name == "student":
            r = tap.extract(build_rollout_batch(inputs[t], policy.config), sample_seed(t, 0), stop_at=PHASES[1], include_mean=False)
            ego = ((ego_vector(inputs[t]) - stats["ego_mean"]) / stats["ego_std"])[None].to("cuda:0")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                traj = head(r["dense"][0][None].to("cuda:0"), ego)["poses"].float().cpu()[0].numpy()
            assert r["forward_calls"] == 3
        else:
            v, k, is_opt = steps(name)
            policy._skip_terminal_action_cache = is_opt
            with torch.inference_mode():
                act, _ = policy.predict(build_rollout_batch(inputs[t], policy.config), v, k, sample_seed(t, 0), video_denoise_fraction=.6)
                traj = policy.pred_traj_to_poses(policy.actions_to_trajectory(act))
        torch.cuda.synchronize(); ms = (time.perf_counter() - s) * 1000
        if name in reference[t]:
            assert np.max(np.abs(np.asarray(traj, np.float32) - reference[t][name])) <= 1e-5, (name, t)
        return {"model": name, "scene_token": t, "policy_ms": ms, "host_load1": load, "sm_clock_mhz": 0}
    for t in tokens[:16]:
        for m in models: run(m, t)
    with (out / "timings.jsonl").open("w") as stream:
        for rep in range(3):
            for i, t in enumerate(tokens):
                k = (i + rep) % len(models); order = models[k:] + models[:k]
                if rep % 2: order.reverse()
                for m in order:
                    rec = run(m, t); rec["repeat"] = rep; stream.write(json.dumps(rec) + "\n")
            stream.flush()
            print(json.dumps({"repeat_done": rep, "utc": datetime.now(timezone.utc).isoformat()}), flush=True)
    (out / "status.json").write_text(json.dumps({"stage": "verified", "samples": 3 * 256 * len(models),
                                                 "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    print("latency done")
