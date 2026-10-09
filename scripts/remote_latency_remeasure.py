"""Latency re-measurement under shared-host load (pre-registration §5).

All models are resident and interleaved per scene (order rotated per scene and
repeat), so slow host periods affect every model alike. Host load average and
GPU clock are recorded per sample. Predictions are checked against the
navtest-dev quality run (baselines) and cached-feature v1 head predictions.
No labels are opened.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from baseline_common import ROOT, file_sha, load_policy, sample_seed
from head_policy_runtime import OnlineHeadPolicy
from navsim_action_opt import make_skip_terminal_action
from navsim_current_dataset import load_policy_input, build_rollout_batch

BASELINES = {"B10": 10, "B05-opt": 5, "B03-opt": 3, "B02-opt": 2, "B01-opt": 1}
HEADS = ["SF3-current", "SF3-first", "SF3-terminal", "FMLP-current", "FMLP-first", "FMLP-terminal", "SF1-terminal"]

def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)

def load1():
    return float(Path("/proc/loadavg").read_text().split()[0])

def sm_clock():
    return int(subprocess.check_output(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "--query-gpu=clocks.current.sm", "--format=csv,noheader,nounits"], text=True).strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="latency_remeasure_2026-10-02")
    parser.add_argument("--scenes", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    assert not out.exists(), "Refusing to overwrite"
    out.mkdir(parents=True)
    def status(stage, **v):
        write(out / "status.json", {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **v})
        print(json.dumps({"stage": stage, **v}), flush=True)
    try:
        torch.set_num_threads(8)
        assert torch.cuda.device_count() == 1
        quality = ROOT / "runs/navtest_dev_quality_2026-10-02/predictions"
        v1 = ROOT / "runs/heads_navtest_dev_2026-10-02/v1"
        rank = lambda t: hashlib.sha256(f"17:latency:{t}".encode()).hexdigest()
        all_tokens = (ROOT / "splits/navtest-dev-tokens.txt").read_text().splitlines()
        tokens = sorted(all_tokens, key=rank)[:args.scenes]
        rows = {r["scene_token"]: r for r in map(json.loads, (ROOT / "splits/navtest-dev-manifest.jsonl").read_text().splitlines())}
        reference = {}
        for name in list(BASELINES) + HEADS:
            path = (quality if name in BASELINES else v1) / f"{name}.npz"
            with np.load(path, allow_pickle=False) as data:
                reference[name] = dict(zip(data["scene_tokens"].astype(str).tolist(), data["trajectories"]))
        inputs = {}
        for token in tokens:
            assert file_sha(rows[token]["input_path"]) == rows[token]["input_sha256"]
            inputs[token] = load_policy_input(rows[token]["input_path"])
        policy, loading = load_policy()
        raw_action, opt_action, _ = make_skip_terminal_action(policy)
        policy._skip_terminal_action_cache = False
        policy._rollout_action_chunk = lambda *a, **k: (opt_action if policy._skip_terminal_action_cache else raw_action)(*a, **k)
        heads = {name: OnlineHeadPolicy(ROOT / "runs/head_pilot_2026-10-01" / name, policy) for name in HEADS}
        models = list(BASELINES) + HEADS
        protocol = {"run_id": args.run_id, "split": "navtest_dev", "tokens": tokens, "models": models,
            "repeats": args.repeats, "warmup": "one full interleaved pass over 16 scenes before timing",
            "interleaving": "per scene, model order rotated by (scene index + repeat), reversed on odd repeats",
            "all_models_resident": True, "note": "head taps stay registered for all runs (identical hook overhead for every model)",
            "timing_scope": "preloaded CPU observation to usable CPU trajectory; disk/scoring excluded",
            "host_load_recorded": "/proc/loadavg 1-min before each sample", "gpu_clock_recorded": "per sample",
            "cpu_scoring_or_training_running": False, "loading": loading,
            "scripts": {n: file_sha(ROOT / "scripts" / n) for n in ["remote_latency_remeasure.py", "head_policy_runtime.py", "world_feature_tap.py"]}}
        write(out / "protocol.json", protocol)

        def run(name, token):
            torch.cuda.synchronize()
            load, clock = load1(), sm_clock()
            start = time.perf_counter()
            if name in BASELINES:
                policy._skip_terminal_action_cache = name.endswith("-opt")
                batch = build_rollout_batch(inputs[token], policy.config)
                with torch.inference_mode():
                    actions, _ = policy.predict(batch, 3, BASELINES[name], sample_seed(token, 0), video_denoise_fraction=.6)
                    trajectory = policy.pred_traj_to_poses(policy.actions_to_trajectory(actions))
            else:
                heads[name].record_head_events = False
                trajectory, _ = heads[name].predict(inputs[token], sample_seed(token, 0))
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            assert np.max(np.abs(np.asarray(trajectory, np.float32) - reference[name][token])) <= 1e-5, (name, token)
            return {"model": name, "scene_token": token, "policy_ms": elapsed, "host_load1": load, "sm_clock_mhz": clock}

        status("warmup")
        for i, token in enumerate(tokens[:16]):
            for name in models:
                run(name, token)
        with (out / "timings.jsonl").open("w") as stream:
            for repeat in range(args.repeats):
                for i, token in enumerate(tokens):
                    k = (i + repeat) % len(models)
                    order = models[k:] + models[:k]
                    if repeat % 2:
                        order.reverse()
                    for name in order:
                        record = run(name, token)
                        record["repeat"] = repeat
                        stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    if (i + 1) % 32 == 0:
                        status("timing", repeat=repeat, completed=i + 1, total=len(tokens), host_load1=load1())
        status("verified", samples=len(models) * len(tokens) * args.repeats)
    except Exception as error:
        status("failed", error=repr(error))
        raise
