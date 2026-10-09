"""Diagnose the absolute-latency inflation of the interleaved re-measurement.

B10 only, 32 navtest-dev scenes, three conditions interleaved per scene:
plain / nvidia-smi query before each sample / 7 head taps registered.
"""
import hashlib, json, subprocess, time
from pathlib import Path
import numpy as np, torch
from baseline_common import ROOT, sample_seed, load_policy
from navsim_current_dataset import load_policy_input, build_rollout_batch
from world_feature_tap import WorldFeatureTap

rows = {r["scene_token"]: r for r in map(json.loads, (ROOT / "splits/navtest-dev-manifest.jsonl").read_text().splitlines())}
tokens = sorted(rows, key=lambda t: hashlib.sha256(f"17:latency:{t}".encode()).hexdigest())[:32]
inputs = {t: load_policy_input(rows[t]["input_path"]) for t in tokens}
policy, _ = load_policy()
taps = []
def b10(token):
    torch.cuda.synchronize(); s = time.perf_counter()
    with torch.inference_mode():
        a, _ = policy.predict(build_rollout_batch(inputs[token], policy.config), 3, 10, sample_seed(token, 0), video_denoise_fraction=.6)
        policy.pred_traj_to_poses(policy.actions_to_trajectory(a))
    torch.cuda.synchronize(); return (time.perf_counter() - s) * 1000
for t in tokens[:8]: b10(t)
res = {"plain": [], "nvidia_smi_before": [], "taps_registered": []}
for t in tokens:
    res["plain"].append(b10(t))
    subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=clocks.current.sm", "--format=csv,noheader,nounits"])
    res["nvidia_smi_before"].append(b10(t))
    taps = [WorldFeatureTap(policy, [9, 19, 29] if i < 3 else [29], collect_phases=["current_primer"], capture_latents=False, record_metadata=False) for i in range(7)]
    res["taps_registered"].append(b10(t))
    for tap in taps: tap.close()
out = {k: {"p50": float(np.median(v)), "mean": float(np.mean(v)), "n": len(v)} for k, v in res.items()}
(ROOT / "runs/latency_remeasure_2026-10-02/overhead-check.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
