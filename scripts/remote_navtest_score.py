"""Assemble navtest-dev predictions and score them with the official NAVSIM v1.1 scorer.

  assemble --run-id R             merge GPU shards into predictions/<model>.npz
  score --predictions P --output O  official PDMS + ADE/FDE per scene (scorer env)
Ground truth is read only in the scoring stage.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
MANIFEST_PATH = ROOT / "splits/navtest-dev-manifest.jsonl"
TOKENS_PATH = ROOT / "splits/navtest-dev-tokens.txt"
CACHE = ROOT / "caches/metrics/navtest-dev"
SCORER = SIMULATOR = LOADER = MANIFEST = None

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def assemble(run_id):
    out = ROOT / "runs" / run_id
    tokens = TOKENS_PATH.read_text().splitlines()
    protocol = json.loads((out / "protocol.json").read_text())
    for shard in [0, 1]:
        assert json.loads((out / f"inference-shard{shard}-status.json").read_text())["stage"] == "verified"
    collected, seeds = {}, {}
    for receipt in sorted(out.glob("shard*/*.json")):
        report = json.loads(receipt.read_text())
        assert sha(report["trajectories_path"]) == report["trajectories_sha256"]
        with np.load(report["trajectories_path"], allow_pickle=False) as data:
            names = data["scene_tokens"].astype(str).tolist()
            assert names == report["tokens"]
            for model in protocol["models"]:
                for token, trajectory in zip(names, data[model.replace("-", "_")]):
                    collected.setdefault(model, {})[token] = trajectory
            for key, trajectory in zip(data["seed_keys"].astype(str).tolist(), data["seed_trajectories"]):
                seeds[key] = trajectory
    (out / "predictions").mkdir(exist_ok=True)
    for model, values in collected.items():
        assert set(values) == set(tokens), model
        np.savez(out / "predictions" / f"{model}.npz", scene_tokens=np.asarray(tokens),
                 trajectories=np.stack([values[t] for t in tokens]))
    assert len(seeds) == 2 * len(protocol["seed_sensitivity"]["tokens"])
    keys = sorted(seeds)
    np.savez(out / "predictions" / "B10-seed-sensitivity.npz", keys=np.asarray(keys), trajectories=np.stack([seeds[k] for k in keys]))
    print(json.dumps({"models": sorted(collected), "scenes": len(tokens), "seed_rows": len(keys)}))

def constant_velocity(run_id):
    # Same definition as the pilot CV: ego-local current vx/vy fixed for 4 s, yaw zero, observation only.
    tokens = TOKENS_PATH.read_text().splitlines()
    rows = {r["scene_token"]: r for r in map(json.loads, MANIFEST_PATH.read_text().splitlines())}
    times = np.arange(1, 9, dtype=np.float32) * .5
    poses = []
    for token in tokens:
        with np.load(rows[token]["input_path"], allow_pickle=False) as data:
            vx, vy = data["ego_velocity_acc"][:2]
        poses.append(np.stack([vx * times, vy * times, np.zeros_like(times)], axis=1))
    out = ROOT / "runs" / run_id / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "CV.npz", scene_tokens=np.asarray(tokens), trajectories=np.stack(poses).astype(np.float32))

def initialize():
    global SCORER, SIMULATOR, LOADER, MANIFEST
    sys.path.insert(0, str(ROOT / "upstream/navsim-v1.1"))
    import torch
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    torch.set_num_threads(1)
    assert os.environ.get("OPENBLAS_CORETYPE") == "Haswell"
    assert np.max(np.abs(np.linalg.pinv(np.eye(40)) - np.eye(40))) < 1e-12
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    SIMULATOR, SCORER = PDMSimulator(sampling), PDMScorer(sampling)
    LOADER = MetricCacheLoader(CACHE)
    MANIFEST = {r["scene_token"]: r for r in map(json.loads, MANIFEST_PATH.read_text().splitlines())}

def score_one(item):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    token, poses = item
    row = {"scene_token": token, "log_name": MANIFEST[token]["log_name"], "map_name": MANIFEST[token]["map_name"],
           "command_id": MANIFEST[token]["command_id"], "speed_mps": MANIFEST[token]["speed_mps"], "valid": False, "failure_reason": ""}
    try:
        assert poses.shape == (8, 3) and np.isfinite(poses).all()
        assert sha(MANIFEST[token]["label_path"]) == MANIFEST[token]["target_sha256"]
        with np.load(MANIFEST[token]["label_path"], allow_pickle=False) as data:
            gt = data["future_poses_4s"].copy()
        distance = np.linalg.norm(poses[:, :2].astype(np.float64) - gt[:, :2].astype(np.float64), axis=1)
        heading = np.abs(np.arctan2(np.sin(poses[:, 2] - gt[:, 2]), np.cos(poses[:, 2] - gt[:, 2])))
        cache = LOADER.get_from_token(token)
        assert Path(cache.file_path).parent.name == token
        assert cache.ego_state.time_point.time_us == MANIFEST[token]["timestamp_us"]
        result = asdict(pdm_score(cache, Trajectory(poses, TrajectorySampling(num_poses=8, interval_length=.5)),
                                  SIMULATOR.proposal_sampling, SIMULATOR, SCORER))
        assert all(np.isfinite(v) and 0 <= v <= 1.000001 for v in result.values())
        row.update(result, ade_4s_m=float(distance.mean()), fde_4s_m=float(distance[-1]),
                   ade_3s_m=float(distance[:6].mean()), fde_3s_m=float(distance[5]), heading_mae_rad=float(heading.mean()),
                   at_fault_collision=bool(np.isfinite(SCORER.time_to_at_fault_collision(1))),
                   ttc_infraction=bool(np.isfinite(SCORER.time_to_ttc_infraction(1))), valid=True)
    except Exception as error:
        row.update(failure_reason=repr(error), traceback=traceback.format_exc())
    return row

def score(predictions, output, workers):
    tokens = TOKENS_PATH.read_text().splitlines()
    with np.load(predictions, allow_pickle=False) as data:
        names = data["scene_tokens"].astype(str).tolist()
        poses = data["trajectories"].astype(np.float32)
    assert names == tokens and poses.shape == (len(tokens), 8, 3) and np.isfinite(poses).all()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"), initializer=initialize) as pool:
        rows = list(pool.map(score_one, zip(names, poses), chunksize=16))
    failures = [r for r in rows if not r["valid"]]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted(set(k for r in rows for k in r))
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    assert not failures, f"{len(failures)} scoring failures: {failures[0]['failure_reason']}"
    summary = {"predictions": str(predictions), "prediction_sha256": sha(predictions), "scores_sha256": sha(output),
               "scenes": len(rows), "coverage": 1.0, "failures": 0, "pdms": float(np.mean([r["score"] for r in rows]) * 100),
               "ade_4s_m": float(np.mean([r["ade_4s_m"] for r in rows])),
               "official_navsim_commit": "3e8291bfa89ff247231e0227778840cd0a036896", "split": TOKENS_PATH.stem.replace("-tokens", "")}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["assemble", "cv", "score"])
    parser.add_argument("--run-id")
    parser.add_argument("--predictions")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--split", choices=["dev", "final"], default="dev")
    args = parser.parse_args()
    if args.split == "final":                      # pre-registration §8; set before worker fork
        MANIFEST_PATH = ROOT / "splits/navtest-final-manifest.jsonl"
        TOKENS_PATH = ROOT / "splits/navtest-final-tokens.txt"
        CACHE = ROOT / "caches/metrics/navtest-final"
    if args.stage == "assemble":
        assemble(args.run_id)
    elif args.stage == "cv":
        constant_velocity(args.run_id)
    else:
        score(args.predictions, args.output, args.workers)
