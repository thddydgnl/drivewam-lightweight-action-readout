"""Score complete baseline predictions in a separate canonical CPU environment.

The official and DriveWAM-vendored NAVSIM packages are loaded in separate
processes. A namespace package is explicit for the vendored tree, which omits
__init__.py; otherwise the installed official package would silently win.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.machinery
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback
import types

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
parser = argparse.ArgumentParser()
parser.add_argument("--run-id", default="opt_devval256_2026-10-01")
parser.add_argument("--backend", choices=["official", "vendored"], default="official")
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()
SOURCE = ROOT / ("upstream/navsim-v1.1" if args.backend == "official" else "work/DriveWAM")
if args.backend == "vendored":
    module = types.ModuleType("navsim")
    module.__package__ = "navsim"
    module.__path__ = [str(SOURCE / "navsim")]
    module.__spec__ = importlib.machinery.ModuleSpec("navsim", loader=None, is_package=True)
    module.__spec__.submodule_search_locations = module.__path__
    sys.modules["navsim"] = module
else:
    sys.path.insert(0, str(SOURCE))

import numpy as np
import torch
from threadpoolctl import threadpool_info
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation import batch_lqr_utils

OUT = ROOT / "runs" / args.run_id
CACHE = ROOT / "caches/metrics/devval256"
STATUS = OUT / f"score-{args.backend}-status.json"
MANIFEST = {row["scene_token"]: row for row in
            (json.loads(line) for line in (ROOT / "splits/pilot-manifest.jsonl").read_text().splitlines())}
SCORER, SIMULATOR, LOADER = None, None, None

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def status(stage, **extra):
    STATUS.write_text(json.dumps({"stage": stage, **extra}, indent=2) + "\n")
    print(json.dumps({"stage": stage, **extra}), flush=True)

def initialize():
    global SCORER, SIMULATOR, LOADER
    torch.set_num_threads(1)
    assert os.environ.get("OPENBLAS_CORETYPE") == "Haswell"
    assert np.max(np.abs(np.linalg.pinv(np.eye(40)) - np.eye(40))) < 1e-12
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    SIMULATOR = PDMSimulator(sampling)
    SCORER = PDMScorer(sampling, PDMScorerConfig())
    LOADER = MetricCacheLoader(CACHE)

def score_record(record):
    token = record["scene_token"]
    row = {key: record[key] for key in ["model", "scene_token", "generation_seed", "sample_seed", "log_name"]}
    row.update(backend=args.backend, valid=False, failure_reason="")
    start = time.monotonic()
    try:
        path = Path(record["prediction_path"])
        assert path.resolve().is_relative_to((OUT / "predictions").resolve())
        assert sha(path) == record["prediction_sha256"]
        with np.load(path, allow_pickle=False) as prediction:
            poses = prediction["trajectory"].copy()
            assert str(prediction["scene_token"]) == token
        assert poses.shape == (8, 3) and np.isfinite(poses).all()
        # Ground truth is accessed only here, outside GPU policy prediction.
        target_path = Path(MANIFEST[token]["label_path"])
        assert sha(target_path) == MANIFEST[token]["target_sha256"]
        with np.load(target_path, allow_pickle=False) as target:
            gt = target["future_poses_4s"].copy()
        delta = poses[:, :2].astype(np.float64) - gt[:, :2].astype(np.float64)
        distance = np.linalg.norm(delta, axis=1)
        heading_error = np.abs(np.arctan2(np.sin(poses[:, 2] - gt[:, 2]), np.cos(poses[:, 2] - gt[:, 2])))
        cache = LOADER.get_from_token(token)
        row.update(prediction_sha256=record["prediction_sha256"], target_sha256=MANIFEST[token]["target_sha256"],
                   metric_cache_sha256=sha(LOADER.metric_cache_paths[token]))
        assert Path(cache.file_path).parent.name == token
        assert cache.ego_state.time_point.time_us == MANIFEST[token]["timestamp_us"]
        result = pdm_score(cache, Trajectory(poses, TrajectorySampling(num_poses=8, interval_length=0.5)),
                           SIMULATOR.proposal_sampling, SIMULATOR, SCORER)
        metrics = asdict(result)
        assert all(np.isfinite(value) and 0 <= value <= 1.000001 for value in metrics.values())
        row.update(metrics, ade_4s_m=float(distance.mean()), fde_4s_m=float(distance[-1]),
                   ade_3s_m=float(distance[:6].mean()), fde_3s_m=float(distance[5]),
                   heading_mae_rad=float(heading_error.mean()),
                   at_fault_collision=bool(np.isfinite(SCORER.time_to_at_fault_collision(1))),
                   ttc_infraction=bool(np.isfinite(SCORER.time_to_ttc_infraction(1))), valid=True)
    except Exception as error:
        row["failure_reason"] = repr(error)
        row["traceback"] = traceback.format_exc()
    row["scoring_seconds"] = time.monotonic() - start
    return row

if __name__ == "__main__":
    try:
        assert json.loads((OUT / "inference-status.json").read_text())["stage"] == "verified", "Finish GPU timing before CPU scoring"
        assert os.environ.get("OPENBLAS_CORETYPE") == "Haswell"
        assert Path(inspect.getfile(pdm_score)).resolve().is_relative_to(SOURCE.resolve())
        assert Path(batch_lqr_utils.__file__).resolve().is_relative_to(SOURCE.resolve())
        csv_files = list((CACHE / "metadata").glob("*.csv"))
        assert len(csv_files) == 1, "Ambiguous cache metadata; reindex explicitly"
        tokens = set((ROOT / "splits/devval256-tokens.txt").read_text().splitlines())
        loader = MetricCacheLoader(CACHE)
        assert set(loader.tokens) == tokens
        for path in loader.metric_cache_paths.values():
            assert Path(path).is_file() and Path(path).resolve().is_relative_to(CACHE.resolve())
        predictions = [json.loads(line) for line in (OUT / "prediction-index.jsonl").read_text().splitlines()]
        keys = {(r["model"], r["generation_seed"], r["scene_token"]) for r in predictions}
        protocol = json.loads((OUT / "protocol.json").read_text())
        assert len(predictions) == len(keys) == protocol["expected_prediction_count"]
        for model in protocol["model_names"]:
            assert {r["scene_token"] for r in predictions if r["model"] == model and r["generation_seed"] == 0} == tokens
        status("scoring", backend=args.backend, requested=len(predictions), workers=args.workers)
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork"), initializer=initialize) as pool:
            for row in pool.map(score_record, predictions, chunksize=8):
                rows.append(row)
                if len(rows) % 64 == 0:
                    status("scoring", backend=args.backend, completed=len(rows), total=len(predictions))
        failures = [row for row in rows if not row["valid"]]
        output = OUT / f"scores-{args.backend}.csv"
        columns = sorted(set(key for row in rows for key in row))
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        (OUT / f"score-{args.backend}-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        assert not failures, f"Scorer failed {len(failures)} rows; incomplete coverage is an error"
        report = {"backend": args.backend, "source": str(SOURCE), "pdm_score_file": inspect.getfile(pdm_score),
             "batch_lqr_file": str(batch_lqr_utils.__file__), "batch_lqr_sha256": sha(batch_lqr_utils.__file__),
             "requested": len(predictions), "scored": len(rows), "primary_each_model": 256,
             "failures": 0, "coverage": 1.0, "prediction_index_sha256": sha(OUT / "prediction-index.jsonl"),
             "scores_sha256": sha(output), "proposal_sampling": {"num_poses": 40, "interval_length": 0.1},
             "model_trajectory_sampling": {"num_poses": 8, "interval_length": 0.5},
             "scorer_config": asdict(PDMScorerConfig()), "blas": threadpool_info(),
             "openblas_coretype": os.environ["OPENBLAS_CORETYPE"], "torch": torch.__version__,
             "navtest_used": False, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        (OUT / f"score-{args.backend}-summary.json").write_text(json.dumps(report, indent=2) + "\n")
        status("verified", backend=args.backend, scored=len(rows), coverage=1.0, failures=0)
    except Exception as error:
        status("failed", error=repr(error))
        raise
