"""Official CPU PDMS of a batch of head predictions on frozen pilot head-val."""
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

ROOT=Path.home()/"DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0,str(ROOT/"upstream/navsim-v1.1"))
import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer

MANIFEST={r["scene_token"]:r for r in (json.loads(line) for line in (ROOT/"splits/pilot-manifest.jsonl").read_text().splitlines())}
SCORER=SIMULATOR=LOADER=None
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def initialize():
    global SCORER,SIMULATOR,LOADER
    torch.set_num_threads(1)
    assert os.environ.get("OPENBLAS_CORETYPE")=="Haswell"
    assert np.max(np.abs(np.linalg.pinv(np.eye(40))-np.eye(40)))<1e-12
    sampling=TrajectorySampling(num_poses=40,interval_length=.1)
    SIMULATOR=PDMSimulator(sampling);SCORER=PDMScorer(sampling)
    LOADER=MetricCacheLoader(ROOT/"caches/metrics/pilot-head-val1000")
def score(item):
    token,poses=item; row={"scene_token":token,"log_name":MANIFEST[token]["log_name"],"valid":False,"failure_reason":""}
    try:
        assert poses.shape==(8,3) and np.isfinite(poses).all()
        target_path=Path(MANIFEST[token]["label_path"])
        assert sha(target_path)==MANIFEST[token]["target_sha256"]
        with np.load(target_path,allow_pickle=False) as data:gt=data["future_poses_4s"].copy()
        distance=np.linalg.norm(poses[:,:2].astype(np.float64)-gt[:,:2].astype(np.float64),axis=1)
        cache=LOADER.get_from_token(token)
        assert Path(cache.file_path).parent.name==token
        assert cache.ego_state.time_point.time_us==MANIFEST[token]["timestamp_us"]
        result=asdict(pdm_score(cache,Trajectory(poses,TrajectorySampling(num_poses=8,interval_length=.5)),
                               SIMULATOR.proposal_sampling,SIMULATOR,SCORER))
        assert all(np.isfinite(v) and 0<=v<=1.000001 for v in result.values())
        row.update(result,ade_4s_m=float(distance.mean()),fde_4s_m=float(distance[-1]),
                   at_fault_collision=bool(np.isfinite(SCORER.time_to_at_fault_collision(1))),
                   ttc_infraction=bool(np.isfinite(SCORER.time_to_ttc_infraction(1))),valid=True,
                   target_sha256=MANIFEST[token]["target_sha256"],metric_cache_sha256=sha(LOADER.metric_cache_paths[token]))
    except Exception as error:row.update(failure_reason=repr(error),traceback=traceback.format_exc())
    return row

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--predictions",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--workers",type=int,default=8);args=parser.parse_args()
    path=Path(args.predictions);out=Path(args.output)
    assert path.resolve().is_relative_to((ROOT/"runs").resolve())
    assert out.resolve().is_relative_to((ROOT/"runs").resolve())
    tokens_expected=set((ROOT/"splits/pilot-head-val-tokens.txt").read_text().splitlines())
    assert len(tokens_expected)==1000
    loader=MetricCacheLoader(ROOT/"caches/metrics/pilot-head-val1000")
    assert set(loader.tokens)==tokens_expected
    for p in loader.metric_cache_paths.values():
        assert Path(p).is_file() and Path(p).resolve().is_relative_to((ROOT/"caches/metrics/pilot-head-val1000").resolve())
    with np.load(path,allow_pickle=False) as data:
        tokens=data["scene_tokens"].astype(str).tolist();poses=data["trajectories"].copy()
    assert len(tokens)==len(set(tokens))==1000 and set(tokens)==tokens_expected
    assert poses.shape==(1000,8,3) and np.isfinite(poses).all()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("fork"),initializer=initialize) as pool:
        rows=list(pool.map(score,zip(tokens,poses),chunksize=16))
    failures=[r for r in rows if not r["valid"]]
    columns=sorted(set(k for r in rows for k in r));out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    out.with_suffix(".failures.json").write_text(json.dumps(failures,indent=2)+"\n")
    assert not failures,f"Official scorer failed {len(failures)} of 1000 rows"
    dev=set((ROOT/"splits/devval256-tokens.txt").read_text().splitlines())
    fields=["score","no_at_fault_collisions","drivable_area_compliance","time_to_collision_within_bound","comfort","ego_progress","driving_direction_compliance"]
    def means(subset):return {field:float(np.mean([r[field] for r in subset])*100) for field in fields}
    report={"coverage":1.,"scenes":1000,"dev_scenes":256,"failures":0,"pilot1000":means(rows),
        "dev256":means([r for r in rows if r["scene_token"] in dev]),"prediction_sha256":sha(path),"scores_sha256":sha(out),
        "official_navsim_commit":"3e8291bfa89ff247231e0227778840cd0a036896","navtest_used":False}
    out.with_suffix(".summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
