"""Official GT metrics and B10 imitation error, aggregating seeds per scene."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from remote_pilot_head_score import ROOT, score, initialize, sha
from remote_baseline_analyze import QUALITY, cluster_bootstrap

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--predictions",required=True);parser.add_argument("--output",required=True);args=parser.parse_args()
    path=Path(args.predictions);out=Path(args.output)
    assert path.resolve().is_relative_to((ROOT/"runs/target_diagnostic_2026-10-01").resolve())
    assert out.resolve().is_relative_to(path.parent.parent.resolve())
    protocol=json.loads((ROOT/"runs/target_diagnostic_2026-10-01/extract-protocol.json").read_text())
    with np.load(path,allow_pickle=False) as stored:
        tokens=stored["scene_tokens"].astype(str).tolist();seeds=stored["generation_seeds"].tolist()
        poses=stored["trajectories"].copy();teacher=stored["teacher_poses"].copy()
    assert len(tokens)==768 and set(tokens)==set(protocol["val_tokens"])
    assert set(zip(tokens,seeds))=={(token,seed) for token in protocol["val_tokens"] for seed in [0,1,2]}
    assert poses.shape==teacher.shape==(768,8,3) and np.isfinite(poses).all() and np.isfinite(teacher).all()
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context("fork"),initializer=initialize) as workers:
        rows=list(workers.map(score,zip(tokens,poses),chunksize=16))
    assert all(r["valid"] for r in rows)
    distance=np.linalg.norm(poses[...,:2].astype(np.float64)-teacher[...,:2].astype(np.float64),axis=-1)
    for position,row in enumerate(rows):
        row.update(generation_seed=seeds[position],imitation_ade_4s_m=float(distance[position].mean()),imitation_fde_4s_m=float(distance[position,-1]))
    frame=pd.DataFrame(rows);frame.to_csv(out,index=False);frame[QUALITY]*=100
    metrics=QUALITY+["ade_4s_m","fde_4s_m","imitation_ade_4s_m","imitation_fde_4s_m"]
    per_scene=frame.groupby(["scene_token","log_name"],sort=True)[metrics].mean().reset_index()
    def variation(array):
        ordered=np.stack([array[np.asarray([i for i,(t,s) in enumerate(zip(tokens,seeds)) if t==token])] for token in protocol["val_tokens"]])
        return float(np.sqrt(np.square(ordered[...,:2]-ordered[...,:2].mean(axis=1,keepdims=True)).mean()))
    result={"scenes":256,"generation_rows":768,"logs":int(per_scene.log_name.nunique()),"coverage":1.,"failures":0,
        "seed_averaged_scene_metrics":cluster_bootstrap(per_scene,metrics),
        "quality_by_generation_seed":{str(seed):frame[frame.generation_seed==seed][metrics].mean().to_dict() for seed in [0,1,2]},
        "mean_within_scene_pdms_population_std_points":float(frame.pivot(index="scene_token",columns="generation_seed",values="score").std(axis=1,ddof=0).mean()),
        "student_within_scene_xy_rms_m":variation(poses),"B10_within_scene_xy_rms_m":variation(teacher),
        "prediction_sha256":sha(path),"scores_sha256":sha(out),"navtest_used":False,
        "uncertainty":"First average three generation seeds within each scene, then log-cluster bootstrap; 768 rows are not 768 independent scenes."}
    out.with_suffix(".summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result),flush=True)
