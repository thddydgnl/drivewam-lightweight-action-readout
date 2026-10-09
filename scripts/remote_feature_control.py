"""Input-matched feature swap with the original ego condition preserved.

Predict on GPU, then score in a separate CPU invocation. Pairing never reads
targets and excludes the same scene/log. A swap is an OOD diagnostic, not a
causal intervention on future dynamics.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path

ROOT=Path.home()/"DriveWAM_Lightweight_Action_Readout"

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["predict","score"],required=True);args=parser.parse_args()
    out=ROOT/"runs/feature_control_2026-10-01";out.mkdir(parents=True,exist_ok=True)
    import numpy as np
    import torch
    def status(stage,**values):
        p=out/(args.stage+"-status.json");temp=p.with_suffix(".tmp")
        record={"stage":stage,"updated_at_utc":datetime.now(timezone.utc).isoformat(),**values}
        temp.write_text(json.dumps(record,indent=2)+"\n");temp.replace(p);print(json.dumps(record),flush=True)
    try:
        if args.stage=="predict":
            from safetensors import safe_open
            from head_policy_runtime import OnlineHeadPolicy
            from remote_train_heads import CONFIGS,sha
            torch.set_num_threads(8);assert torch.cuda.device_count()==1
            manifest_path=ROOT/"splits/pilot-manifest.jsonl"
            manifest={r["scene_token"]:r for r in map(json.loads,manifest_path.read_text().splitlines())}
            tokens=(ROOT/"splits/devval256-tokens.txt").read_text().splitlines()
            pool=(ROOT/"splits/pilot-head-val-tokens.txt").read_text().splitlines()
            rng=np.random.default_rng(17);pairs=[];excluded=[]
            def matched(first,second):
                a,b=manifest[first],manifest[second]
                return first!=second and a["log_name"]!=b["log_name"] and a["command_id"]==b["command_id"] and (a["speed_mps"]<2)==(b["speed_mps"]<2)
            for token in tokens:
                candidates=sorted(t for t in pool if matched(token,t))
                same_map=[t for t in candidates if manifest[t]["map_name"]==manifest[token]["map_name"]]
                if same_map:candidates=same_map
                if not candidates:excluded.append(token);continue
                donor=candidates[int(rng.integers(len(candidates)))]
                pairs.append({"scene_token":token,"donor_token":donor,"log_name":manifest[token]["log_name"],
                    "donor_log_name":manifest[donor]["log_name"],"command_id":manifest[token]["command_id"],
                    "speed_bin":"under_2_mps" if manifest[token]["speed_mps"]<2 else "at_least_2_mps",
                    "same_map":manifest[token]["map_name"]==manifest[donor]["map_name"]})
            assert len(pairs)>=32 and len({r["log_name"] for r in pairs})>=2,"Insufficient matched scenes; report exclusions before interpreting"
            needed={r[key] for r in pairs for key in ["scene_token","donor_token"]}
            cache=ROOT/"caches/features/world_features_pilot_2026-10-01"
            index={r["scene_token"]:r for r in map(json.loads,(cache/"index.jsonl").read_text().splitlines())}
            features={}
            for token in needed:
                row=index[token]
                with safe_open(row["shard_path"],framework="pt",device="cpu") as stored:
                    position=row["row_index"]
                    features[token]=(stored.get_slice("mean")[position:position+1][0,:,5].clone(),
                        stored.get_slice("dense")[position:position+1][0].index_select(1,torch.tensor([1,3,5])),
                        stored.get_slice("ego")[position:position+1][0].clone())
            protocol={"scope":"exploratory feature swap on dev256; selected checkpoints from pilot1000","pairs":pairs,"excluded_tokens":excluded,
                "pairing":"seed17, same command and speed bin, different log; same region preferred; no GT or target sorting",
                "reused_donors":len(pairs)-len({r["donor_token"] for r in pairs}),"ego_condition":"original scene preserved",
                "manifest_sha256":sha(manifest_path),"feature_lock_sha256":sha(cache/"FEATURE_LOCK.json"),"train_seed":17,"generation_seed":0,
                "navtest_used":False,"causal_future_dynamics_claim":False,"scripts":{name:sha(ROOT/"scripts"/name) for name in ["remote_feature_control.py","head_policy_runtime.py","lightweight_heads.py"]}}
            protocol["selected_checkpoints"]={name:json.loads((ROOT/"runs/head_pilot_2026-10-01"/name/"summary.json").read_text())["best_by_pilot_PDMS"] for name in CONFIGS}
            (out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
            online_root=ROOT/"runs/heads_online_devval256_2026-10-01-r2"
            assert json.loads((online_root/"analysis-status.json").read_text())["stage"]=="verified"
            online_index={(r["model"],r["scene_token"]):r for r in map(json.loads,(online_root/"prediction-index.jsonl").read_text().splitlines())}
            for name in CONFIGS:
                policy=OnlineHeadPolicy(ROOT/"runs/head_pilot_2026-10-01"/name,cache_only=True)
                original,swapped=[],[]
                for row in pairs:
                    mean,dense,ego=features[row["scene_token"]];donor_mean,donor_dense,_=features[row["donor_token"]]
                    original.append(policy.predict_cached(mean[policy.phase],dense[policy.phase],ego))
                    swapped.append(policy.predict_cached(donor_mean[policy.phase],donor_dense[policy.phase],ego))
                    record=online_index[name,row["scene_token"]]
                    assert sha(record["prediction_path"])==record["prediction_sha256"]
                    with np.load(record["prediction_path"],allow_pickle=False) as online:
                        assert np.max(np.abs(original[-1]-online["trajectory"]))<=1e-5,"Control/main online prediction differs"
                original=np.stack(original);swapped=np.stack(swapped)
                if name=="ego-only":assert np.array_equal(original,swapped),"Ego control unexpectedly uses features"
                np.savez(out/(name+".npz"),scene_tokens=np.asarray([r["scene_token"] for r in pairs]),original=original,swapped=swapped)
                policy.close();del policy;torch.cuda.empty_cache();status("predicting",model=name,scenes=len(pairs))
            status("verified",scenes=len(pairs),models=len(CONFIGS))
        else:
            import pandas as pd
            from remote_pilot_head_score import score,initialize,sha
            from remote_baseline_analyze import QUALITY,cluster_bootstrap
            assert json.loads((out/"predict-status.json").read_text())["stage"]=="verified"
            status("scoring",completed_models=0,total_models=12)
            protocol=json.loads((out/"protocol.json").read_text());all_rows=[];results={}
            for path in sorted(out.glob("*.npz")):
                with np.load(path,allow_pickle=False) as stored:
                    tokens=stored["scene_tokens"].astype(str).tolist();arrays={kind:stored[kind].copy() for kind in ["original","swapped"]}
                assert tokens==[r["scene_token"] for r in protocol["pairs"]]
                for kind,poses in arrays.items():
                    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context("fork"),initializer=initialize) as workers:
                        rows=list(workers.map(score,zip(tokens,poses),chunksize=16))
                    assert all(r["valid"] for r in rows)
                    for row in rows:row.update(model=path.stem,condition=kind,prediction_sha256=sha(path))
                    all_rows.extend(rows)
                frame=pd.DataFrame(all_rows);subset=frame[frame.model==path.stem].copy();subset[QUALITY]*=100
                original=subset[subset.condition=="original"].set_index("scene_token").loc[tokens]
                swapped=subset[subset.condition=="swapped"].set_index("scene_token").loc[tokens]
                assert (original.target_sha256==swapped.target_sha256).all() and (original.metric_cache_sha256==swapped.metric_cache_sha256).all()
                delta=swapped[QUALITY+["ade_4s_m","fde_4s_m"]]-original[QUALITY+["ade_4s_m","fde_4s_m"]];delta["log_name"]=original.log_name
                results[path.stem]={"original_pdms":float(original.score.mean()),"swapped_pdms":float(swapped.score.mean()),
                    "swapped_minus_original":cluster_bootstrap(delta,QUALITY+["ade_4s_m","fde_4s_m"]),"scenes":len(tokens),"logs":int(original.log_name.nunique())}
                status("scoring",completed_models=len(results),total_models=12,model=path.stem)
            assert len(results)==12
            pd.DataFrame(all_rows).to_csv(out/"scores.csv",index=False)
            summary={"models":results,"scenes":len(protocol["pairs"]),"excluded_scenes":len(protocol["excluded_tokens"]),
                "interpretation":"Swap tests scene-specific feature dependence; OOD scene combinations do not prove causal future dynamics.","navtest_used":False}
            (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
            lock={"status":"input_matched_feature_swap_verified","files":[{"path":str(p),"sha256":sha(p)} for p in sorted(out.iterdir()) if p.suffix in [".npz",".csv",".json"] and p.name not in ["CONTROL_LOCK.json","score-status.json"]]}
            (out/"CONTROL_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")
            status("verified",models=12,scenes=len(protocol["pairs"]))
    except Exception as error:status("failed",error=repr(error));raise
