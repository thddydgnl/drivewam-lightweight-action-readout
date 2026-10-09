"""Paired terminal features and B10 teacher trajectories for target diagnosis.

512 distinct training scenes, 256 validation scenes, three generation seeds.
Labels are not opened during extraction. This is not a new independent dataset.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from baseline_common import ROOT, file_sha, load_policy, sample_seed
from navsim_current_dataset import load_policy_input, build_rollout_batch
from remote_baseline_infer import tensor_sha, telemetry, write_json
from world_feature_tap import WorldFeatureTap, PHASES
from head_policy_runtime import observation_ego

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--run-id",default="target_diagnostic_2026-10-01");args=parser.parse_args()
    out=ROOT/"runs"/args.run_id;out.mkdir(parents=True,exist_ok=True)
    def status(stage,**extra):
        value={"stage":stage,"updated_at_utc":datetime.now(timezone.utc).isoformat(),**extra}
        write_json(out/"extract-status.json",value);print(json.dumps(value),flush=True)
    try:
        torch.set_num_threads(8);assert torch.cuda.device_count()==1
        manifest_path=ROOT/"splits/pilot-manifest.jsonl"
        manifest={r["scene_token"]:r for r in map(json.loads,manifest_path.read_text().splitlines())}
        train_all=(ROOT/"splits/pilot-head-train-tokens.txt").read_text().splitlines()
        val=(ROOT/"splits/devval256-tokens.txt").read_text().splitlines()
        # Input-only stratification across log/command/speed bins; no targets.
        groups={}
        for token in train_all:
            row=manifest[token];key=row["log_name"],row["command_id"],int(row["speed_mps"]>=2)
            groups.setdefault(key,[]).append(token)
        rng=np.random.default_rng(17)
        keys=sorted(groups);rng.shuffle(keys)
        for values in groups.values():rng.shuffle(values)
        train=[];depth=0
        while len(train)<512:
            for key in keys:
                if depth<len(groups[key]):train.append(groups[key][depth])
                if len(train)==512:break
            depth+=1
        assert len(set(train))==512 and len(set(val))==256 and not set(train)&set(val)
        assert not {manifest[t]["log_name"] for t in train}&{manifest[t]["log_name"] for t in val}
        cache=ROOT/"caches/features/world_features_pilot_2026-10-01"
        source_index={r["scene_token"]:r for r in map(json.loads,(cache/"index.jsonl").read_text().splitlines())}
        source=ROOT/"work/DriveWAM"
        assert subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True)==""
        assert subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()=="abbefb407c9c553f221522fc26077d307620adb4"
        weight_sha=file_sha(ROOT/"checkpoints/drivewam/navsim/diffusion_pytorch_model.safetensors")
        assert weight_sha=="27e4846b0a25e22fc299f7aca90c01d802072d2b297741c85ed074b689ef26a9"
        protocol={"run_id":args.run_id,"train_tokens":train,"val_tokens":val,"generation_seeds":[0,1,2],
            "distinct_train_scenes":512,"distinct_val_scenes":256,"augmented_train_rows":1536,"augmented_val_rows":768,
            "selection":"seed17 round robin over input log/command/speed strata; no targets",
            "manifest_sha256":file_sha(manifest_path),"checkpoint_sha256":weight_sha,"feature_lock_sha256":file_sha(cache/"FEATURE_LOCK.json"),
            "source_commit":"abbefb407c9c553f221522fc26077d307620adb4","source_modified":False,
            "phase":PHASES[2],"layer_indices_zero_based":[9,19,29],"teacher":"same-seed original B10 trajectory",
            "future_action_denoising":True,"GT_labels_opened":False,"navtest_used":False,
            "scripts":{name:file_sha(ROOT/"scripts"/name) for name in ["remote_target_diagnostic_extract.py","world_feature_tap.py","baseline_common.py","navsim_current_dataset.py"]}}
        protocol["input_distribution"]={split:{"logs":len({manifest[t]["log_name"] for t in chosen}),
            "command":dict(Counter(str(manifest[t]["command_id"]) for t in chosen)),
            "speed_bin":dict(Counter("under_2_mps" if manifest[t]["speed_mps"]<2 else "at_least_2_mps" for t in chosen)),
            "map":dict(Counter(manifest[t]["map_name"] for t in chosen))} for split,chosen in [("train",train),("val",val)]}
        path=out/"extract-protocol.json"
        if path.exists():assert json.loads(path.read_text())==protocol,"Resume protocol mismatch"
        else:write_json(path,protocol)
        status("loading_frozen_backbone")
        policy,loading=load_policy()
        tap=WorldFeatureTap(policy,layers=[9,19,29],collect_phases=[PHASES[2]])
        references={(r["scene_token"],r["generation_seed"]):r for r in map(json.loads,(ROOT/"runs/opt_devval256_2026-10-01/prediction-index.jsonl").read_text().splitlines()) if r["model"]=="B10"}
        records=[];shards=[];start=time.monotonic()
        for split,tokens in [("head_train",train),("head_val",val)]:
            for seed in [0,1,2]:
                for offset in range(0,len(tokens),128):
                    chosen=tokens[offset:offset+128];shard=out/f"{split}-seed{seed}-{offset:04d}.safetensors";receipt=shard.with_suffix(".json")
                    if receipt.exists():
                        report=json.loads(receipt.read_text());assert report["tokens"]==chosen and report["protocol_sha256"]==file_sha(path)
                        assert file_sha(shard)==report["sha256"]
                    else:
                        dense,ego,teacher=[],[],[]
                        for token in chosen:
                            row=manifest[token];assert file_sha(row["input_path"])==row["input_sha256"]
                            obs=load_policy_input(row["input_path"])
                            result=tap.extract(build_rollout_batch(obs,policy.config),sample_seed(token,seed),skip_actions=False,include_mean=False)
                            trajectory=policy.pred_traj_to_poses(policy.actions_to_trajectory(result["original_outputs"][0]))
                            assert np.isfinite(trajectory).all() and result["forward_calls"]==17
                            assert tensor_sha(result["latents"])==tensor_sha(result["original_outputs"][1])
                            if (token,seed) in references:
                                import hashlib
                                assert hashlib.sha256(trajectory.tobytes()).hexdigest()==references[token,seed]["trajectory_sha256"]
                            feature=result["dense"][0].cpu()
                            if seed==0:
                                entry=source_index[token]
                                with safe_open(entry["shard_path"],framework="pt",device="cpu") as old:
                                    reference=old.get_slice("dense")[entry["row_index"]:entry["row_index"]+1][0,2].index_select(0,torch.tensor([1,3,5]))
                                assert torch.equal(feature,reference),"Diagnostic/main pilot feature parity failed"
                            dense.append(feature);ego.append(observation_ego(obs));teacher.append(torch.from_numpy(trajectory.copy()))
                            del result
                        tensors={"dense":torch.stack(dense),"ego":torch.stack(ego),"teacher_poses":torch.stack(teacher)}
                        for value in tensors.values():assert torch.isfinite(value).all()
                        temporary=shard.with_suffix(".safetensors.tmp");save_file(tensors,str(temporary));temporary.replace(shard)
                        report={"path":str(shard),"sha256":file_sha(shard),"tokens":chosen,"split":split,"generation_seed":seed,
                            "protocol_sha256":file_sha(path),"bytes":shard.stat().st_size}
                        write_json(receipt,report);del tensors,dense,ego,teacher
                    shards.append(report)
                    for position,token in enumerate(chosen):records.append({"scene_token":token,"generation_seed":seed,"sample_seed":sample_seed(token,seed),
                        "split":split,"log_name":manifest[token]["log_name"],"shard_path":str(shard),"row_index":position,"shard_sha256":report["sha256"]})
                    status("extracting",completed=len(records),total=2304,elapsed_seconds=time.monotonic()-start)
        tap.close();assert len(records)==2304 and len({(r["scene_token"],r["generation_seed"]) for r in records})==2304
        (out/"index.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        write_json(out/"EXTRACT_LOCK.json",{"status":"paired_terminal_features_and_B10_targets_verified","shards":shards,
            "files":[{"path":str(out/name),"sha256":file_sha(out/name)} for name in ["extract-protocol.json","index.jsonl"]],
            "main_pilot_generation0_feature_bitwise_parity_scenes":768,"loading_info":loading,"telemetry_after":telemetry(),"elapsed_seconds":time.monotonic()-start})
        status("verified",rows=2304,distinct_scenes=768)
    except Exception as error:status("failed",error=repr(error));raise
