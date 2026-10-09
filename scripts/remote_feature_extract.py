"""Validated, resumable dense world feature cache for the frozen 4k/1k pilot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors.torch import save_file, load_file
from baseline_common import ROOT, load_policy, file_sha, sample_seed
from navsim_current_dataset import load_policy_input, build_rollout_batch
from world_feature_tap import WorldFeatureTap, PHASES, LAYERS
from remote_baseline_infer import tensor_sha, write_json, telemetry

def ego_vector(observation):
    return torch.cat([observation["history_poses"].flatten(), observation["ego_velocity_acc"],
                      torch.nn.functional.one_hot(observation["command_id"],num_classes=4).float()])

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache-id",default="world_features_pilot_2026-10-01")
    parser.add_argument("--shard-size",type=int,default=128)
    args=parser.parse_args()
    out=ROOT/"caches/features"/args.cache_id
    out.mkdir(parents=True,exist_ok=True)
    status_path=out/"status.json"
    def status(stage,**extra):
        write_json(status_path,{"stage":stage,"updated_at_utc":datetime.now(timezone.utc).isoformat(),**extra})
        print(json.dumps({"stage":stage,**extra}),flush=True)
    try:
        torch.set_num_threads(8)
        assert torch.cuda.device_count()==1
        if status_path.exists() and json.loads(status_path.read_text()).get("stage")=="verified":
            print("Cache already verified; preserving completed artifacts",flush=True)
            sys.exit(0)
        manifest_path=ROOT/"splits/pilot-manifest.jsonl"
        manifest={r["scene_token"]:r for r in (json.loads(line) for line in manifest_path.read_text().splitlines())}
        data_lock=json.loads((ROOT/"DATA_LOCK.json").read_text())
        assert file_sha(manifest_path)==data_lock["manifest_sha256"]
        train=(ROOT/"splits/pilot-head-train-tokens.txt").read_text().splitlines()
        val=(ROOT/"splits/pilot-head-val-tokens.txt").read_text().splitlines()
        assert len(train)==4000 and len(val)==1000 and not set(train)&set(val)
        assert not {manifest[t]["log_name"] for t in train}&{manifest[t]["log_name"] for t in val}
        assert all(manifest[t]["split"]=="head_train" for t in train)
        assert all(manifest[t]["split"]=="head_val" for t in val)
        source=ROOT/"work/DriveWAM"
        assert subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()=="abbefb407c9c553f221522fc26077d307620adb4"
        assert subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True)==""
        status("verifying_provenance")
        weight_sha=file_sha(ROOT/"checkpoints/drivewam/navsim/diffusion_pytorch_model.safetensors")
        assert weight_sha=="27e4846b0a25e22fc299f7aca90c01d802072d2b297741c85ed074b689ef26a9"
        scripts={name:file_sha(ROOT/"scripts"/name) for name in ["baseline_common.py","navsim_current_dataset.py","world_feature_tap.py","remote_feature_extract.py"]}
        policy,loading_info=load_policy()
        tap=WorldFeatureTap(policy)
        prototype={"cache_id":args.cache_id,"manifest_sha256":file_sha(manifest_path),"checkpoint_sha256":weight_sha,
            "source_commit":"abbefb407c9c553f221522fc26077d307620adb4","source_modified":False,
            "layer_indices_zero_based":LAYERS,"layer_depths_one_based":[i+1 for i in LAYERS],"phases":PHASES,
            "generation_seed":0,"sample_seed":"big-endian first 4 SHA256 bytes of UTF-8 scene_token:generation_seed",
            "video_solver_steps":3,"video_denoise_fraction":.6,"batch_size":1,
            "dense_dtype":"bfloat16","mean_dtype":"float32","dense_shape_per_scene":[3,6,112,3072],
            "mean_shape_per_scene":[3,6,3072],"ego_shape_per_scene":[20],"shard_size":args.shard_size,
            "train_tokens":train,"val_tokens":val,"scripts":scripts,"loading_info":loading_info,
            "future_action_denoising":False,"target_files_opened":False,"navtest_used":False}
        if (out/"protocol.json").exists():
            assert json.loads((out/"protocol.json").read_text())==prototype,"Resume protocol mismatch"
        else:
            write_json(out/"protocol.json",prototype)
        smoke=(ROOT/"splits/smoke32-tokens.txt").read_text().splitlines()
        assert len(smoke)==32 and set(smoke)<=set(train)
        parity=[]
        status("feature_parity",completed=0,total=32)
        for number,token in enumerate(smoke):
            assert file_sha(manifest[token]["input_path"])==manifest[token]["input_sha256"]
            obs=load_policy_input(manifest[token]["input_path"])
            batch=build_rollout_batch(obs,policy.config)
            seed=sample_seed(token,0)
            with torch.inference_mode():
                actions,latent=policy.predict(batch,3,10,seed,video_denoise_fraction=.6)
            raw_hashes=tensor_sha(actions),tensor_sha(latent)
            del actions,latent
            full=tap.extract(batch,seed,skip_actions=False)
            assert raw_hashes==(tensor_sha(full["original_outputs"][0]),tensor_sha(full["original_outputs"][1]))
            full_dense=full["dense"].cpu();full_world=tensor_sha(full["latents"])
            assert full_world==raw_hashes[1]
            del full
            world=tap.extract(batch,seed)
            assert torch.equal(full_dense,world["dense"].cpu())
            assert tensor_sha(world["latents"])==full_world
            phase_metadata=world["phase_metadata"]
            del world
            for phase in PHASES[:2]:
                early=tap.extract(batch,seed,stop_at=phase)
                count=PHASES.index(phase)+1
                assert torch.equal(full_dense[:count],early["dense"].cpu()),(token,phase)
                del early
            if number<2:
                changed=build_rollout_batch(obs,policy.config)
                generator=torch.Generator().manual_seed(17000+number)
                changed["images"][:,1:]=torch.randint(0,256,changed["images"][:,1:].shape,dtype=torch.uint8,generator=generator)
                changed["actions"]=torch.randn(changed["actions"].shape,generator=generator)*100
                permuted=tap.extract(changed,seed)
                assert torch.equal(full_dense,permuted["dense"].cpu())
                assert tensor_sha(permuted["latents"])==full_world
                del permuted,changed
            parity.append({"scene_token":token,"hook_raw_action_and_world_bitwise_equal":True,
                "world_only_features_bitwise_equal":True,"world_only_latents_bitwise_equal":True,
                "current_and_first_early_stop_features_bitwise_equal":True,
                "unused_future_rgb_action_permutation_checked":number<2})
            del full_dense
            if (number+1)%8==0:status("feature_parity",completed=number+1,total=32)
        write_json(out/"parity.json",{"scenes":32,"rows":parity,"phase_metadata":phase_metadata})
        write_json(out/"tensor-contract.json",{"dense_shape_per_scene":[3,6,112,3072],"dtype":"bfloat16",
            "mean_shape_per_scene":[3,6,3072],"phase_metadata":phase_metadata,
            "native_forward_calls":{"current":1,"first":3,"terminal":6},"future_action_calls":0})
        shard_reports=[];index=[];start=time.monotonic()
        for split,tokens in [("head_train",train),("head_val",val)]:
            for offset in range(0,len(tokens),args.shard_size):
                shard_tokens=tokens[offset:offset+args.shard_size]
                name=f"{split}-{offset:05d}.safetensors"
                path=out/name;receipt=out/(name+".json")
                if receipt.exists():
                    report=json.loads(receipt.read_text())
                    assert report["tokens"]==shard_tokens and path.is_file() and file_sha(path)==report["sha256"]
                    assert report["protocol_sha256"]==file_sha(out/"protocol.json")
                else:
                    dense,means,egos=[],[],[]
                    for token in shard_tokens:
                        row=manifest[token]
                        assert file_sha(row["input_path"])==row["input_sha256"]
                        obs=load_policy_input(row["input_path"])
                        result=tap.extract(build_rollout_batch(obs,policy.config),sample_seed(token,0))
                        assert result["phase_metadata"]==phase_metadata
                        dense.append(result["dense"].cpu());means.append(result["mean"].cpu());egos.append(ego_vector(obs))
                        del result
                    tensors={"dense":torch.stack(dense),"mean":torch.stack(means),"ego":torch.stack(egos)}
                    for tensor in tensors.values():assert torch.isfinite(tensor).all()
                    temporary=path.with_suffix(".safetensors.tmp")
                    save_file(tensors,str(temporary),metadata={"protocol_sha256":file_sha(out/"protocol.json"),"split":split})
                    temporary.replace(path)
                    report={"path":str(path),"tokens":shard_tokens,"split":split,"bytes":path.stat().st_size,
                        "sha256":file_sha(path),"protocol_sha256":file_sha(out/"protocol.json")}
                    write_json(receipt,report)
                    del tensors,dense,means,egos
                shard_reports.append(report)
                for row_index,token in enumerate(shard_tokens):
                    index.append({"scene_token":token,"split":split,"log_name":manifest[token]["log_name"],
                        "sample_seed":sample_seed(token,0),"generation_seed":0,"input_sha256":manifest[token]["input_sha256"],
                        "shard_path":str(path),"row_index":row_index,"shard_sha256":report["sha256"]})
                status("extracting",completed=len(index),total=5000,split=split,elapsed_seconds=time.monotonic()-start)
        assert len(index)==len({r["scene_token"] for r in index})==5000
        (out/"index.jsonl").write_text("".join(json.dumps(r)+"\n" for r in index))
        tap.close()
        lock={"status":"pilot_current_first_terminal_dense_features_verified","cache_id":args.cache_id,
            "workspace":str(ROOT),"train_scenes":4000,"val_scenes":1000,"generation_seed":0,"navtest_used":False,
            "files":[{"path":str(out/name),"sha256":file_sha(out/name)} for name in ["protocol.json","parity.json","tensor-contract.json","index.jsonl"]],
            "shards":shard_reports,"dense_cache_bytes":sum(r["bytes"] for r in shard_reports),
            "elapsed_seconds":time.monotonic()-start,"telemetry_after":telemetry()}
        write_json(out/"FEATURE_LOCK.json",lock)
        status("verified",train_scenes=4000,val_scenes=1000,bytes=lock["dense_cache_bytes"])
    except Exception as error:
        status("failed",error=repr(error))
        raise
