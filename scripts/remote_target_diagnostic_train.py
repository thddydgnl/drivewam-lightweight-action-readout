"""Same SF3 initialization, input rows, minibatches and 2000 steps; target only differs."""
import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from lightweight_heads import build_head, normalize_deltas, decode_poses, action_loss, parameter_count
from remote_train_heads import ROOT, sha, write, seed_all, gradient_gate

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--target",choices=["GT","Teacher"],required=True);args=parser.parse_args()
    root=ROOT/"runs/target_diagnostic_2026-10-01";out=root/("D-"+args.target);out.mkdir(parents=True,exist_ok=True)
    def status(stage,**values):
        write(out/"status.json",{"stage":stage,**values});print(json.dumps({"target":args.target,"stage":stage,**values}),flush=True)
    try:
        torch.set_num_threads(8);assert torch.cuda.device_count()==1;device=torch.device("cuda:0")
        assert json.loads((root/"extract-status.json").read_text())["stage"]=="verified"
        lock=json.loads((root/"EXTRACT_LOCK.json").read_text());extract=json.loads((root/"extract-protocol.json").read_text())
        for record in lock["files"]+lock["shards"]:assert sha(record["path"])==record["sha256"]
        index=list(map(json.loads,(root/"index.jsonl").read_text().splitlines()))
        dense,ego,teacher=[],[],[]
        for shard in lock["shards"]:
            with safe_open(shard["path"],framework="pt",device="cpu") as stored:
                dense.append(stored.get_tensor("dense"));ego.append(stored.get_tensor("ego"));teacher.append(stored.get_tensor("teacher_poses"))
        dense=torch.cat(dense);ego=torch.cat(ego);teacher=torch.cat(teacher)
        manifest={r["scene_token"]:r for r in map(json.loads,(ROOT/"splits/pilot-manifest.jsonl").read_text().splitlines())}
        assert sha(ROOT/"splits/pilot-manifest.jsonl")==extract["manifest_sha256"]
        ground_truth=[]
        for row in index:
            target=manifest[row["scene_token"]];assert sha(target["label_path"])==target["target_sha256"]
            with np.load(target["label_path"],allow_pickle=False) as stored:ground_truth.append(torch.from_numpy(stored["future_poses_4s"].copy()))
        ground_truth=torch.stack(ground_truth)
        train=torch.tensor([i for i,row in enumerate(index) if row["split"]=="head_train"])
        val=torch.tensor([i for i,row in enumerate(index) if row["split"]=="head_val"])
        assert len(train)==1536 and len(val)==768
        # Fit ego statistics on distinct diagnostic training scenes (seed 0).
        unique=torch.tensor([i for i,row in enumerate(index) if row["split"]=="head_train" and row["generation_seed"]==0])
        stats={"ego_mean":ego[unique].mean(0),"ego_std":ego[unique].std(0,unbiased=False).clamp_min(1e-5)}
        save_file(stats,str(out/"stats.safetensors"));ego=(ego-stats["ego_mean"])/stats["ego_std"]
        seed_all(17);model=build_head("sf3").to(device)
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(out/"initial.safetensors"))
        poses=ground_truth if args.target=="GT" else teacher
        deltas=poses-torch.cat([torch.zeros_like(poses[:,:1]),poses[:,:-1]],dim=1)
        normalized=normalize_deltas(deltas)
        assert torch.max(torch.abs(decode_poses(normalized)-poses))<1e-4
        protocol={"target":args.target,"structure":"SF3-terminal proj512 / 16 queries / layers10,20,30","train_seed":17,
            "steps":2000,"batch_size":32,"optimizer":"AdamW","learning_rate":3e-4,"weight_decay":.01,
            "warmup_steps":100,"schedule":"cosine","loss":"position Huber + .2 circular yaw + .1 normalized-delta Huber",
            "target_definition":"recorded GT trajectory" if args.target=="GT" else "same-world original B10 trajectory",
            "minibatches":"same isolated CPU generator seed17+100003*epoch for both targets",
            "distinct_training_scenes":512,"generation_seeds":[0,1,2],"distinct_validation_scenes":256,
            "statistics":"ego affine fit only on 512 distinct diagnostic train scenes; head LayerNorm for dense world",
            "EXTRACT_LOCK_sha256":sha(root/"EXTRACT_LOCK.json"),"stats_sha256":sha(out/"stats.safetensors"),
            "initial_parameters_sha256":sha(out/"initial.safetensors"),
            "comparison_checkpoint":"predeclared final step2000; scores every500 are learning diagnostics, not selection",
            "navtest_used":False,"scripts":{name:sha(ROOT/"scripts"/name) for name in ["remote_target_diagnostic_train.py","remote_target_diagnostic_score.py","lightweight_heads.py"]}}
        if (out/"protocol.json").exists():assert json.loads((out/"protocol.json").read_text())==protocol
        else:write(out/"protocol.json",protocol)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01,betas=(.9,.999))
        history=[];checks=[];step=0;order=None;epoch=0;position=0;gradients=None
        resume=out/"last.pt"
        if resume.exists():
            saved=torch.load(resume,map_location="cpu",weights_only=False);assert saved["protocol_sha256"]==sha(out/"protocol.json")
            model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
            for state in optimizer.state.values():
                for key,value in state.items():
                    if isinstance(value,torch.Tensor):state[key]=value.to(device)
            step=saved["step"];order=saved["order"];epoch=saved["epoch"];position=saved["position"];history=saved["history"];checks=saved["checks"];gradients=saved["gradients"]
            random.setstate(saved["python_rng"]);np.random.set_state(saved["numpy_rng"]);torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state(saved["cuda_rng"],device)
        started=time.monotonic()
        while step<2000:
            if order is None or position>=len(train):
                epoch+=1;order=train[torch.randperm(len(train),generator=torch.Generator().manual_seed(17+100003*epoch))];position=0
            selected=order[position:position+32];position+=len(selected);step+=1
            lr=3e-4*(step/100 if step<=100 else .5*(1+math.cos(math.pi*(step-100)/1900)))
            for group in optimizer.param_groups:group["lr"]=lr
            model.train();optimizer.zero_grad(set_to_none=True)
            world=dense[selected].to(device);condition=ego[selected].to(device);target_poses=poses[selected].to(device);target_norm=normalized[selected].to(device)
            with torch.autocast("cuda",dtype=torch.bfloat16):prediction=model(world,condition);loss,_=action_loss(prediction,target_poses,target_norm)
            assert torch.isfinite(loss);loss.backward()
            if gradients is None:gradients=gradient_gate(model)
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));optimizer.step()
            if step%100==0:history.append({"step":step,"loss":float(loss.detach()),"elapsed_seconds":time.monotonic()-started});status("training",**history[-1])
            if step%500==0:
                model.eval();predictions=[]
                with torch.inference_mode():
                    for offset in range(0,len(val),64):
                        selected=val[offset:offset+64]
                        with torch.autocast("cuda",dtype=torch.bfloat16):prediction=model(dense[selected].to(device),ego[selected].to(device))
                        predictions.append(decode_poses(prediction).cpu())
                predictions=torch.cat(predictions).numpy();assert np.isfinite(predictions).all()
                checkpoint=out/f"step-{step:04d}.safetensors";save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint))
                path=out/f"step-{step:04d}-predictions.npz"
                np.savez(path,scene_tokens=np.asarray([index[i]["scene_token"] for i in val]),generation_seeds=np.asarray([index[i]["generation_seed"] for i in val]),
                    trajectories=predictions,teacher_poses=teacher[val].numpy())
                scores=out/f"step-{step:04d}-scores.csv"
                command=["bash","-c",'source "$1/scripts/remote_env.sh" scorer; python "$1/scripts/remote_target_diagnostic_score.py" --predictions "$2" --output "$3"',"diagnostic-score",str(ROOT),str(path),str(scores)]
                with (out/f"step-{step:04d}-score.log").open("w") as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
                checks.append({"step":step,"checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),"scores":str(scores),"score_summary":str(scores.with_suffix(".summary.json"))})
                temporary=out/"last.pt.tmp";torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"protocol_sha256":sha(out/"protocol.json"),
                    "step":step,"order":order,"epoch":epoch,"position":position,"history":history,"checks":checks,"gradients":gradients,
                    "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(device)},temporary);temporary.replace(resume)
        write(out/"summary.json",{"target":args.target,"final_step":2000,"final_check":checks[-1],"checks":checks,"gradient_gate":gradients,
            "parameters":parameter_count(model),"history":history,"protocol_sha256":sha(out/"protocol.json"),"navtest_used":False})
        status("verified",steps=2000)
    except Exception as error:status("failed",error=repr(error));raise
