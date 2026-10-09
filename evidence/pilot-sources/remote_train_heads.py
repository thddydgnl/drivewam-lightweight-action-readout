"""Frozen-feature pilot heads, gradient/overfit gates and official validation.

The backbone is absent from this training process. Inputs are verified feature
cache rows; targets are loaded through a separate, explicit training path.
"""
import argparse
from datetime import datetime, timezone
import hashlib
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
from lightweight_heads import build_head, parameter_count, matched_single_layer_width, normalize_deltas, decode_poses, action_loss

ROOT=Path.home()/"DriveWAM_Lightweight_Action_Readout"
PHASES={"current":0,"first":1,"terminal":2}
CONFIGS={"ego-only":{"kind":"ego","phase":"current"},
         **{f"linear-{phase}":{"kind":"linear","phase":phase} for phase in PHASES},
         **{f"FMLP-{phase}":{"kind":"fmlp","phase":phase} for phase in PHASES},
         "SF1-terminal":{"kind":"sf1","phase":"terminal"},
         "SF1-wide-terminal":{"kind":"sf1-wide","phase":"terminal"},
         **{f"SF3-{phase}":{"kind":"sf3","phase":phase} for phase in PHASES}}

def sha(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(16*1024**2),b""):result.update(block)
    return result.hexdigest()
def write(path,data):
    temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n");temp.replace(path)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def load_dataset(cache,need_dense):
    lock=json.loads((cache/"FEATURE_LOCK.json").read_text())
    assert lock["status"]=="pilot_current_first_terminal_dense_features_verified"
    protocol=json.loads((cache/"protocol.json").read_text())
    for record in lock["files"]:assert sha(record["path"])==record["sha256"]
    index=[json.loads(line) for line in (cache/"index.jsonl").read_text().splitlines()]
    assert [r["scene_token"] for r in index]==protocol["train_tokens"]+protocol["val_tokens"]
    mean,ego,dense=[],[],[]
    for record in lock["shards"]:
        assert sha(record["path"])==record["sha256"]
        with safe_open(record["path"],framework="pt",device="cpu") as stored:
            mean.append(stored.get_tensor("mean")[:,:,5].contiguous())
            ego.append(stored.get_tensor("ego").clone())
            if need_dense:dense.append(stored.get_tensor("dense").index_select(2,torch.tensor([1,3,5])))
    data={"mean":torch.cat(mean),"ego":torch.cat(ego)}
    if need_dense:data["dense"]=torch.cat(dense)
    manifest={r["scene_token"]:r for r in (json.loads(line) for line in (ROOT/"splits/pilot-manifest.jsonl").read_text().splitlines())}
    assert sha(ROOT/"splits/pilot-manifest.jsonl")==protocol["manifest_sha256"]
    poses,deltas=[],[]
    for record in index:
        row=manifest[record["scene_token"]]
        assert row["split"]==record["split"] and row["input_sha256"]==record["input_sha256"]
        assert sha(row["label_path"])==row["target_sha256"]
        with np.load(row["label_path"],allow_pickle=False) as stored:
            poses.append(torch.from_numpy(stored["future_poses_4s"].copy()))
            deltas.append(torch.from_numpy(stored["action_deltas_4s"].copy()))
    data["poses"]=torch.stack(poses);data["target"]=normalize_deltas(torch.stack(deltas))
    assert torch.max(torch.abs(decode_poses(data["target"])-data["poses"]))<1e-4
    for tensor in data.values():assert torch.isfinite(tensor).all() and not tensor.requires_grad
    stats={"ego_mean":data["ego"][:4000].mean(0),"ego_std":data["ego"][:4000].std(0,unbiased=False).clamp_min(1e-5),
           "world_mean":data["mean"][:4000].mean(0),"world_std":data["mean"][:4000].std(0,unbiased=False).clamp_min(1e-5)}
    return data,index,stats,protocol,lock

def batch(data,stats,config,indices,device):
    phase=PHASES[config["phase"]]
    kind=config["kind"]
    ego=(data["ego"].index_select(0,indices)-stats["ego_mean"])/stats["ego_std"]
    if kind in ["sf1","sf1-wide","sf3"]:
        world=data["dense"][:,phase]
        if kind in ["sf1","sf1-wide"]:world=world[:,-1:]
        world=world.index_select(0,indices)
    else:
        world=data["mean"][:,phase].index_select(0,indices)
        if kind=="linear":world=(world-stats["world_mean"][phase])/stats["world_std"][phase]
    return world.to(device),ego.to(device),data["poses"].index_select(0,indices).to(device),data["target"].index_select(0,indices).to(device)

def evaluate(model,data,stats,config,indices,device):
    model.eval();predictions=[]
    with torch.inference_mode():
        for offset in range(0,len(indices),64):
            world,ego,_,_=batch(data,stats,config,indices[offset:offset+64],device)
            with torch.autocast("cuda",dtype=torch.bfloat16):normalized=model(world,ego)
            poses=decode_poses(normalized)
            assert poses.shape==(len(world),8,3) and torch.isfinite(poses).all()
            predictions.append(poses.cpu())
    predictions=torch.cat(predictions)
    target=data["poses"].index_select(0,indices)
    distance=torch.linalg.vector_norm(predictions[...,:2]-target[...,:2],dim=-1)
    return predictions,{"ade_4s_m":float(distance.mean()),"fde_4s_m":float(distance[:,-1].mean())}

def gradient_gate(model):
    parameters=list(model.named_parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in parameters)
    assert sum(float(p.grad.float().square().sum()) for _,p in parameters)>0
    queries={name:float(p.grad.float().norm()) for name,p in parameters if name.endswith(".query")}
    assert all(value>0 for value in queries.values()),queries
    return {"all_parameter_gradients_finite":True,"query_gradient_norms":queries,"backbone_present":False}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--cache-id",default="world_features_pilot_2026-10-01")
    parser.add_argument("--run-id",default="head_pilot_2026-10-01");parser.add_argument("--group",choices=["probes","fusion"],required=True)
    parser.add_argument("--seed",type=int,default=17);args=parser.parse_args()
    out=ROOT/"runs"/args.run_id;out.mkdir(parents=True,exist_ok=True)
    status_path=out/(args.group+"-status.json")
    def status(stage,**extra):
        write(status_path,{"stage":stage,"updated_at_utc":datetime.now(timezone.utc).isoformat(),**extra})
        print(json.dumps({"stage":stage,**extra}),flush=True)
    try:
        assert torch.cuda.device_count()==1
        device=torch.device("cuda:0");torch.set_num_threads(8)
        cache=ROOT/"caches/features"/args.cache_id
        names=[name for name,c in CONFIGS.items() if (c["kind"] in ["sf1","sf1-wide","sf3"])==(args.group=="fusion")]
        status("loading_verified_features",group=args.group,models=names)
        data,index,stats,cache_protocol,cache_lock=load_dataset(cache,args.group=="fusion")
        stats_path=out/(args.group+"-stats.safetensors")
        if not stats_path.exists():save_file({k:v.contiguous() for k,v in stats.items()},str(stats_path))
        selected_width=matched_single_layer_width()
        seed_all(args.seed)
        if args.group=="fusion":
            sf3=build_head("sf3");wide=build_head("sf1-wide")
            count3,countwide=parameter_count(sf3),parameter_count(wide)
            assert abs(count3-countwide)/count3<=.05
            del sf3,wide
        else:count3=countwide=None
        prototype={"group":args.group,"models":{name:CONFIGS[name] for name in names},"train_seed":args.seed,
            "cache_protocol_sha256":sha(cache/"protocol.json"),"feature_lock_sha256":sha(cache/"FEATURE_LOCK.json"),
            "stats_sha256":sha(stats_path),
            "train_scenes":4000,"val_scenes":1000,"generation_seed":0,"backbone_trained":False,
            "optimizer":"AdamW","learning_rate":3e-4,"weight_decay":.01,"betas":[.9,.999],
            "batch_size":32,"max_epochs":30,"early_stopping_no_ADE_improvement_validations":5,
            "minibatch_order":"isolated CPU torch.Generator seed=train_seed+100003*epoch; identical across architectures",
            "warmup_fraction":.05,"gradient_clip_norm":1.,"dropout":0.,"amp":"bfloat16","parameter_dtype":"float32",
            "loss":"position Huber(delta=1m) + 0.2 circular yaw + 0.1 normalized-delta Huber",
            "validation":"ADE/FDE on all 1000 each epoch; official PDMS on all 1000 every 5 epochs and last epoch",
            "selection":"pilot1000 PDMS, then ADE, then epoch; full head-val selection remains pending",
            "wide_projection_dim":selected_width,"SF3_parameters":count3,"SF1_wide_parameters":countwide,
            "ego_standardization":"fit on head-train only","linear_world_standardization":"fit on head-train only",
            "world_feature_width":3072,"selected_layer_depths_one_based":[10,20,30],
            "navtest_used":False,"scripts":{name:sha(ROOT/"scripts"/name) for name in ["lightweight_heads.py","remote_train_heads.py","remote_pilot_head_score.py"]}}
        protocol_path=out/(args.group+"-protocol.json")
        if protocol_path.exists():assert json.loads(protocol_path.read_text())==prototype,"Resume protocol mismatch"
        else:write(protocol_path,prototype)
        if args.group=="probes" and not (out/"overfit.json").exists():
            status("overfit_sanity",steps=0,total=2000)
            seed_all(args.seed);model=build_head("fmlp").to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0)
            smoke=(ROOT/"splits/smoke32-tokens.txt").read_text().splitlines();lookup={row["scene_token"]:i for i,row in enumerate(index)}
            indices=torch.tensor([lookup[t] for t in smoke]);config={"kind":"fmlp","phase":"terminal"};history=[]
            for step in range(1,2001):
                model.train();world,ego,poses,target=batch(data,stats,config,indices,device);optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):output=model(world,ego);loss,_=action_loss(output,poses,target)
                assert torch.isfinite(loss);loss.backward()
                if step==1:gradients=gradient_gate(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                if step%100==0:
                    _,metrics=evaluate(model,data,stats,config,indices,device);history.append({"step":step,**metrics})
                    status("overfit_sanity",step=step,**metrics)
                    if metrics["ade_4s_m"]<.1 and metrics["fde_4s_m"]<.25:break
            assert metrics["ade_4s_m"]<.1 and metrics["fde_4s_m"]<.25,"32-scene overfit failed; inspect before scaling"
            write(out/"overfit.json",{"passed":True,"scenes":32,"steps":step,"metrics":metrics,"history":history,"gradient_gate":gradients})
            save_file({name:t.detach().cpu().contiguous() for name,t in model.state_dict().items()},str(out/"overfit.safetensors"))
            del model,optimizer;torch.cuda.empty_cache()
        train_indices=torch.arange(4000);val_indices=torch.arange(4000,5000)
        results={}
        for name in names:
            folder=out/name;folder.mkdir(exist_ok=True)
            if (folder/"summary.json").exists():
                previous=json.loads((folder/"summary.json").read_text());assert previous["protocol_sha256"]==sha(protocol_path)
                results[name]=previous;continue
            config=CONFIGS[name];seed_all(args.seed);model=build_head(config["kind"]).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01,betas=(.9,.999))
            total_steps=30*math.ceil(4000/32);warmup=int(total_steps*.05);step=0;history=[];score_checks=[]
            best_ade=float("inf");bad_epochs=0;start_epoch=1;gradient_report=None
            resume=folder/"last.pt"
            if resume.exists():
                saved=torch.load(resume,map_location="cpu",weights_only=False)
                assert saved["protocol_sha256"]==sha(protocol_path)
                model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
                for state in optimizer.state.values():
                    for key,value in state.items():
                        if isinstance(value,torch.Tensor):state[key]=value.to(device)
                step=saved["step"];start_epoch=saved["epoch"]+1;history=saved["history"];score_checks=saved["score_checks"]
                best_ade=saved["best_ade"];bad_epochs=saved["bad_epochs"];gradient_report=saved["gradient_report"]
                random.setstate(saved["python_rng"]);np.random.set_state(saved["numpy_rng"]);torch.set_rng_state(saved["torch_rng"])
                torch.cuda.set_rng_state(saved["cuda_rng"],device)
            start=time.monotonic()
            for epoch in range(start_epoch,31):
                model.train();order=train_indices[torch.randperm(4000,generator=torch.Generator().manual_seed(args.seed+100003*epoch))];loss_sum=0.
                for offset in range(0,4000,32):
                    selected=order[offset:offset+32];world,ego,poses,target=batch(data,stats,config,selected,device)
                    step+=1
                    factor=step/max(1,warmup) if step<=warmup else .5*(1+math.cos(math.pi*(step-warmup)/(total_steps-warmup)))
                    for group in optimizer.param_groups:group["lr"]=3e-4*factor
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda",dtype=torch.bfloat16):output=model(world,ego);loss,_=action_loss(output,poses,target)
                    assert torch.isfinite(loss);loss.backward()
                    if gradient_report is None:gradient_report=gradient_gate(model)
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
                    optimizer.step();loss_sum+=float(loss.detach())*len(selected)
                predictions,metrics=evaluate(model,data,stats,config,val_indices,device)
                if metrics["ade_4s_m"]<best_ade-1e-6:best_ade=metrics["ade_4s_m"];bad_epochs=0
                else:bad_epochs+=1
                record={"epoch":epoch,"step":step,"train_loss":loss_sum/4000,**metrics}
                history.append(record);final_epoch=(epoch==30 or (epoch>=5 and bad_epochs>=5))
                weights=folder/f"epoch-{epoch:03d}.safetensors"
                save_file({key:value.detach().cpu().contiguous() for key,value in model.state_dict().items()},str(weights))
                if epoch%5==0 or final_epoch:
                    prediction_path=folder/f"epoch-{epoch:03d}-predictions.npz"
                    np.savez(prediction_path,scene_tokens=np.asarray([row["scene_token"] for row in index[4000:]]),trajectories=predictions.numpy())
                    score_path=folder/f"epoch-{epoch:03d}-scores.csv"
                    command=["bash","-c",'source "$1/scripts/remote_env.sh" scorer; python "$1/scripts/remote_pilot_head_score.py" --predictions "$2" --output "$3"',"head-official-score",str(ROOT),str(prediction_path),str(score_path)]
                    subprocess.run(command,check=True,stdout=(folder/f"epoch-{epoch:03d}-score.log").open("w"),stderr=subprocess.STDOUT)
                    score_report=json.loads(score_path.with_suffix(".summary.json").read_text())
                    assert score_report["coverage"]==1 and score_report["failures"]==0
                    check={"epoch":epoch,"checkpoint":str(weights),"checkpoint_sha256":sha(weights),"prediction_path":str(prediction_path),
                        "scores_path":str(score_path),"pilot1000_pdms":score_report["pilot1000"]["score"],
                        "dev256_pdms":score_report["dev256"]["score"],**metrics}
                    score_checks.append(check);record.update(pilot1000_pdms=check["pilot1000_pdms"],dev256_pdms=check["dev256_pdms"])
                saved={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"step":step,"history":history,
                    "score_checks":score_checks,"best_ade":best_ade,"bad_epochs":bad_epochs,"gradient_report":gradient_report,
                    "protocol_sha256":sha(protocol_path),"config":config,"stats":stats,"python_rng":random.getstate(),
                    "numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(device)}
                temporary=folder/"last.pt.tmp";torch.save(saved,temporary);temporary.replace(resume)
                write(folder/"history.json",history)
                status("training",group=args.group,model=name,epoch=epoch,elapsed_seconds=time.monotonic()-start,**{k:v for k,v in record.items() if k!="epoch"})
                if final_epoch:break
            assert score_checks
            best=min(score_checks,key=lambda r:(-r["pilot1000_pdms"],r["ade_4s_m"],r["epoch"]))
            summary={"model":name,"config":config,"parameters":parameter_count(model),"train_seed":args.seed,
                "protocol_sha256":sha(protocol_path),"epochs":len(history),"steps":step,"elapsed_seconds":time.monotonic()-start,
                "best_by_pilot_PDMS":best,"score_checks":score_checks,"gradient_gate":gradient_report,"navtest_used":False}
            write(folder/"summary.json",summary);results[name]=summary
            del model,optimizer,saved;torch.cuda.empty_cache()
        write(out/(args.group+"-summary.json"),{"models":results,"protocol_sha256":sha(protocol_path),"navtest_used":False})
        status("verified",group=args.group,models=len(results),backbone_trained=False)
    except Exception as error:
        status("failed",error=repr(error))
        raise
