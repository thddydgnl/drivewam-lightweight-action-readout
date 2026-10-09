"""Native batch-one head policies, cache parity and isolated dev256 timing.

No labels are opened. One head is resident at a time. Ego-only is measured
before loading DriveWAM; the world policies rotate order across three passes.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from safetensors import safe_open
from baseline_common import ROOT, file_sha, load_policy, sample_seed
from head_policy_runtime import OnlineHeadPolicy
from navsim_action_opt import make_skip_terminal_action
from navsim_current_dataset import load_policy_input, build_rollout_batch
from remote_train_heads import CONFIGS


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def telemetry():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu,temperature.gpu,clocks.current.sm,power.draw", "--format=csv,noheader"], text=True).strip()


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


class PhaseEvents:
    """Record phase completion even when native early exits raise FeatureStop."""
    def __init__(self, policy):
        self.policy = policy
        self.events, self.originals = [], {}
        for method, phase in {"_encode_images":"vae_encode", "_encode_prompt":"prompt_cache",
                              "_prime_rollout_cache":"primer", "_rollout_video_frame":"world",
                              "_rollout_action_chunk":"action"}.items():
            original = getattr(policy, method)
            self.originals[method] = original
            def wrapper(*args, _original=original, _phase=phase, **kwargs):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                try:
                    return _original(*args, **kwargs)
                finally:
                    end.record()
                    self.events.append((_phase, start, end))
            setattr(policy, method, wrapper)
    def reset(self): self.events.clear()
    def phases(self):
        values = {}
        for phase, start, end in self.events:
            key = phase + "_gpu_ms"
            values[key] = values.get(key, 0.) + start.elapsed_time(end)
        return values
    def close(self):
        for method, original in self.originals.items(): setattr(self.policy, method, original)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="heads_online_devval256_2026-10-01")
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    assert not out.exists(), f"Refusing to overwrite {out}"
    out.mkdir(parents=True)
    def status(stage, **values):
        record = {"stage":stage, "updated_at_utc":datetime.now(timezone.utc).isoformat(), **values}
        write(out / "inference-status.json", record)
        print(json.dumps(record), flush=True)
    try:
        torch.set_num_threads(8)
        assert torch.cuda.device_count() == 1
        torch.cuda.set_device(0)
        head_root = ROOT / "runs/head_pilot_2026-10-01"
        cache = ROOT / "caches/features/world_features_pilot_2026-10-01"
        opt_run = ROOT / "runs/opt_devval256_2026-10-01"
        for group in ["probes", "fusion"]:
            assert json.loads((head_root / (group+"-status.json")).read_text())["stage"] == "verified"
        assert json.loads((head_root / "overfit.json").read_text())["passed"]
        assert json.loads((opt_run / "analysis-status.json").read_text())["stage"] == "verified"
        opt_summary = json.loads((opt_run / "summary.json").read_text())
        opt_name = opt_summary["provisional_fastest_eligible_opt"]
        assert opt_name is not None
        steps = json.loads((opt_run / "protocol.json").read_text())["action_steps"][opt_name]
        tokens_path = ROOT / "splits/devval256-tokens.txt"
        tokens = tokens_path.read_text().splitlines()
        assert len(tokens) == len(set(tokens)) == 256
        manifest = ROOT / "splits/pilot-manifest.jsonl"
        rows = {r["scene_token"]:r for r in map(json.loads, manifest.read_text().splitlines())}
        index = {r["scene_token"]:r for r in map(json.loads, (cache / "index.jsonl").read_text().splitlines())}
        feature_protocol = json.loads((cache / "protocol.json").read_text())
        assert file_sha(manifest) == feature_protocol["manifest_sha256"]
        source = ROOT / "work/DriveWAM"
        assert subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip() == feature_protocol["source_commit"]
        assert subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True) == ""
        for group in ["probes", "fusion"]:
            trained_protocol = json.loads((head_root/(group+"-protocol.json")).read_text())
            assert file_sha(ROOT/"scripts/lightweight_heads.py") == trained_protocol["scripts"]["lightweight_heads.py"]
            assert file_sha(cache/"FEATURE_LOCK.json") == trained_protocol["feature_lock_sha256"]
        status("preloading_verified_observations_and_parity_features")
        inputs, parity_cache = {}, {}
        for token in tokens:
            row = rows[token]
            assert row["split"] == "head_val" and file_sha(row["input_path"]) == row["input_sha256"]
            inputs[token] = load_policy_input(row["input_path"])
        for token in tokens[:32]:
            entry = index[token]
            assert entry["generation_seed"] == 0 and entry["input_sha256"] == rows[token]["input_sha256"]
            with safe_open(entry["shard_path"], framework="pt", device="cpu") as stored:
                position = entry["row_index"]
                mean = stored.get_slice("mean")[position:position+1][0,:,5].clone()
                dense = stored.get_slice("dense")[position:position+1][0].index_select(1,torch.tensor([1,3,5]))
                ego = stored.get_slice("ego")[position:position+1][0].clone()
            parity_cache[token] = mean, dense, ego
        summaries = {name:json.loads((head_root/name/"summary.json").read_text()) for name in CONFIGS}
        models = list(CONFIGS) + ["B10", opt_name]
        protocol = {"run_id":args.run_id,"scope":"navtrain internal pilot dev256; checkpoint selection used pilot1000",
            "tokens":tokens,"model_names":models,"expected_prediction_count":len(models)*256,
            "timing_repeats":3,"warmup_per_model_per_repeat":20,"batch_size":1,
            "timing_scope":"preloaded CPU observation to usable CPU trajectory; includes native VAE/world/head and excludes file I/O/scoring/weight loading",
            "one_head_resident_at_a_time":True,"ego_before_backbone_loading":True,
            "model_order":"three rotated/reversed passes for world policies; ego-only measured first without backbone",
            "native_forward_calls":{"current":1,"first":3,"terminal":6,"ego":0},
            "manifest_sha256":file_sha(manifest),"tokens_sha256":file_sha(tokens_path),
            "feature_lock_sha256":file_sha(cache/"FEATURE_LOCK.json"),
            "head_group_protocol_sha256":{group:file_sha(head_root/(group+"-protocol.json")) for group in ["probes","fusion"]},
            "selected_checkpoints":{name:summary["best_by_pilot_PDMS"] for name,summary in summaries.items()},
            "fresh_baseline_controls":{"B10":10,opt_name:steps},"provisional_opt":opt_name,"B_star_selected":False,
            "phase_gpu_events":True,"telemetry_before":telemetry(),"gpu":torch.cuda.get_device_name(0),
            "torch":torch.__version__,"cuda_runtime":torch.version.cuda,"dtype":"backbone BF16, head FP32 with BF16 autocast, decode FP32",
            "navtest_used":False,"scripts":{name:file_sha(ROOT/"scripts"/name) for name in [
                "remote_head_online_infer.py","head_policy_runtime.py","world_feature_tap.py","lightweight_heads.py",
                "baseline_common.py","navsim_action_opt.py","navsim_current_dataset.py"]}}
        write(out/"protocol.json",protocol)
        hashes, parity, timing_count = {}, {}, {name:0 for name in models}
        previous = {(r["model"],r["scene_token"]):r for r in map(json.loads,(opt_run/"prediction-index.jsonl").read_text().splitlines()) if r["generation_seed"]==0}
        world_policy, phases = None, None
        with (out/"timings.jsonl").open("w") as timings, (out/"prediction-index.jsonl").open("w") as predictions:
            def save(token,name,trajectory,record,repeat):
                trajectory_hash=digest(trajectory)
                key=name,token
                if key in hashes: assert hashes[key]==trajectory_hash, "Same-seed repeat changed"
                else: hashes[key]=trajectory_hash
                record.update(model=name,scene_token=token,generation_seed=0,sample_seed=sample_seed(token,0),
                              log_name=rows[token]["log_name"],repeat=repeat,trajectory_sha256=trajectory_hash)
                timings.write(json.dumps(record)+"\n");timings.flush();timing_count[name]+=1
                if repeat==0:
                    folder=out/"predictions"/name/"generation_seed_0";folder.mkdir(parents=True,exist_ok=True)
                    path=folder/(token+".npz");np.savez(path,trajectory=trajectory,scene_token=np.asarray(token))
                    predictions.write(json.dumps({**record,"prediction_path":str(path),"prediction_sha256":file_sha(path)})+"\n");predictions.flush()

            def head_run(policy,token):
                if phases: phases.reset()
                policy.record_head_events=True
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter()
                trajectory,calls=policy.predict(inputs[token],sample_seed(token,0))
                torch.cuda.synchronize();elapsed=(time.perf_counter()-started)*1000
                assert trajectory.shape==(8,3) and np.isfinite(trajectory).all()
                assert calls==(0 if policy.kind=="ego" else [1,3,6][policy.phase])
                values={"policy_ms":elapsed,"actual_forward_calls":calls,
                    "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                    "head_gpu_ms":policy.head_events[0].elapsed_time(policy.head_events[1]),
                    "world_latents_sha256":"native_phase_feature_readout"}
                if phases: values.update(phases.phases())
                return trajectory,values

            # Ego control is measured without a resident backbone.
            ego=OnlineHeadPolicy(head_root/"ego-only")
            parity["ego-only"]=[]
            for token in tokens[:32]:
                online,_=head_run(ego,token)
                mean,dense,cached_ego=parity_cache[token]
                cached=ego.predict_cached(mean[0],dense[0],cached_ego)
                error=float(np.max(np.abs(online-cached)))
                assert error<=1e-5
                parity["ego-only"].append({"scene_token":token,"max_abs_difference":error,"bitwise_equal":digest(online)==digest(cached)})
            for repeat in range(3):
                for token in tokens[:20]:head_run(ego,token)
                for number,token in enumerate(tokens):
                    trajectory,record=head_run(ego,token);save(token,"ego-only",trajectory,record,repeat)
                write(out/f"telemetry-ego-repeat-{repeat}.json",{"telemetry":telemetry()})
            ego.close();del ego;gc.collect();torch.cuda.empty_cache()
            status("loading_frozen_backbone")
            weight=ROOT/"checkpoints/drivewam/navsim/diffusion_pytorch_model.safetensors"
            assert file_sha(weight)==feature_protocol["checkpoint_sha256"]
            world_policy,loading=load_policy()
            raw_action,opt_action,metadata=make_skip_terminal_action(world_policy)
            world_policy._skip_terminal_action_cache=False
            def dispatch(*args,**kwargs):
                return (opt_action if world_policy._skip_terminal_action_cache else raw_action)(*args,**kwargs)
            world_policy._rollout_action_chunk=dispatch
            phases=PhaseEvents(world_policy)
            call_list=[]
            handle=world_policy.transformer.register_forward_pre_hook(lambda m,a,k:call_list.append(bool(k.get("action_mode",False))),with_kwargs=True)
            def baseline_run(name,token):
                world_policy._skip_terminal_action_cache=name.endswith("-opt")
                phases.reset();call_list.clear();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                batch=build_rollout_batch(inputs[token],world_policy.config)
                with torch.inference_mode():
                    actions,latents=world_policy.predict(batch,3,10 if name=="B10" else steps,sample_seed(token,0),video_denoise_fraction=.6)
                    trajectory=world_policy.pred_traj_to_poses(world_policy.actions_to_trajectory(actions))
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                assert digest(trajectory)==previous[name,token]["trajectory_sha256"]
                assert len(call_list)==(17 if name=="B10" else steps+6)
                return trajectory,{"policy_ms":elapsed,"actual_forward_calls":len(call_list),
                    "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                    "world_latents_sha256":previous[name,token]["world_latents_sha256"],**phases.phases()}
            order=[name for name in models if name!="ego-only"]
            for repeat in range(3):
                rotated=order[repeat:]+order[:repeat]
                if repeat%2:rotated.reverse()
                for name in rotated:
                    status("native_online_timing",model=name,repeat=repeat,completed=0,total=256)
                    policy=None if name in ["B10",opt_name] else OnlineHeadPolicy(head_root/name,world_policy)
                    if policy and repeat==0:
                        comparisons=[]
                        for token in tokens[:32]:
                            policy.record_head_events=False
                            online,_=policy.predict(inputs[token],sample_seed(token,0))
                            mean,dense,ego=parity_cache[token]
                            cached=policy.predict_cached(mean[policy.phase],dense[policy.phase],ego)
                            error=float(np.max(np.abs(online-cached)))
                            assert error<=1e-5, (name,token,error,"online/cache parity")
                            comparisons.append({"scene_token":token,"max_abs_difference":error,"bitwise_equal":digest(online)==digest(cached)})
                        timed,_=head_run(policy,tokens[0])
                        phases.close();policy.record_head_events=False
                        uninstrumented,_=policy.predict(inputs[tokens[0]],sample_seed(tokens[0],0))
                        assert digest(timed)==digest(uninstrumented), "Instrumentation changes prediction"
                        phases=PhaseEvents(world_policy)
                        parity[name]=comparisons
                    runner=(lambda token:baseline_run(name,token)) if policy is None else (lambda token:head_run(policy,token))
                    for token in tokens[:20]:runner(token)
                    for number,token in enumerate(tokens):
                        trajectory,record=runner(token);save(token,name,trajectory,record,repeat)
                        if (number+1)%32==0:status("native_online_timing",model=name,repeat=repeat,completed=number+1,total=256)
                    write(out/f"telemetry-{name}-repeat-{repeat}.json",{"telemetry":telemetry()})
                    if policy:policy.close();del policy
                    gc.collect();torch.cuda.empty_cache()
            handle.remove();phases.close()
        write(out/"online-cache-parity.json",{"scenes_each_head":32,"max_abs_tolerance":1e-5,"rows":parity})
        assert all(value==768 for value in timing_count.values())
        records=[json.loads(line) for line in (out/"prediction-index.jsonl").read_text().splitlines()]
        assert len(records)==protocol["expected_prediction_count"]
        write(out/"inference-summary.json",{"coverage_each_model":1.,"predictions":len(records),"timing_count":timing_count,
            "repeated_predictions_bitwise_equal":True,"all_outputs_finite":True,"online_cache_parity_passed":True,
            "prediction_index_sha256":file_sha(out/"prediction-index.jsonl"),"timings_sha256":file_sha(out/"timings.jsonl"),
            "telemetry_after":telemetry(),"completed_at_utc":datetime.now(timezone.utc).isoformat()})
        status("verified",predictions=len(records))
    except Exception as exc:
        status("failed",error=repr(exc))
        raise
