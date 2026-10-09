"""Recheck complete pilot artifacts before target/control diagnostics."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np

ROOT=Path.home()/"DriveWAM_Lightweight_Action_Readout"
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(16*1024**2),b""):h.update(block)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())

if __name__=="__main__":
    start=time.monotonic();checks={};snapshot=[]
    for name in ["opt_devval256_2026-10-01","heads_online_devval256_2026-10-01-r2"]:
        folder=ROOT/"runs"/name;lock=load(folder/"EVALUATION_LOCK.json")
        for entry in lock["files"]+lock["scripts"]:assert sha(entry["path"])==entry["sha256"],entry["path"]
        entries=list(map(json.loads,(folder/"prediction-index.jsonl").read_text().splitlines()))
        assert len(entries)==(2496 if name.startswith("opt_") else 3584)
        assert len({(r["model"],r["scene_token"],r["generation_seed"]) for r in entries})==len(entries)
        for entry in entries:
            assert sha(entry["prediction_path"])==entry["prediction_sha256"]
            with np.load(entry["prediction_path"],allow_pickle=False) as data:
                poses=data["trajectory"];assert poses.shape==(8,3) and np.isfinite(poses).all()
                assert hashlib.sha256(poses.tobytes()).hexdigest()==entry["trajectory_sha256"]
                assert str(data["scene_token"])==entry["scene_token"]
        checks[name]={"files_scripts_and_predictions_verified":True,"predictions":len(entries)}
        snapshot.append({"path":str(folder/"EVALUATION_LOCK.json"),"sha256":sha(folder/"EVALUATION_LOCK.json")})
    cache=ROOT/"caches/features/world_features_pilot_2026-10-01";lock=load(cache/"FEATURE_LOCK.json");protocol=load(cache/"protocol.json")
    for entry in lock["files"]:assert sha(entry["path"])==entry["sha256"]
    for name,value in protocol["scripts"].items():assert sha(ROOT/"scripts"/name)==value
    index=list(map(json.loads,(cache/"index.jsonl").read_text().splitlines()))
    assert [r["scene_token"] for r in index]==protocol["train_tokens"]+protocol["val_tokens"]
    assert len(index)==5000 and len({r["scene_token"] for r in index})==5000
    for entry in lock["shards"]:
        assert sha(entry["path"])==entry["sha256"]
        with Path(entry["path"]).open("rb") as f:
            size=struct.unpack("<Q",f.read(8))[0];assert size<1024**2;header=json.loads(f.read(size))
        n=len(entry["tokens"])
        assert header["dense"]["shape"]==[n,3,6,112,3072] and header["dense"]["dtype"]=="BF16"
        assert header["mean"]["shape"]==[n,3,6,3072] and header["mean"]["dtype"]=="F32"
        assert header["ego"]["shape"]==[n,20] and header["ego"]["dtype"]=="F32"
    manifest_path=ROOT/"splits/pilot-manifest.jsonl";assert sha(manifest_path)==protocol["manifest_sha256"]
    manifest={r["scene_token"]:r for r in map(json.loads,manifest_path.read_text().splitlines())}
    assert not {manifest[t]["log_name"] for t in protocol["train_tokens"]}&{manifest[t]["log_name"] for t in protocol["val_tokens"]}
    for entry in index:
        row=manifest[entry["scene_token"]];assert row["input_sha256"]==entry["input_sha256"]
        assert sha(row["input_path"])==row["input_sha256"] and sha(row["label_path"])==row["target_sha256"]
    parity=load(cache/"parity.json");assert parity["scenes"]==32
    assert all(r["world_only_features_bitwise_equal"] and r["world_only_latents_bitwise_equal"] and r["hook_raw_action_and_world_bitwise_equal"] and r["current_and_first_early_stop_features_bitwise_equal"] for r in parity["rows"])
    checks["feature_cache"]={"scenes":5000,"shards":len(lock["shards"]),"bytes":lock["dense_cache_bytes"],"full_shard_sha256_rechecked":True,"split_input_target_parity_verified":True}
    head=ROOT/"runs/head_pilot_2026-10-01";overfit=load(head/"overfit.json");assert overfit["passed"]
    assert overfit["metrics"]["ade_4s_m"]<.1 and overfit["metrics"]["fde_4s_m"]<.25
    configs={};selected=[]
    for group in ["probes","fusion"]:
        p=head/(group+"-protocol.json");trained=load(p)
        assert load(head/(group+"-status.json"))["stage"]=="verified"
        assert sha(cache/"FEATURE_LOCK.json")==trained["feature_lock_sha256"]
        assert sha(head/(group+"-stats.safetensors"))==trained["stats_sha256"]
        for name,value in trained["scripts"].items():assert sha(ROOT/"scripts"/name)==value
        configs.update(trained["models"])
        for name,config in trained["models"].items():
            result=load(head/name/"summary.json");assert result["protocol_sha256"]==sha(p)
            assert result["gradient_gate"]["all_parameter_gradients_finite"] and not result["gradient_gate"]["backbone_present"]
            if config["kind"].startswith("sf"):assert all(v>0 for v in result["gradient_gate"]["query_gradient_norms"].values())
            best=result["best_by_pilot_PDMS"];assert sha(best["checkpoint"])==best["checkpoint_sha256"]
            score_summary=load(Path(best["scores_path"]).with_suffix(".summary.json"));assert score_summary["coverage"]==1 and score_summary["failures"]==0
            assert sha(best["prediction_path"])==score_summary["prediction_sha256"] and sha(best["scores_path"])==score_summary["scores_sha256"]
            with np.load(best["prediction_path"],allow_pickle=False) as data:
                assert set(data["scene_tokens"].astype(str))==set(protocol["val_tokens"])
                assert data["trajectories"].shape==(1000,8,3) and np.isfinite(data["trajectories"]).all()
            selected.append({"model":name,"checkpoint":best["checkpoint"],"sha256":best["checkpoint_sha256"]})
    assert len(configs)==12
    online=ROOT/"runs/heads_online_devval256_2026-10-01-r2";parity=load(online/"online-cache-parity.json")
    assert set(parity["rows"])==set(configs)
    assert all(len(rows)==32 and all(r["max_abs_difference"]<=1e-5 for r in rows) for rows in parity["rows"].values())
    checks["head_training_and_native_online"]={"models":12,"overfit_passed":True,"selected_checkpoints_verified":selected,"online_cache_comparisons":384,"native_paths_verified":True}
    report={"status":"complete_pilot_artifacts_rechecked","checks":checks,"evaluation_locks":snapshot,
        "feature_lock_sha256":sha(cache/"FEATURE_LOCK.json"),"navtest_used":False,"B_star_selected":False,
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.monotonic()-start,
        "audit_script_sha256":sha(ROOT/"scripts/remote_pilot_audit.py")}
    path=ROOT/"reports/pilot-artifact-audit-2026-10-01.json";temp=path.with_suffix(".tmp");temp.write_text(json.dumps(report,indent=2)+"\n");temp.replace(path)
    print(json.dumps(report,indent=2))
