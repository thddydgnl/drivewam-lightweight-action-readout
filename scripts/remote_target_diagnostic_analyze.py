"""Paired target/control evidence; hold expansion when pilot quality fails."""
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from remote_pilot_head_score import ROOT,sha
from remote_baseline_analyze import QUALITY,cluster_bootstrap

if __name__=="__main__":
    root=ROOT/"runs/target_diagnostic_2026-10-01";control=ROOT/"runs/feature_control_2026-10-01"
    assert json.loads((control/"score-status.json").read_text())["stage"]=="verified"
    protocol=json.loads((root/"extract-protocol.json").read_text());tokens=protocol["val_tokens"]
    fields=QUALITY+["ade_4s_m","fde_4s_m","imitation_ade_4s_m","imitation_fde_4s_m"]
    frames={};results={};training={};locks=[]
    paths={"B10":root/"B10-reference/reference-scores.csv"}
    for target in ["GT","Teacher"]:
        folder=root/("D-"+target);assert json.loads((folder/"status.json").read_text())["stage"]=="verified"
        summary=json.loads((folder/"summary.json").read_text());p=json.loads((folder/"protocol.json").read_text())
        assert summary["protocol_sha256"]==sha(folder/"protocol.json") and summary["final_step"]==2000
        assert summary["gradient_gate"]["all_parameter_gradients_finite"] and all(v>0 for v in summary["gradient_gate"]["query_gradient_norms"].values())
        check=summary["final_check"];assert check["step"]==2000 and sha(check["checkpoint"])==check["checkpoint_sha256"]
        assert sha(folder/"initial.safetensors")==p["initial_parameters_sha256"] and sha(folder/"stats.safetensors")==p["stats_sha256"]
        paths["D-"+target]=Path(check["scores"]);training[target]=p
        locks.append({"model":"D-"+target,"checkpoint":check["checkpoint"],"sha256":check["checkpoint_sha256"]})
    assert {k:v for k,v in training["GT"].items() if k not in ["target","target_definition"]}=={k:v for k,v in training["Teacher"].items() if k not in ["target","target_definition"]},"Targets must be the only protocol difference"
    for name,path in paths.items():
        report=json.loads(path.with_suffix(".summary.json").read_text());assert report["coverage"]==1 and report["failures"]==0 and report["scenes"]==256
        assert sha(path)==report["scores_sha256"]
        frame=pd.read_csv(path,keep_default_na=False);assert len(frame)==768 and frame.valid.all()
        assert set(zip(frame.scene_token,frame.generation_seed))=={(token,seed) for token in tokens for seed in [0,1,2]}
        frame[QUALITY]*=100
        per_scene=frame.groupby(["scene_token","log_name"])[fields].mean().reset_index().set_index("scene_token").loc[tokens]
        frames[name]=per_scene
        results[name]={"metrics":cluster_bootstrap(per_scene,fields),"quality_by_generation_seed":report["quality_by_generation_seed"],
            "student_within_scene_xy_rms_m":report["student_within_scene_xy_rms_m"],"B10_within_scene_xy_rms_m":report["B10_within_scene_xy_rms_m"],
            "mean_within_scene_pdms_population_std_points":report["mean_within_scene_pdms_population_std_points"]}
    paired={};differences=[]
    for candidate,reference in [("D-Teacher","D-GT"),("D-GT","B10"),("D-Teacher","B10")]:
        a,b=frames[candidate],frames[reference];assert list(a.log_name)==list(b.log_name)
        delta=a[fields]-b[fields];delta["log_name"]=a.log_name
        key=candidate+"_minus_"+reference;paired[key]=cluster_bootstrap(delta,fields)
        delta["comparison"]=key;differences.append(delta.reset_index())
    pd.concat(differences,ignore_index=True).to_csv(root/"paired-differences.csv",index=False)
    swap=json.loads((control/"summary.json").read_text())
    online=json.loads((ROOT/"runs/heads_online_devval256_2026-10-01-r2/summary.json").read_text())
    strong_models=[name for name,screen in online["pilot_screening"].items() if screen["pilot_quality_threshold"] and screen["pilot_speed_target"]]
    decision={"full_scale_expansion":False,"reason":"All pilot direct heads miss the predeclared quality threshold; control and target diagnosis precede any expansion.",
        "qualified_pilot_heads":strong_models,"B_star_selected":False,"navtest_used":False,
        "target_protocol_match_except_target":True,"feature_control":"matched command/speed, different log; no GT pairing",
        "next_review":"Inspect GT/teacher imitation and official scores, optimization and a limited confirmation seed before selecting a bounded follow-up.",
        "terminal_future_dynamics_claim":"not established by swaps or native-phase timing comparisons"}
    summary={"models":results,"paired_differences":paired,"feature_control":swap,"decision":decision,
        "distinct_train_scenes":512,"distinct_validation_scenes":256,"generation_seeds":[0,1,2],"train_seed":17,
        "limitations":["Three noise seeds are averaged per scene, not treated as three independent scenes.",
            "Both diagnostics use 512 training scenes; they do not predict full 92599-scene training results.",
            "Both heads share initialization, inputs, minibatches, optimizer and 2000 steps; only target changes.",
            "B10 is a reference teacher, not a physical oracle; easier or smoother teacher labels can explain gains.",
            "Swapped features can create OOD combinations; degradation does not prove causal future dynamics.",
            "This is internal navtrain validation with one training seed; independent navtest is unused."],
        "generated_at_utc":datetime.now(timezone.utc).isoformat()}
    (root/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    (root/"pilot_decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    table=[]
    for name,value in results.items():table.append({"model":name,**{field:stat["mean"] for field,stat in value["metrics"].items()}})
    pd.DataFrame(table).to_csv(root/"summary.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for index,(name,value) in enumerate(results.items()):
        for ax,field,title in zip(axes,["score","imitation_ade_4s_m"],["Official GT PDMS / scene-averaged seeds","B10 imitation ADE (m)"]):
            stat=value["metrics"][field];ax.bar(index,stat["mean"],color=plt.get_cmap("tab10")(index))
            ax.vlines(index,stat["ci95_low"],stat["ci95_high"],color="black");ax.set(title=title,xticks=range(3),xticklabels=list(results))
    for ax in axes:ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    fig.savefig(root/"target-diagnostic.png",dpi=180);fig.savefig(root/"target-diagnostic.svg");plt.close(fig)
    files=["extract-protocol.json","EXTRACT_LOCK.json","index.jsonl","summary.json","summary.csv","paired-differences.csv","pilot_decision.json","target-diagnostic.png","target-diagnostic.svg"]
    lock={"status":"paired_GT_B10_target_and_feature_control_diagnostics_verified","files":[{"path":str(root/name),"sha256":sha(root/name)} for name in files],
        "final_checkpoints":locks,"CONTROL_LOCK_sha256":sha(control/"CONTROL_LOCK.json"),
        "scripts":[{"path":str(ROOT/"scripts"/name),"sha256":sha(ROOT/"scripts"/name)} for name in ["remote_target_diagnostic_extract.py","remote_target_reference_prepare.py","remote_target_diagnostic_train.py","remote_target_diagnostic_score.py","remote_target_diagnostic_analyze.py","remote_feature_control.py"]],
        "navtest_used":False,"B_star_selected":False}
    (root/"DIAGNOSTIC_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")
    (root/"analysis-status.json").write_text(json.dumps({"stage":"verified","summary":str(root/"summary.json")},indent=2)+"\n")
    print(pd.DataFrame(table).to_string(index=False))
