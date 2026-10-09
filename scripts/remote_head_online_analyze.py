"""Paired pilot head evidence; no automatic full-scale expansion decision."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from remote_baseline_analyze import QUALITY, ERRORS, cluster_bootstrap, distribution

ROOT=Path.home()/"DriveWAM_Lightweight_Action_Readout"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--run-id",default="heads_online_devval256_2026-10-01");args=parser.parse_args()
    out=ROOT/"runs"/args.run_id;head_root=ROOT/"runs/head_pilot_2026-10-01"
    for stage in ["inference","score-official","score-vendored"]:
        assert json.loads((out/(stage+"-status.json")).read_text())["stage"]=="verified"
    protocol=json.loads((out/"protocol.json").read_text());tokens=protocol["tokens"];models=protocol["model_names"]
    score=pd.read_csv(out/"scores-official.csv",keep_default_na=False)
    vendored=pd.read_csv(out/"scores-vendored.csv",keep_default_na=False)
    assert len(score)==len(vendored)==protocol["expected_prediction_count"] and score.valid.all() and vendored.valid.all()
    joined=score.merge(vendored,on=["model","scene_token","generation_seed"],suffixes=("_official","_vendored"),validate="one_to_one")
    assert len(joined)==len(score)
    for field in ["prediction_sha256","target_sha256","metric_cache_sha256"]:
        assert (joined[field+"_official"]==joined[field+"_vendored"]).all()
    scorer_difference={key:float(((joined[key+"_official"]-joined[key+"_vendored"])*100).abs().max()) for key in QUALITY}
    assert max(scorer_difference.values())<1e-6
    old=ROOT/"runs/opt_devval256_2026-10-01"
    cv=pd.read_csv(old/"scores-official.csv",keep_default_na=False);cv=cv[cv.model=="CV"]
    assert len(cv)==256
    score=pd.concat([score,cv],ignore_index=True);score[QUALITY]*=100
    timing=pd.DataFrame(map(json.loads,(out/"timings.jsonl").read_text().splitlines()))
    results={};table=[]
    for name in models+["CV"]:
        quality=score[score.model==name].set_index("scene_token").loc[tokens].reset_index()
        assert len(quality)==256 and quality.scene_token.is_unique
        result={"quality":cluster_bootstrap(quality,QUALITY+ERRORS),"scenes":256,"logs":int(quality.log_name.nunique()),
            "at_fault_collision_scenes":int(quality.at_fault_collision.sum()),"dac_failure_scenes":int((quality.drivable_area_compliance<100).sum()),
            "ttc_infraction_scenes":int(quality.ttc_infraction.sum())}
        if name!="CV":
            measured=timing[timing.model==name];assert len(measured)==768
            by_repeat={}
            for repeat in range(3):
                subset=measured[measured["repeat"]==repeat];assert len(subset)==256 and set(subset.scene_token)==set(tokens)
                by_repeat[str(repeat)]=distribution(subset.policy_ms)
            medians=[v["p50"] for v in by_repeat.values()]
            result.update(policy_ms=distribution(measured.policy_ms),policy_ms_by_repeat=by_repeat,
                repeat_p50_relative_range=float((max(medians)-min(medians))/np.median(medians)),
                gpu_phase_ms={field:distribution(measured[field].dropna()) for field in measured.columns if field.endswith("_gpu_ms") and measured[field].notna().any()},
                peak_allocated_gib=float(measured.peak_allocated_bytes.max()/1024**3),
                peak_reserved_gib=float(measured.peak_reserved_bytes.max()/1024**3),
                forward_calls=sorted(measured.actual_forward_calls.unique().astype(int).tolist()))
        else:result.update(reference_only=True,source_run=str(old),policy_ms=json.loads((old/"summary.json").read_text())["models"]["CV"]["policy_ms"])
        if name not in ["B10",protocol["provisional_opt"],"CV"]:
            trained=json.loads((head_root/name/"summary.json").read_text())
            result.update(parameters=trained["parameters"],train_seed=trained["train_seed"],selected_epoch=trained["best_by_pilot_PDMS"]["epoch"],
                pilot1000_selection_pdms=trained["best_by_pilot_PDMS"]["pilot1000_pdms"])
        results[name]=result
        table.append({"model":name,**{key:value["mean"] for key,value in result["quality"].items()},
            "policy_p50_ms":result["policy_ms"]["p50"],"policy_p95_ms":result["policy_ms"]["p95"],
            "peak_allocated_gib":result.get("peak_allocated_gib",0),"head_parameters":result.get("parameters",0)})
    references=["B10",protocol["provisional_opt"],"ego-only","CV","SF3-current","SF3-first","SF1-wide-terminal","FMLP-terminal"]
    paired={};differences=[]
    for reference in references:
        base=score[score.model==reference].set_index("scene_token").loc[tokens]
        paired[reference]={}
        for name in models:
            if name==reference:continue
            candidate=score[score.model==name].set_index("scene_token").loc[tokens]
            assert list(candidate.log_name)==list(base.log_name)
            delta=candidate[QUALITY+ERRORS]-base[QUALITY+ERRORS];delta["log_name"]=base.log_name
            paired[reference][name]=cluster_bootstrap(delta,QUALITY+ERRORS)
            delta["model"]=name;delta["reference"]=reference;differences.append(delta.reset_index())
    screening={}
    opt=protocol["provisional_opt"]
    for name in models:
        if name in ["B10",opt,"ego-only"]:continue
        quality_ok=all(paired[reference][name]["score"]["ci95_low"]>=-2 and
            paired[reference][name]["no_at_fault_collisions"]["mean"]>=-.5 and
            paired[reference][name]["drivable_area_compliance"]["mean"]>=-.5 for reference in ["B10",opt])
        ratio=results[name]["policy_ms"]["p50"]/results[opt]["policy_ms"]["p50"]
        screening[name]={"pilot_quality_threshold":quality_ok,"p50_ratio_vs_provisional_opt":ratio,
            "pilot_speed_target":ratio<=.8 and results[name]["policy_ms"]["p95"]<=results[opt]["policy_ms"]["p95"],
            "pdms_mean_gain_vs_ego":paired["ego-only"][name]["score"]["mean"],
            "pdms_mean_gain_vs_CV":paired["CV"][name]["score"]["mean"]}
    count3=results["SF3-terminal"]["parameters"];countwide=results["SF1-wide-terminal"]["parameters"]
    assert abs(count3-countwide)/count3<=.05
    summary={"run_id":args.run_id,"models":results,"paired_differences_by_reference":paired,"pilot_screening":screening,
        "provisional_opt":opt,"parameter_match_relative_difference":abs(count3-countwide)/count3,
        "official_vendored_max_abs_difference_points":scorer_difference,"B_star_selected":False,"navtest_used":False,
        "full_expansion_authorized_by_evidence":False,"remaining_pilot_gates":["matched feature swap control","D-GT/D-Teacher target diagnostics"],
        "limitations":["Dev256 is part of the pilot1000 used for checkpoint selection; this is exploratory validation, not independent test.",
            "Only one head training seed (17); 12 validation logs and three regions.",
            "Head-val is inside navtrain and may overlap public backbone training.",
            "Sequential head blocks rotate across repeats; only one head is resident. GPU clocks are automatic.",
            "CPU CV timing is a reference from the prior opt run; ego-only is measured without a resident backbone.",
            "Early-exit phase comparisons jointly change noise and compute; they do not isolate future dynamics causally.",
            "Full head-val B* selection, final seeds and navtest remain pending."],
        "uncertainty":{"method":"paired log-cluster bootstrap, sample-weighted means","resamples":2000,"seed":17},
        "generated_at_utc":datetime.now(timezone.utc).isoformat()}
    frame=pd.DataFrame(table);frame.to_csv(out/"summary.csv",index=False)
    pd.concat(differences,ignore_index=True).to_csv(out/"paired-differences.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for position,name in enumerate(models):
        value=results[name];q=value["quality"]["score"];color=plt.get_cmap("tab20")(position)
        axes[0].scatter(value["policy_ms"]["p50"],q["mean"],color=color,s=35)
        axes[0].vlines(value["policy_ms"]["p50"],q["ci95_low"],q["ci95_high"],color=color)
        axes[0].annotate(name,(value["policy_ms"]["p50"],q["mean"]),xytext=(3,4),textcoords="offset points",fontsize=7)
    axes[0].set(xlabel="Online policy p50 (ms)",ylabel="Official PDMS (0-100)",title="Exploratory dev256 / log bootstrap intervals")
    for name in ["ego-only","FMLP-terminal","SF1-wide-terminal","SF3-current","SF3-first","SF3-terminal"]:
        history=json.loads((head_root/name/"history.json").read_text())
        axes[1].plot([row["epoch"] for row in history],[row["ade_4s_m"] for row in history],label=name)
    axes[1].set(xlabel="Training epoch",ylabel="Pilot1000 ADE (m)",title="Validation during frozen-feature training");axes[1].legend(fontsize=8)
    for ax in axes:ax.spines[["top","right"]].set_visible(False);ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.savefig(out/"head-comparison.png",dpi=180);fig.savefig(out/"head-comparison.svg");plt.close(fig)
    files=["protocol.json","online-cache-parity.json","inference-summary.json","prediction-index.jsonl","timings.jsonl",
           "scores-official.csv","scores-vendored.csv","score-official-summary.json","score-vendored-summary.json",
           "summary.json","summary.csv","paired-differences.csv","head-comparison.png","head-comparison.svg"]
    lock={"status":"pilot_head_native_online_dev256_verified","files":[{"path":str(out/name),"sha256":sha(out/name)} for name in files],
        "scripts":[{"path":str(ROOT/"scripts"/name),"sha256":sha(ROOT/"scripts"/name)} for name in ["remote_head_online_infer.py","remote_head_online_analyze.py","head_policy_runtime.py","remote_opt_score.py"]],
        "CV_reference_summary_sha256":sha(old/"summary.json"),"navtest_used":False,"B_star_selected":False}
    (out/"EVALUATION_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")
    (out/"analysis-status.json").write_text(json.dumps({"stage":"verified","summary":str(out/"summary.json")},indent=2)+"\n")
    print(frame.to_string(index=False))
