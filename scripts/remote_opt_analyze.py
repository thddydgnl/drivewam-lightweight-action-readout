"""Official dev256 raw/opt/CV quality, latency and provisional screening."""
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

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="opt_devval256_2026-10-01")
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    for stage in ["inference", "score-official", "score-vendored"]:
        assert json.loads((out / (stage + "-status.json")).read_text())["stage"] == "verified"
    protocol = json.loads((out / "protocol.json").read_text())
    models, tokens = protocol["model_names"], protocol["tokens"]
    official = pd.read_csv(out / "scores-official.csv", keep_default_na=False)
    vendored = pd.read_csv(out / "scores-vendored.csv", keep_default_na=False)
    assert len(official) == len(vendored) == protocol["expected_prediction_count"]
    assert official.valid.all() and vendored.valid.all()
    primary = official[official.generation_seed == 0].copy()
    assert len(primary) == 256 * len(models)
    primary[QUALITY] *= 100
    timing = pd.DataFrame(json.loads(line) for line in (out / "timings.jsonl").read_text().splitlines())
    measured = timing[timing.generation_seed == 0]
    results, table = {}, []
    for model in models:
        quality = primary[primary.model == model].set_index("scene_token").loc[tokens].reset_index()
        latency = measured[measured.model == model]
        assert len(quality) == 256 and len(latency) == 768
        assert quality.scene_token.is_unique
        repeat_stats = {}
        for repeat in range(3):
            subset = latency[latency["repeat"] == repeat]
            assert len(subset) == 256 and set(subset.scene_token) == set(tokens)
            repeat_stats[str(repeat)] = distribution(subset.policy_ms)
        medians = [v["p50"] for v in repeat_stats.values()]
        phases = {key: distribution(latency[key].dropna()) for key in latency.columns
                  if key.endswith("_gpu_ms") and not latency[key].isna().all()}
        confidence = cluster_bootstrap(quality, QUALITY + ERRORS)
        results[model] = {"quality": confidence, "policy_ms": distribution(latency.policy_ms),
            "policy_ms_by_repeat": repeat_stats,
            "repeat_p50_relative_range": float((max(medians)-min(medians)) / np.median(medians)),
            "execution_device": "cpu" if model == "CV" else "cuda:0",
            "gpu_phase_ms": phases, "coverage": 1.0, "scenes": 256,
            "logs": int(quality.log_name.nunique()),
            "actual_forward_calls": sorted(latency.actual_forward_calls.unique().astype(int).tolist()),
            "peak_allocated_gib": float(latency.peak_allocated_bytes.max() / 1024**3),
            "peak_reserved_gib": float(latency.peak_reserved_bytes.max() / 1024**3),
            "at_fault_collision_scenes": int(quality.at_fault_collision.sum()),
            "dac_failure_scenes": int((quality.drivable_area_compliance < 100).sum()),
            "ttc_infraction_scenes": int(quality.ttc_infraction.sum())}
        table.append({"model": model, "scenes": 256, **{key: confidence[key]["mean"] for key in QUALITY + ERRORS},
            "policy_p50_ms": results[model]["policy_ms"]["p50"],
            "policy_p95_ms": results[model]["policy_ms"]["p95"],
            "action_gpu_p50_ms": phases.get("action_gpu_ms", {}).get("p50", 0.0),
            "forward_calls": results[model]["actual_forward_calls"][0],
            "peak_allocated_gib": results[model]["peak_allocated_gib"]})
    reference = primary[primary.model == "B10"].set_index("scene_token").loc[tokens]
    paired, candidates = {}, {}
    difference_rows = []
    for model in models:
        if model == "B10":
            continue
        quality = primary[primary.model == model].set_index("scene_token").loc[tokens]
        assert list(quality.log_name) == list(reference.log_name)
        difference = quality[QUALITY + ERRORS] - reference[QUALITY + ERRORS]
        difference["log_name"] = reference.log_name
        paired[model] = cluster_bootstrap(difference, QUALITY + ERRORS)
        difference["model"] = model
        difference_rows.append(difference.reset_index())
        if model.endswith("-opt"):
            candidates[model] = {
                "pilot_quality_eligible": paired[model]["score"]["ci95_low"] >= -2
                    and paired[model]["no_at_fault_collisions"]["mean"] >= -.5
                    and paired[model]["drivable_area_compliance"]["mean"] >= -.5,
                "pdms_difference_ci95": paired[model]["score"],
                "policy_p50_reduction_vs_B10_percent": (1-results[model]["policy_ms"]["p50"] / results["B10"]["policy_ms"]["p50"]) * 100}
    pd.concat(difference_rows, ignore_index=True).to_csv(out / "paired-differences.csv", index=False)
    eligible = [m for m,v in candidates.items() if v["pilot_quality_eligible"]]
    provisional = min(eligible, key=lambda m: (results[m]["policy_ms"]["p50"], results[m]["policy_ms"]["p95"], -protocol["action_steps"][m])) if eligible else None
    skipped_pass = {model: {"policy_p50_saved_ms": results[model]["policy_ms"]["p50"] - results[model+"-opt"]["policy_ms"]["p50"],
            "policy_p50_reduction_percent": (1-results[model+"-opt"]["policy_ms"]["p50"] / results[model]["policy_ms"]["p50"]) * 100}
        for model in ["B10", "B05"]}
    keys = ["model", "generation_seed", "scene_token"]
    joined = official.merge(vendored, on=keys, suffixes=("_official", "_vendored"), how="outer", indicator=True, validate="one_to_one")
    assert len(joined) == protocol["expected_prediction_count"] and (joined._merge == "both").all()
    for field in ["prediction_sha256", "target_sha256", "metric_cache_sha256"]:
        assert (joined[field+"_official"] == joined[field+"_vendored"]).all()
    scorer_diff = {}
    for metric in QUALITY:
        delta = (joined[metric+"_vendored"] - joined[metric+"_official"]) * 100
        scorer_diff[metric] = {"mean_vendored_minus_official_points": float(delta.mean()),
            "max_abs_difference_points": float(delta.abs().max()), "rows_above_1e-6_points": int((delta.abs()>1e-6).sum())}
    diagnostics = {}
    for model in protocol["gpu_model_names"]:
        subset = official[(official.model == model) & official.scene_token.isin(protocol["diagnostic_tokens"])]
        assert len(subset) == 96
        diagnostics[model] = {"scenes": 32, "seeds": [0,1,2],
            "seed_pdms_means": {str(seed): float(subset[subset.generation_seed == seed].score.mean()*100) for seed in [0,1,2]},
            "mean_within_scene_pdms_population_std_points": float((subset.pivot(index="scene_token", columns="generation_seed", values="score")*100).std(axis=1, ddof=0).mean())}
    manifest = {r["scene_token"]: r for r in (json.loads(line) for line in (ROOT / "splits/pilot-manifest.jsonl").read_text().splitlines())}
    subgroup_rows = []
    for key in ["map_name", "command_id", "speed_group"]:
        primary[key] = primary.scene_token.map(lambda t: ("under_2_mps" if manifest[t]["speed_mps"]<2 else "at_least_2_mps") if key=="speed_group" else manifest[t][key])
        for (model, group), subset in primary.groupby(["model", key], sort=True):
            subgroup_rows.append({"model": model, "dimension": key, "group": str(group), "scenes": len(subset),
                "logs": int(subset.log_name.nunique()), "pdms": float(subset.score.mean()),
                "ade_4s_m": float(subset.ade_4s_m.mean()), "fde_4s_m": float(subset.fde_4s_m.mean())})
    pd.DataFrame(subgroup_rows).to_csv(out / "subgroups.csv", index=False)
    summary = {"run_id": args.run_id, "scope": "navtrain internal head-val dev256 pilot; not navtest/full head-val",
        "models": results, "paired_vs_B10": paired, "terminal_action_pass_savings": skipped_pass,
        "pilot_opt_screening": candidates, "provisional_fastest_eligible_opt": provisional,
        "baseline_B_star_selected": False, "navtest_used": False,
        "generation_seed_diagnostic": diagnostics, "scorer_differences_all_predictions": scorer_diff,
        "uncertainty": {"method": "paired log-cluster bootstrap, sample-weighted means", "logs": 12, "resamples": 2000, "seed":17},
        "limitations": ["Only 12 logs and 3 regions; Pittsburgh absent.",
            "Head-val comes from navtrain and may overlap public backbone training; this does not assess backbone generalization.",
            "Cached route text embeddings; online VAE/world/action/CPU trajectory included, file I/O and scoring excluded.",
            "Phase events and forward counting add common instrumentation overhead.",
            "CV is an observation-only CPU reference, never a B* candidate.",
            "B* must be selected on complete head-val; pilot screening is provisional."],
        "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    unstable = [m for m in protocol["gpu_model_names"] if results[m]["repeat_p50_relative_range"]>.05]
    if unstable: summary["limitations"].append("Repeat median spread exceeds 5%: " + ", ".join(unstable))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    frame = pd.DataFrame(table)
    frame.to_csv(out / "summary.csv", index=False)
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=(12,4.5), constrained_layout=True)
    for index, model in enumerate(models):
        stats=results[model]["quality"]["score"]
        axes[0].bar(index, stats["mean"], color=colors[index])
        axes[0].vlines(index,stats["ci95_low"],stats["ci95_high"],color="black",linewidth=1)
        axes[0].hlines([stats["ci95_low"],stats["ci95_high"]],index-.07,index+.07,color="black",linewidth=1)
        if model != "CV":
            latency=np.sort(measured[measured.model==model].policy_ms)
            axes[1].plot(latency,np.arange(1,len(latency)+1)/len(latency),label=model,color=colors[index])
    axes[0].set(xticks=list(range(len(models))),xticklabels=models,ylabel="Official PDMS (0-100)",ylim=(0,100),title="Dev256 / 12 log clusters")
    axes[0].tick_params(axis="x",labelrotation=35,labelsize=8)
    axes[1].set(xlabel="Online GPU policy latency (ms)",ylabel="Cumulative fraction",title="Same A5000 / 3 x 256 per model")
    axes[1].legend(fontsize=8)
    for ax in axes: ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.15); ax.set_axisbelow(True)
    fig.savefig(out/"opt-comparison.png",dpi=180);fig.savefig(out/"opt-comparison.svg");plt.close(fig)
    files=["protocol.json","action-opt-transformation.json","raw-opt-parity.json","instrumentation-parity.json",
        "inference-summary.json","prediction-index.jsonl","timings.jsonl","scores-official.csv","scores-vendored.csv",
        "score-official-summary.json","score-vendored-summary.json","summary.json","summary.csv","paired-differences.csv","subgroups.csv","opt-comparison.png","opt-comparison.svg"]
    scripts=["baseline_common.py","remote_baseline_infer.py","remote_baseline_analyze.py","navsim_action_opt.py","remote_opt_infer.py","remote_opt_score.py","remote_opt_analyze.py"]
    lock={"status":"dev256_raw_opt_CV_quality_and_three_repeat_timing_verified","run_id":args.run_id,"workspace":str(ROOT),
        "files":[{"path":str(out/name),"bytes":(out/name).stat().st_size,"sha256":sha(out/name)} for name in files],
        "scripts":[{"path":str(ROOT/"scripts"/name),"sha256":sha(ROOT/"scripts"/name)} for name in scripts],
        "models":models,"primary_scored_each_model":256,"timing_samples_each_model":768,
        "prediction_count":protocol["expected_prediction_count"],"B_star_selected":False,"navtest_used":False}
    (out/"EVALUATION_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")
    (out/"analysis-status.json").write_text(json.dumps({"stage":"verified","summary":str(out/"summary.json")},indent=2)+"\n")
    print(frame.to_string(index=False))
    print(json.dumps({"provisional_fastest_eligible_opt":provisional,"pass_savings":skipped_pass},indent=2))
