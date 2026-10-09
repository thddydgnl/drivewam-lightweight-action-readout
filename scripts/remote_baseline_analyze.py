"""Summarize paired dev256 quality, three-repeat latency and scorer differences."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
MODELS = ["B10", "B05"]
QUALITY = ["score", "no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
           "comfort", "ego_progress", "driving_direction_compliance"]
ERRORS = ["ade_4s_m", "fde_4s_m", "ade_3s_m", "fde_3s_m", "heading_mae_rad"]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "mean": float(values.mean()), "p50": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "p95": float(np.quantile(values, .95)),
            "min": float(values.min()), "max": float(values.max())}

def cluster_bootstrap(frame, metrics):
    # Each sampled log contributes all of its selected scenes, preserving the
    # official sample-weighted mean instead of averaging equally weighted logs.
    grouped = frame.groupby("log_name", sort=True)
    counts = grouped.size().to_numpy()
    sums = grouped[metrics].sum().to_numpy()
    draws = np.random.default_rng(17).multinomial(len(counts), np.ones(len(counts)) / len(counts), size=2000)
    bootstrap = (draws @ sums) / (draws @ counts)[:, None]
    return {metric: {"mean": float(frame[metric].mean()),
                    "ci95_low": float(np.quantile(bootstrap[:, i], .025)),
                    "ci95_high": float(np.quantile(bootstrap[:, i], .975))}
            for i, metric in enumerate(metrics)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="baselines_devval256_2026-10-01")
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    for name in ["inference", "score-official", "score-vendored"]:
        assert json.loads((out / (name + "-status.json")).read_text())["stage"] == "verified", name
    protocol = json.loads((out / "protocol.json").read_text())
    tokens = protocol["tokens"]
    official = pd.read_csv(out / "scores-official.csv", keep_default_na=False)
    vendored = pd.read_csv(out / "scores-vendored.csv", keep_default_na=False)
    assert official["valid"].all() and vendored["valid"].all()
    assert len(official) == len(vendored) == 640
    timing = pd.DataFrame(json.loads(line) for line in (out / "timings.jsonl").read_text().splitlines())
    primary = official[official.generation_seed == 0].copy()
    measured = timing[timing.generation_seed == 0].copy()
    assert len(primary) == 512 and len(measured) == 1536
    rows = {r["scene_token"]: r for r in (json.loads(line) for line in
            (ROOT / "splits/pilot-manifest.jsonl").read_text().splitlines())}
    for metric in QUALITY:
        primary[metric] *= 100
    results = {}
    table = []
    for model in MODELS:
        quality = primary[primary.model == model].set_index("scene_token").loc[tokens].reset_index()
        latency = measured[measured.model == model]
        assert len(quality) == 256 and len(latency) == 768
        assert quality.scene_token.is_unique and set(latency.scene_token) == set(tokens)
        for repeat in range(3):
            assert len(latency[latency["repeat"] == repeat]) == 256
        confidence = cluster_bootstrap(quality, QUALITY + ERRORS)
        latency_stats = distribution(latency.policy_ms)
        repeat_stats = {str(repeat): distribution(latency[latency["repeat"] == repeat].policy_ms) for repeat in range(3)}
        repeat_p50s = [stats["p50"] for stats in repeat_stats.values()]
        repeat_spread = (max(repeat_p50s) - min(repeat_p50s)) / np.median(repeat_p50s)
        phases = {name: distribution(latency[name]) for name in latency.columns if name.endswith("_gpu_ms")}
        results[model] = {"quality": confidence, "coverage": 1.0, "scenes": 256, "logs": int(quality.log_name.nunique()),
            "at_fault_collision_scenes": int(quality.at_fault_collision.sum()),
            "nc_below_one_scenes": int((quality.no_at_fault_collisions < 100).sum()),
            "dac_failure_scenes": int((quality.drivable_area_compliance < 100).sum()),
            "ttc_infraction_scenes": int(quality.ttc_infraction.sum()),
            "policy_ms": latency_stats, "policy_ms_by_repeat": repeat_stats,
            "repeat_p50_relative_range": float(repeat_spread), "gpu_phase_ms": phases,
            "actual_forward_calls": sorted(latency.actual_forward_calls.unique().astype(int).tolist()),
            "peak_allocated_gib": float(latency.peak_allocated_bytes.max() / 1024**3),
            "peak_reserved_gib": float(latency.peak_reserved_bytes.max() / 1024**3)}
        table.append({"model": model, "scenes": 256, **{m: confidence[m]["mean"] for m in QUALITY + ERRORS},
           "policy_p50_ms": latency_stats["p50"], "policy_p90_ms": latency_stats["p90"],
           "policy_p95_ms": latency_stats["p95"], "action_gpu_p50_ms": phases["action_gpu_ms"]["p50"],
           "peak_allocated_gib": results[model]["peak_allocated_gib"],
           "at_fault_collision_scenes": results[model]["at_fault_collision_scenes"],
           "dac_failure_scenes": results[model]["dac_failure_scenes"]})
    b10 = primary[primary.model == "B10"].set_index("scene_token").loc[tokens]
    b05 = primary[primary.model == "B05"].set_index("scene_token").loc[tokens]
    assert list(b10.log_name) == list(b05.log_name)
    differences = pd.DataFrame({metric: b05[metric] - b10[metric] for metric in QUALITY + ERRORS})
    differences["log_name"] = b10.log_name
    paired_ci = cluster_bootstrap(differences, QUALITY + ERRORS)
    differences.reset_index().to_csv(out / "paired-differences.csv", index=False)
    quality_speed = {"policy_p50_ratio_B05_over_B10": results["B05"]["policy_ms"]["p50"] / results["B10"]["policy_ms"]["p50"],
       "policy_p95_ratio_B05_over_B10": results["B05"]["policy_ms"]["p95"] / results["B10"]["policy_ms"]["p95"]}
    quality_speed["policy_p50_reduction_percent"] = (1 - quality_speed["policy_p50_ratio_B05_over_B10"]) * 100
    keys = ["model", "generation_seed", "scene_token"]
    joined = official.merge(vendored, on=keys, how="outer", suffixes=("_official", "_vendored"), validate="one_to_one", indicator=True)
    assert (joined._merge == "both").all()
    for field in ["prediction_sha256", "target_sha256", "metric_cache_sha256"]:
        assert (joined[field + "_official"] == joined[field + "_vendored"]).all(), field
    scorer_difference = {}
    for metric in QUALITY:
        delta = (joined[metric + "_vendored"] - joined[metric + "_official"]) * 100
        scorer_difference[metric] = {"mean_vendored_minus_official_points": float(delta.mean()),
            "max_abs_difference_points": float(delta.abs().max()), "rows_differing_above_1e-6_points": int((delta.abs() > 1e-6).sum())}
    pd.DataFrame({**{key: joined[key] for key in keys},
                  **{metric + "_vendored_minus_official_points": (joined[metric + "_vendored"] - joined[metric + "_official"]) * 100 for metric in QUALITY}}).to_csv(out / "scorer-differences.csv", index=False)
    # Prespecified metadata subgroups; sparse groups stay descriptive.
    subgroup_rows = []
    primary["map_name"] = primary.scene_token.map(lambda t: rows[t]["map_name"])
    primary["command_id"] = primary.scene_token.map(lambda t: rows[t]["command_id"])
    primary["speed_group"] = primary.scene_token.map(lambda t: "under_2_mps" if rows[t]["speed_mps"] < 2 else "at_least_2_mps")
    for dimension in ["map_name", "command_id", "speed_group"]:
        for (model, group), frame in primary.groupby(["model", dimension], sort=True):
            subgroup_rows.append({"model": model, "dimension": dimension, "group": str(group), "scenes": len(frame),
                "logs": int(frame.log_name.nunique()), "pdms": float(frame.score.mean()),
                "ade_4s_m": float(frame.ade_4s_m.mean()), "fde_4s_m": float(frame.fde_4s_m.mean())})
    pd.DataFrame(subgroup_rows).to_csv(out / "subgroups.csv", index=False)
    diagnostics = {}
    diagnostic_tokens = protocol["diagnostic_tokens"]
    for model in MODELS:
        subset = official[(official.model == model) & official.scene_token.isin(diagnostic_tokens)].copy()
        assert len(subset) == 96
        means = {str(seed): {"pdms": float(subset[subset.generation_seed == seed].score.mean() * 100),
                          "ade_4s_m": float(subset[subset.generation_seed == seed].ade_4s_m.mean())} for seed in [0, 1, 2]}
        within_scene = subset.pivot(index="scene_token", columns="generation_seed", values="score") * 100
        spreads = []
        for token in diagnostic_tokens:
            predictions = []
            for seed in [0, 1, 2]:
                with np.load(out / "predictions" / model / f"generation_seed_{seed}" / (token + ".npz"), allow_pickle=False) as data:
                    predictions.append(data["trajectory"][:, :2])
            predictions = np.stack(predictions)
            spreads.append(float(np.sqrt(np.mean(np.sum((predictions - predictions.mean(axis=0))**2, axis=-1)))))
        diagnostics[model] = {"scenes": 32, "generation_seeds": [0, 1, 2], "seed_means": means,
           "mean_within_scene_pdms_population_std_points": float(within_scene.std(axis=1, ddof=0).mean()),
           "mean_trajectory_rms_spread_m": float(np.mean(spreads)),
           "seed_rows_are_not_independent_scenes": True}
    summary = {"run_id": args.run_id, "client_date": "2026-10-01", "scope": "head-val dev256 pilot only; not navtest or full head-val",
        "models": results, "paired_B05_minus_B10": paired_ci, "speed": quality_speed,
        "uncertainty": {"method": "paired log-cluster bootstrap preserving sample-weighted means", "resamples": 2000,
                        "bootstrap_seed": 17, "unique_logs": int(primary.log_name.nunique()), "confidence": 0.95},
        "scorer_difference_all_640_predictions": scorer_difference, "generation_seed_diagnostic": diagnostics,
        "metadata": {"maps": dict(Counter(rows[t]["map_name"] for t in tokens)),
                     "commands": dict(Counter(str(rows[t]["command_id"]) for t in tokens))},
        "baseline_B_star_selected": False, "navtest_used": False,
        "limitations": [f"Dev256 archive cohort covers {int(primary.log_name.nunique())} head-val logs; Pittsburgh is absent.",
                        "Head-val is split from navtrain and may overlap the public backbone's training data; this does not measure backbone generalization.",
                        "This evaluates public frozen checkpoint/code; it is not reproduction of a published navtest score.",
                        "Timing uses fixed precomputed text embeddings and includes online VAE/world/action; disk and CPU scoring excluded.",
                        "Phase CUDA events and forward-count hooks add small measurement overhead and are identical for both models.",
                        "B* requires opt parity/candidates and complete head-val; it is not selected from these raw baselines."],
        "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    unstable = [model for model in MODELS if results[model]["repeat_p50_relative_range"] > .05]
    if unstable:
        summary["limitations"].append("Repeat p50 range exceeds 5% for " + ", ".join(unstable) + "; inspect telemetry before using fine latency differences.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(table).to_csv(out / "summary.csv", index=False)
    palette = {"B10": "#2563eb", "B05": "#0f766e"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4), constrained_layout=True)
    for index, model in enumerate(MODELS):
        stats = results[model]["quality"]["score"]
        axes[0].bar(index, stats["mean"], color=palette[model], width=.55)
        axes[0].errorbar(index, stats["mean"], yerr=[[stats["mean"]-stats["ci95_low"]], [stats["ci95_high"]-stats["mean"]]],
                          color="#111827", capsize=5, fmt="none")
        axes[0].text(index, stats["mean"] + 1, f"{stats['mean']:.2f}", ha="center", va="bottom")
        latency = np.sort(measured[measured.model == model].policy_ms.to_numpy())
        axes[1].plot(latency, np.arange(1, len(latency)+1)/len(latency), color=palette[model], label=model, linewidth=2)
    axes[0].set(xticks=[0, 1], xticklabels=MODELS, ylabel="Official PDMS (0-100)", ylim=(0, 100),
                title=f"Dev256 quality / {int(primary.log_name.nunique())} log clusters")
    axes[1].set(xlabel="Online policy latency (ms)", ylabel="Cumulative fraction", title="Same A5000 / 3 x 256 samples")
    axes[1].axhline(.5, color="#9ca3af", linestyle=":", linewidth=1)
    axes[1].axhline(.95, color="#9ca3af", linestyle=":", linewidth=1)
    axes[1].legend(loc="lower right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(out / "baseline-comparison.png", dpi=180)
    fig.savefig(out / "baseline-comparison.svg")
    plt.close(fig)
    # Prespecified first dev token, not a post-hoc best/worst example. This
    # visual also makes the current rear-axle frame and 0.5 s horizon explicit.
    token = tokens[0]
    with np.load(rows[token]["input_path"], allow_pickle=False) as data:
        current_rgb = data["rgb_current"].copy()
    with np.load(rows[token]["label_path"], allow_pickle=False) as data:
        gt = data["future_poses_4s"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(current_rgb)
    axes[0].axis("off")
    axes[0].set_title("Current front camera / observation only")
    for model in MODELS + ["GT"]:
        if model == "GT":
            poses, color = gt, "#111827"
        else:
            with np.load(out / "predictions" / model / "generation_seed_0" / (token + ".npz"), allow_pickle=False) as data:
                poses = data["trajectory"].copy()
            color = palette[model]
        poses = np.concatenate([np.zeros((1, 3)), poses], axis=0)
        axes[1].plot(poses[:, 1], poses[:, 0], ".-", color=color, label=model)
    axes[1].scatter([0], [0], marker="x", color="#b91c1c", label="Current rear axle")
    axes[1].set(xlabel="Ego-local lateral y (m)", ylabel="Ego-local forward x (m)",
                title="0.5 s steps / 4 s horizon")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Fixed first dev token: {token}", fontsize=10)
    fig.savefig(out / "trajectory-example.png", dpi=180)
    plt.close(fig)
    critical = ["protocol.json", "instrumentation-parity.json", "inference-summary.json", "timings.jsonl",
                "prediction-index.jsonl", "scores-official.csv", "scores-vendored.csv", "score-official-summary.json",
                "score-vendored-summary.json", "summary.json", "summary.csv", "paired-differences.csv",
                "scorer-differences.csv", "subgroups.csv", "baseline-comparison.png", "baseline-comparison.svg", "trajectory-example.png"]
    record = {"status": "dev256_B10_B05_quality_and_three_repeat_timing_verified", "run_id": args.run_id,
              "workspace": str(ROOT), "protocol_sha256": sha(out / "protocol.json"),
              "files": [{"path": str(out / filename), "bytes": (out / filename).stat().st_size, "sha256": sha(out / filename)} for filename in critical],
              "scripts": [{"path": str(ROOT / "scripts" / filename), "sha256": sha(ROOT / "scripts" / filename)}
                          for filename in ["baseline_common.py", "remote_baseline_infer.py", "remote_baseline_score.py", "remote_baseline_analyze.py"]],
              "primary_scored_each_model": 256, "timing_measured_each_model": 768, "diagnostic_scenes": 32,
              "navtest_used": False, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    (out / "EVALUATION_LOCK.json").write_text(json.dumps(record, indent=2) + "\n")
    (out / "analysis-status.json").write_text(json.dumps({"stage": "verified", "summary": str(out / "summary.json")}, indent=2) + "\n")
    print(pd.DataFrame(table).to_string(index=False))
    print(json.dumps({"paired_score_delta": paired_ci["score"], "speed": quality_speed}, indent=2))
