"""navtest-dev analysis per the 2026-10-02 pre-registration (scorer environment).

Reads official per-scene scores, computes sample-weighted means, paired
log-cluster bootstrap differences (34 logs), the head-val vs navtest-dev
memorization comparison, quality-only B* screening and generation-seed
sensitivity. Latency gates are applied later with the re-measured timings.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from remote_baseline_analyze import QUALITY, cluster_bootstrap

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
QUALITY_RUN = ROOT / "runs/navtest_dev_quality_2026-10-02"
HEADS = ROOT / "runs/heads_navtest_dev_2026-10-02"
ERRORS = ["ade_4s_m", "fde_4s_m", "ade_3s_m", "fde_3s_m", "heading_mae_rad"]
METRICS = QUALITY + ERRORS

def load(path):
    frame = pd.read_csv(path, keep_default_na=False)
    assert frame["valid"].astype(str).isin(["True", "true", "1"]).all() and len(frame) == 3250
    frame[QUALITY] = frame[QUALITY].astype(float) * 100
    return frame.set_index("scene_token")

def paired(model, reference):
    joined = model[METRICS].subtract(reference[METRICS])
    joined["log_name"] = reference["log_name"]
    return cluster_bootstrap(joined.reset_index(drop=True), METRICS)

if __name__ == "__main__":
    tokens = (ROOT / "splits/navtest-dev-tokens.txt").read_text().splitlines()
    scores = {}
    for path in sorted((QUALITY_RUN / "scores").glob("*.csv")):
        scores[path.stem] = load(path)
    for path in sorted((HEADS / "scores/v1").glob("*.csv")):
        scores["v1:" + path.stem] = load(path)
    for path in sorted((HEADS / "scores/v2").glob("*.csv")):
        scores["v2:" + path.stem] = load(path)
    for frame in scores.values():
        assert sorted(frame.index) == sorted(tokens)
    scores = {name: frame.loc[tokens] for name, frame in scores.items()}
    means = {name: {m: float(f[m].mean()) for m in METRICS} for name, f in scores.items()}

    refs = [r for r in ["B10", "B03-opt", "B02-opt", "v1:SF3-terminal", "v1:SF3-first", "v1:SF3-current"] if r in scores]
    pairs = {ref: {name: paired(scores[name], scores[ref]) for name in scores if name != ref} for ref in refs}
    for seed in [17, 29]:
        for ref in [f"v2:SF3-terminal@seed{seed}", f"v2:SF3-first@seed{seed}", f"v2:SF3-current@seed{seed}", f"v2:SF1-wide-terminal@seed{seed}"]:
            if ref in scores:
                pairs[ref] = {name: paired(scores[name], scores[ref]) for name in scores
                              if name != ref and (name.endswith(f"@seed{seed}") or not name.startswith("v2:"))}

    # Memorization: same models on navtrain head-val dev256 (different scene sets; descriptive only).
    opt = pd.read_csv(ROOT / "runs/opt_devval256_2026-10-01/summary.csv").set_index("model")
    heads = pd.read_csv(ROOT / "runs/heads_online_devval256_2026-10-01-r2/summary.csv").set_index("model")
    memorization = {}
    for name in scores:
        base = name.split(":")[-1]
        source = opt if base in opt.index else heads if (name.startswith("v1:") and base in heads.index) else None
        if source is not None:
            memorization[name] = {"headval_dev256_pdms": float(source.loc[base, "score"]), "navtest_dev_pdms": means[name]["score"],
                                  "headval_dev256_ade4": float(source.loc[base, "ade_4s_m"]), "navtest_dev_ade4": means[name]["ade_4s_m"]}

    # Quality-only B* screening (pre-registered §1.3 applied to navtest-dev).
    screening = {}
    for name in ["B05-opt", "B03-opt", "B02-opt", "B01-opt"]:
        if name in scores:
            d = pairs["B10"][name]
            screening[name] = {"pdms_diff": d["score"], "nc_mean_drop": -d["no_at_fault_collisions"]["mean"],
                "dac_mean_drop": -d["drivable_area_compliance"]["mean"],
                "quality_eligible": d["score"]["ci95_low"] >= -2.0 and -d["no_at_fault_collisions"]["mean"] <= .5
                                    and -d["drivable_area_compliance"]["mean"] <= .5}

    # Generation-seed sensitivity of B10 on the fixed 32 scenes.
    sensitivity = None
    seed_path = QUALITY_RUN / "predictions/B10-seed-sensitivity.npz"
    if seed_path.exists():
        with np.load(seed_path, allow_pickle=False) as data:
            keys = data["keys"].astype(str).tolist(); trajectories = data["trajectories"]
        with np.load(QUALITY_RUN / "predictions/B10.npz", allow_pickle=False) as data:
            seed0 = dict(zip(data["scene_tokens"].astype(str).tolist(), data["trajectories"]))
        manifest = {r["scene_token"]: r for r in map(json.loads, (ROOT / "splits/navtest-dev-manifest.jsonl").read_text().splitlines())}
        spreads, ades = [], {0: [], 1: [], 2: []}
        for token in sorted({k.split(":")[0] for k in keys}):
            stack = np.stack([seed0[token]] + [trajectories[keys.index(f"{token}:{g}")] for g in [1, 2]])
            spreads.append(float(np.sqrt(((stack[:, :, :2] - stack[:, :, :2].mean(0)) ** 2).sum(-1).mean())))
            with np.load(manifest[token]["label_path"], allow_pickle=False) as target:
                gt = target["future_poses_4s"]
            for g in range(3):
                ades[g].append(float(np.linalg.norm(stack[g, :, :2] - gt[:, :2], axis=1).mean()))
        sensitivity = {"scenes": len(spreads), "mean_trajectory_rms_spread_m": float(np.mean(spreads)),
                       "ade4_by_seed_m": {g: float(np.mean(v)) for g, v in ades.items()},
                       "headval_reference_rms_spread_m": 0.0351}

    summary = {"split": "navtest_dev", "scenes": len(tokens), "logs": int(scores["B10"]["log_name"].nunique()),
        "preregistration": "reports/사전등록_navtest-dev_2026-10-02.md", "means": means, "paired_differences": pairs,
        "memorization_headval_vs_navtest_dev": memorization, "B_star_quality_screening": screening,
        "B_star_selected": False, "B_star_note": "final B* needs the re-measured latency (pre-registration §5)",
        "B10_generation_seed_sensitivity": sensitivity, "navtest_final_used": False,
        "uncertainty": {"method": "paired log-cluster bootstrap, sample-weighted", "resamples": 2000, "seed": 17},
        "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    (ROOT / "reports").mkdir(exist_ok=True)
    out = QUALITY_RUN / "analysis-summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    table = pd.DataFrame(means).T[["score", "no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
                                   "comfort", "ego_progress", "ade_4s_m", "fde_4s_m"]]
    table.to_csv(QUALITY_RUN / "analysis-means.csv")
    print(table.round(3).to_string())
    print(json.dumps({"memorization": memorization, "screening": {k: v["quality_eligible"] for k, v in screening.items()},
                      "seed_sensitivity": sensitivity}, indent=1))
