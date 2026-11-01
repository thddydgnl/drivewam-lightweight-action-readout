"""Round-5 analysis (pre-registration 2026-11-01, reports/사전등록_보완실험_2026-11-01.md).

  assemble --run R --split dev|final   merge NAVSIM shard npz into predictions/<model>.npz
  nav-screen                           NAVSIM dev quality screening vs B10 (§1.2), old v=3 + grid
  pai-screen                           PhysicalAI dev quality screening vs B10-opt (§1.2), old v=3 + grid
  select                               B* re-selection from the idle re-measurement (§1.4)
  seed-diag                            pass-1 noise seed sensitivity, both benchmarks (§3)
  phase-diag                           PhysicalAI 20k current vs first and the ego-only control (§2)
  changed --bench navsim|physicalai    exit 3 if B* kept, else print the re-selected B* (run name)
  nav-final / pai-final                original final verdict rules with the (re-)selected B* and the
                                       idle latency (§1.4; navtest §8, PhysicalAI §6)
NAVSIM uses the paired log-cluster bootstrap of remote_baseline_analyze.cluster_bootstrap,
re-implemented with numpy so it runs in either environment; the stored 10/2 screening is
reproduced as a check. PhysicalAI reuses remote_pai_analyze (model environment).
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import sys

import numpy as np

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
REPORTS = ROOT / "reports"
NAV_GRID = "r5_navsim_grid_dev_2026-11-01"
PAI_GRID = "pai_dev_grid-V{}_2026-11-01"
GRID = [f"V{v}-A{k}" for v in [0, 1, 2] for k in [1, 2, 3, 5]]
QUALITY = ["score", "no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
           "comfort", "ego_progress", "driving_direction_compliance"]
ERRORS = ["ade_4s_m", "fde_4s_m", "ade_3s_m", "fde_3s_m", "heading_mae_rad"]

def steps(name):
    """(video steps, action steps) of a baseline name."""
    m = re.fullmatch(r"V(\d)-A(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 3, 10 if name in ["B10", "B10-opt"] else int(re.fullmatch(r"B(\d+)-opt", name).group(1))

def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)

def assemble(run_id, split):
    run = ROOT / "runs" / run_id
    for s in [0, 1]:
        assert json.loads((run / f"inference-shard{s}-status.json").read_text())["stage"] == "verified", s
    tokens = (ROOT / f"splits/navtest-{split}-tokens.txt").read_text().splitlines()
    got = {}
    for f in sorted(run.glob("shard*/*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        with np.load(f) as d:
            names = d["scene_tokens"].astype(str)
            for key in d.files:
                if key != "scene_tokens":
                    for t, x in zip(names, d[key]):
                        got.setdefault(key, {})[t] = x
    (run / "predictions").mkdir(exist_ok=True)
    for key, v in got.items():
        assert set(v) == set(tokens), (key, len(v))
        name = key.replace("__", "@").replace("_", "-")
        np.savez(run / "predictions" / f"{name}.npz", scene_tokens=np.asarray(tokens),
                 trajectories=np.stack([v[t] for t in tokens]).astype(np.float32))
    print("assembled", run_id, sorted(got), flush=True)

def nav_load(path, n=3250):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n and all(r["valid"] in ["True", "true", "1"] for r in rows), path
    return {r["scene_token"]: {"log_name": r["log_name"], **{m: float(r[m]) * (100 if m in QUALITY else 1) for m in QUALITY + ERRORS}}
            for r in rows}

def cluster_bootstrap(model, reference, tokens, metrics=QUALITY + ERRORS):
    """remote_baseline_analyze.cluster_bootstrap on per-scene differences (model - reference)."""
    logs = sorted({reference[t]["log_name"] for t in tokens})
    pos = {g: i for i, g in enumerate(logs)}
    diff = np.array([[model[t][m] - reference[t][m] for m in metrics] for t in tokens])
    group = np.array([pos[reference[t]["log_name"]] for t in tokens])
    counts = np.bincount(group, minlength=len(logs))
    sums = np.stack([np.bincount(group, weights=diff[:, i], minlength=len(logs)) for i in range(len(metrics))], axis=1)
    draws = np.random.default_rng(17).multinomial(len(counts), np.ones(len(counts)) / len(counts), size=2000)
    boot = (draws @ sums) / (draws @ counts)[:, None]
    return {m: {"mean": float(diff[:, i].mean()), "ci95_low": float(np.quantile(boot[:, i], .025)),
                "ci95_high": float(np.quantile(boot[:, i], .975))} for i, m in enumerate(metrics)}

def nav_screen():
    tokens = (ROOT / "splits/navtest-dev-tokens.txt").read_text().splitlines()
    old_run = ROOT / "runs/navtest_dev_quality_2026-10-02"
    b10 = nav_load(old_run / "scores/B10.csv")
    paths = {m: old_run / f"scores/{m}.csv" for m in ["B05-opt", "B03-opt", "B02-opt", "B01-opt"]}
    paths.update({m: ROOT / "runs" / NAV_GRID / f"scores/{m}.csv" for m in GRID})
    stored = json.loads((old_run / "analysis-summary.json").read_text())["B_star_quality_screening"]
    out = {}
    for name, path in paths.items():
        d = cluster_bootstrap(nav_load(path), b10, tokens)
        out[name] = {"video_steps": steps(name)[0], "action_steps": steps(name)[1], "pdms_diff": d["score"],
                     "nc_mean_drop": -d["no_at_fault_collisions"]["mean"], "dac_mean_drop": -d["drivable_area_compliance"]["mean"],
                     "ade_4s_diff": d["ade_4s_m"], "paired_vs_B10": d}
        out[name]["quality_eligible"] = bool(d["score"]["ci95_low"] >= -2.0 and out[name]["nc_mean_drop"] <= .5
                                             and out[name]["dac_mean_drop"] <= .5)
        if name in stored:                       # the 10/2 screening must be reproduced exactly
            for k in ["mean", "ci95_low", "ci95_high"]:
                assert abs(stored[name]["pdms_diff"][k] - d["score"][k]) < 1e-9, (name, k)
            assert stored[name]["quality_eligible"] == out[name]["quality_eligible"], name
    report = {"benchmark": "navsim", "split": "navtest_dev", "scenes": len(tokens), "reference": "B10",
              "rule": "PDMS diff CI95 low >= -2.0, NC and DAC mean drop <= 0.5 (log-cluster bootstrap 2000, seed 17)",
              "candidates": out, "eligible": [m for m in out if out[m]["quality_eligible"]],
              "stored_2026_10_02_screening_reproduced": True, "utc": datetime.now(timezone.utc).isoformat()}
    write(REPORTS / "r5-nav-screen.json", report)
    print(json.dumps({"eligible": report["eligible"]}), flush=True)

def pai_screen():
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts"))
    import remote_pai_analyze as pa
    gts = pa.gt_targets("dev")
    base = pa.baselines("pai_dev_baselines_2026-10-04")
    for v in [0, 1, 2]:
        grid = pa.baselines(PAI_GRID.format(v))
        assert set(grid) == set(base), (v, len(grid))
        for c in base:
            base[c].update(grid[c])
    clips = sorted(set(gts) & set(base))
    assert len(clips) == len(gts) == len(base) == 1000
    idx = pd.read_parquet(ROOT / "data/physicalai/clip_index.parquet")
    clusters = {c: int(idx.loc[c, "chunk"]) for c in clips}
    b10 = {c: base[c]["B10-opt"] for c in clips}
    stored = json.loads((REPORTS / "pai-analysis-dev.json").read_text())["bstar_candidates"]
    out = {}
    for name in ["B10-opt", "B05-opt", "B03-opt", "B02-opt", "B01-opt"] + GRID:
        cmp = pa.compare({c: base[c][name] for c in clips}, b10, clusters, clips)
        out[name] = {"video_steps": steps(name)[0], "action_steps": steps(name)[1], "vs_B10": cmp,
                     "summary": pa.summary({c: base[c][name] for c in clips}, clips),
                     "eligible": bool(cmp["ade"]["hi"] <= pa.ADE_MARGIN and cmp["fde"]["hi"] <= pa.FDE_MARGIN)}
        if name in stored:
            for m in ["ade", "fde"]:
                for k in ["mean", "lo", "hi"]:
                    assert abs(stored[name]["vs_B10"][m][k] - cmp[m][k]) < 1e-9, (name, m, k)
            assert stored[name]["eligible"] == out[name]["eligible"], name
    report = {"benchmark": "physicalai", "split": "dev", "clips": len(clips), "clusters": len(set(clusters.values())),
              "reference": "B10-opt", "rule": "ADE diff CI95 high <= +0.05 m and FDE diff CI95 high <= +0.15 m (cluster bootstrap 2000)",
              "candidates": out, "eligible": [m for m in out if out[m]["eligible"]],
              "stored_2026_10_04_screening_reproduced": True, "utc": datetime.now(timezone.utc).isoformat()}
    write(REPORTS / "r5-pai-screen.json", report)
    print(json.dumps({"eligible": report["eligible"]}), flush=True)

def pct(values, q):
    v = sorted(values); k = (len(v) - 1) * q; f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)

def nav_latency(run_id):
    rows = [json.loads(line) for line in (ROOT / "runs" / run_id / "timings.jsonl").read_text().splitlines()]
    cell = {(r["model"], r["scene_token"], r["repeat"]): r["policy_ms"] for r in rows}
    models = list(dict.fromkeys(r["model"] for r in rows)); reps = sorted({r["repeat"] for r in rows})
    stats = {m: {"p50": pct([r["policy_ms"] for r in rows if r["model"] == m], .5),
                 "p95": pct([r["policy_ms"] for r in rows if r["model"] == m], .95)} for m in models}
    def paired(m, ref):
        per = {rep: statistics.median(cell[(m, t, rep)] / cell[(ref, t, rep)] for (mm, t, r) in cell
                                      if mm == m and r == rep) for rep in reps}
        lo, hi = min(per.values()), max(per.values())
        return {"median_paired_ratio": statistics.median(cell[k] / cell[(ref, k[1], k[2])] for k in cell if k[0] == m),
                "repeat_median_ratios": per, "speed_gate_0.80": "pass" if hi <= .8 else "fail" if lo > .8 else "판정 보류"}
    return stats, paired, {"host_load1_median": statistics.median(r["host_load1"] for r in rows),
                           "host_load1_max": max(r["host_load1"] for r in rows)}

def select():
    nav = json.loads((REPORTS / "r5-nav-screen.json").read_text())
    pai = json.loads((REPORTS / "r5-pai-screen.json").read_text())
    report = {"preregistration": "reports/사전등록_보완실험_2026-11-01.md §1.4", "utc": datetime.now(timezone.utc).isoformat()}
    stats, paired, load = nav_latency("r5_navsim_latency_2026-11-01")
    ok = [m for m in nav["eligible"] if m in stats]
    assert set(nav["eligible"]) <= set(stats), set(nav["eligible"]) - set(stats)
    bstar = min(ok, key=lambda m: (round(stats[m]["p50"], 3), -steps(m)[1], -steps(m)[0]))
    report["navsim"] = {"old_bstar": "B02-opt", "bstar": bstar, "changed": bstar != "B02-opt", "latency": stats, "host_load": load,
                        "student_vs_bstar": paired("student", bstar), "student_vs_B10": paired("student", "B10"),
                        "student_vs_old_bstar": paired("student", "B02-opt"),
                        "p95_ok": stats["student"]["p95"] <= stats[bstar]["p95"]}
    sys.path.insert(0, str(ROOT / "scripts"))
    from remote_pai_analyze import latency
    lat = latency("r5_pai_latency_2026-11-01")
    assert set(pai["eligible"]) <= set(lat), set(pai["eligible"]) - set(lat)
    bstar = min(pai["eligible"], key=lambda m: (round(lat[m]["decision_p50"], 3), -steps(m)[1], -steps(m)[0]))
    s = lat["S-first"]
    report["physicalai"] = {"old_bstar": "B03-opt", "bstar": bstar, "changed": bstar != "B03-opt", "latency": lat,
                            "student_ratio_p50": s["decision_p50"] / lat[bstar]["decision_p50"],
                            "speed_gate_0.80": s["decision_p50"] <= .8 * lat[bstar]["decision_p50"],
                            "p95_ok": s["decision_p95"] <= lat[bstar]["decision_p95"]}
    write(REPORTS / "r5-selection.json", report)
    print(json.dumps({b: {k: report[b][k] for k in ["bstar", "changed"]} for b in ["navsim", "physicalai"]}), flush=True)

def pai_context(set_name="dev"):
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts"))
    import remote_pai_analyze as pa
    ev, gts = pa.light_evaluator(), pa.gt_targets(set_name)
    clips = sorted(gts)
    assert len(clips) == 1000
    idx = pd.read_parquet(ROOT / "data/physicalai/clip_index.parquet")
    return pa, ev, gts, clips, {c: int(idx.loc[c, "chunk"]) for c in clips}

def head_rows(path):
    """{clip: [K, 40, 3]} normalized clip-frame head predictions (pred-*.npz)."""
    z = np.load(path); by = {}
    for pred, clip, chunk in zip(z["normalized"], z["clips"], z["chunks"]):
        by.setdefault(str(clip), {})[int(chunk)] = pred
    return {c: np.stack([v[k] for k in sorted(v)]) for c, v in by.items()}

def base_rows(run_id, model):
    """{clip: [K, 40, 3]} normalized clip-frame baseline chunks from a remote_pai_baselines run."""
    out = {}
    for f in (ROOT / "runs" / run_id).glob("shard*/*.json"):
        r = json.loads(f.read_text())
        out[r["clip"]] = np.asarray(r["models"][model]["pred_norm"], dtype=np.float64).transpose(1, 2, 0)
    return out

def pai_rms(a, b, clips):
    """RMS position difference (m) over all clip-chunk-steps; positions restart every chunk as the metric does."""
    from pai_heads import Q01, Q99
    lo, hi = np.asarray(Q01), np.asarray(Q99)
    xy = lambda n: np.cumsum((np.asarray(n, np.float64) + 1) / 2 * (hi - lo + 1e-6) + lo, axis=1)[..., :2]
    sq = np.concatenate([np.sum((xy(a[c]) - xy(b[c])) ** 2, axis=-1).ravel() for c in clips])
    return float(np.sqrt(sq.mean()))

def nav_rms(a, b):
    with np.load(a) as x, np.load(b) as y:
        assert (x["scene_tokens"] == y["scene_tokens"]).all()
        d = x["trajectories"][..., :2].astype(np.float64) - y["trajectories"][..., :2].astype(np.float64)
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=-1))))

def seed_diag():
    pa, ev, gts, clips, clusters = pai_context()
    gt = {c: [g for g, _ in gts[c]] for c in clips}
    report = {"preregistration": "§3", "utc": datetime.now(timezone.utc).isoformat(), "physicalai": {}, "navsim": {}}
    heads = ROOT / "runs/pai_heads_2026-10-04"
    for s in [17, 29, 43]:
        folder = heads / f"train20k-normal-first-seed{s}"
        rows = {0: head_rows(folder / "pred-dev.npz"), 1: head_rows(folder / "pred-dev-first-seed1.npz"),
                2: head_rows(folder / "pred-dev-first-seed2.npz")}
        met = {g: {c: pa.clip_metrics(ev, rows[g][c], gt[c]) for c in clips} for g in rows}
        e = {f"g{g}": pa.summary(met[g], clips) for g in rows}
        for g in [1, 2]:
            e[f"g{g}_minus_g0"] = pa.compare(met[g], met[0], clusters, clips)
            e[f"g{g}_rms_vs_g0_m"] = pai_rms(rows[g], rows[0], clips)
        e["within_margin"] = all(abs(e[f"g{g}_minus_g0"]["ade"]["mean"]) <= .02 and abs(e[f"g{g}_minus_g0"]["fde"]["mean"]) <= .06
                                 for g in [1, 2])
        report["physicalai"][f"student-seed{s}"] = e
    runs = {0: "pai_dev_baselines_2026-10-04", 1: "pai_dev_b10seed1_2026-10-04", 2: "pai_dev_b10seed2_2026-10-04",
            3: "pai_dev_b10seed3_2026-10-04"}
    rows = {g: base_rows(r, "B10-opt") for g, r in runs.items()}
    met = {}
    for g, r in runs.items():
        b = pa.baselines(r)
        met[g] = {c: b[c]["B10-opt"] for c in clips}
    e = {f"g{g}": pa.summary(met[g], clips) for g in runs}
    for g in [1, 2, 3]:
        e[f"g{g}_minus_g0"] = pa.compare(met[g], met[0], clusters, clips)
        e[f"g{g}_rms_vs_g0_m"] = pai_rms(rows[g], rows[0], clips)
    report["physicalai"]["B10-opt"] = e
    report["physicalai"]["noise_effect_small"] = all(report["physicalai"][f"student-seed{s}"]["within_margin"] for s in [17, 29, 43])
    tokens = (ROOT / "splits/navtest-dev-tokens.txt").read_text().splitlines()
    run = ROOT / "runs/r5_navsim_noise_seed_2026-11-01"
    old = ROOT / "runs/navtest_dev_quality_2026-10-02"; v3 = ROOT / "runs/heads_v3_2026-10-03"
    g0 = {"B10": (old / "scores/B10.csv", old / "predictions/B10.npz")}
    for s in [17, 29]:
        g0[f"student-seed{s}"] = (v3 / f"scores/full-R-first-seed{s}.csv", v3 / f"full-R-first-seed{s}/navtest-dev-predictions.npz")
    for name, (score0, pred0) in g0.items():
        base = nav_load(score0)
        e = {"g0": {m: float(np.mean([base[t][m] for t in tokens])) for m in QUALITY[:3] + ["ade_4s_m"]}}
        for g in [1, 2]:
            other = nav_load(run / f"scores/{name}@g{g}.csv")
            e[f"g{g}"] = {m: float(np.mean([other[t][m] for t in tokens])) for m in QUALITY[:3] + ["ade_4s_m"]}
            e[f"g{g}_minus_g0"] = cluster_bootstrap(other, base, tokens)
            e[f"g{g}_rms_vs_g0_m"] = nav_rms(run / f"predictions/{name}@g{g}.npz", pred0)
        if name != "B10":
            e["within_margin"] = all(abs(e[f"g{g}_minus_g0"]["score"]["mean"]) <= .5 for g in [1, 2])
        report["navsim"][name] = e
    report["navsim"]["noise_effect_small"] = all(report["navsim"][f"student-seed{s}"]["within_margin"] for s in [17, 29])
    write(REPORTS / "r5-seed-diag.json", report)
    print(json.dumps({b: report[b]["noise_effect_small"] for b in ["physicalai", "navsim"]}), flush=True)

def phase_diag():
    pa, ev, gts, clips, clusters = pai_context()
    heads = ROOT / "runs/pai_heads_2026-10-04"
    met = lambda name: pa.student(heads / name, "dev", ev, gts)
    report = {"preregistration": "§2", "utc": datetime.now(timezone.utc).isoformat(), "seeds": {}}
    cv = pa.cv_reference(ev, gts)
    report["cv"] = pa.summary(cv, clips)
    for s in [17, 29, 43]:
        first, cur, ego = (met(f"train20k-normal-{k}-seed{s}") for k in ["first", "current", "egoonly"])
        first4, cur4 = met(f"train4k-normal-first-seed{s}"), met(f"train4k-normal-current-seed{s}")
        report["seeds"][s] = {"first": pa.summary(first, clips), "current": pa.summary(cur, clips), "egoonly": pa.summary(ego, clips),
            "current_minus_first": pa.compare(cur, first, clusters, clips),
            "egoonly_minus_first": pa.compare(ego, first, clusters, clips),
            "egoonly_minus_cv": pa.compare(ego, cv, clusters, clips),
            "train4k_current_minus_first": pa.compare(cur4, first4, clusters, clips)}
    cm = [report["seeds"][s]["current_minus_first"] for s in [17, 29, 43]]
    keep = all(c["ade"]["lo"] > 0 for c in cm)
    equiv = all(c["ade"]["lo"] >= -.05 and c["ade"]["hi"] <= .05 and c["fde"]["lo"] >= -.15 and c["fde"]["hi"] <= .15 for c in cm)
    report["verdict"] = "first 우위 유지" if keep else "실질 동등" if equiv else "판단 보류"
    write(REPORTS / "r5-phase-diag.json", report)
    print(json.dumps({"verdict": report["verdict"]}, ensure_ascii=False), flush=True)

def changed(bench):
    sel = json.loads((REPORTS / "r5-selection.json").read_text())[bench]
    if not sel["changed"]:
        sys.exit(3)
    v, k = steps(sel["bstar"])
    print(f"V{v}-A{k}" if bench == "navsim" else sel["bstar"])

def nav_final():
    sel = json.loads((REPORTS / "r5-selection.json").read_text())["navsim"]
    bstar = sel["bstar"]
    tokens = (ROOT / "splits/navtest-final-tokens.txt").read_text().splitlines()
    old = ROOT / "runs/navtest_final_2026-10-03"
    s = {n: nav_load(old / f"scores/{n}.csv", 8896) for n in ["B10", "B02-opt", "student-seed17", "student-seed29"]}
    if bstar not in s:
        v, k = steps(bstar)
        s[bstar] = nav_load(ROOT / f"runs/r5_navsim_final_bstar_2026-11-01/scores/V{v}-A{k}.csv", 8896)
    stats, paired, load = nav_latency("r5_navsim_latency_2026-11-01")
    ratio = paired("student", bstar)
    out = {"preregistration": "§1.4 with navtest §8", "bstar": bstar, "bstar_changed": sel["changed"],
           "means": {n: {m: float(np.mean([f[t][m] for t in tokens])) for m in QUALITY + ERRORS} for n, f in s.items()},
           "paired": {}, "dac_fail_scenes": {n: int(sum(f[t]["drivable_area_compliance"] == 0 for t in tokens)) for n, f in s.items()}}
    for n in ["student-seed17", "student-seed29", bstar]:
        for ref in ["B10", bstar]:
            if n != ref:
                out["paired"][f"{n} - {ref}"] = cluster_bootstrap(s[n], s[ref], tokens)
    verdict = {}
    for seed in [17, 29]:
        checks = {}
        for ref in ["B10", bstar]:
            d = out["paired"][f"student-seed{seed} - {ref}"]
            checks[f"pdms_ci_low_vs_{ref}>=-2"] = d["score"]["ci95_low"] >= -2.0
            checks[f"nc_drop_vs_{ref}<=0.5"] = -d["no_at_fault_collisions"]["mean"] <= .5
            checks[f"dac_drop_vs_{ref}<=0.5"] = -d["drivable_area_compliance"]["mean"] <= .5
        checks["p50_ratio_vs_B*<=0.80"] = ratio["speed_gate_0.80"] == "pass"
        verdict[f"seed{seed}"] = {"checks": checks, "pass": all(checks.values())}
    out.update(verdict=verdict, final_pass=all(v["pass"] for v in verdict.values()), latency_ratio_vs_Bstar=ratio,
               latency=stats, host_load=load, utc=datetime.now(timezone.utc).isoformat())
    write(REPORTS / "r5-nav-final.json", out)
    print(json.dumps({"bstar": bstar, "final_pass": out["final_pass"], "verdict": verdict}, ensure_ascii=False), flush=True)

def pai_final():
    sel = json.loads((REPORTS / "r5-selection.json").read_text())["physicalai"]
    bstar = sel["bstar"]
    pa, ev, gts, clips, clusters = pai_context("final")
    base = pa.baselines("pai_final_baselines_2026-10-04")
    if bstar not in next(iter(base.values())):
        extra = pa.baselines("pai_final_bstar_2026-11-01")
        assert set(extra) == set(base)
        for c in base:
            base[c].update(extra[c])
    assert set(clips) == set(base)
    lat = pa.latency("r5_pai_latency_2026-11-01")
    b10, bs = {c: base[c]["B10-opt"] for c in clips}, {c: base[c][bstar] for c in clips}
    out = {"preregistration": "§1.4 with PhysicalAI §6", "bstar": bstar, "bstar_changed": sel["changed"],
           "baselines": {m: pa.summary({c: base[c][m] for c in clips}, clips) for m in ["B10-opt", bstar]},
           "latency": lat, "students": {}, "utc": datetime.now(timezone.utc).isoformat()}
    for s in [17, 29, 43]:
        st = pa.student(ROOT / f"runs/pai_heads_2026-10-04/train20k-normal-first-seed{s}", "final", ev, gts)
        cmp10, cmpb = pa.compare(st, b10, clusters, clips), pa.compare(st, bs, clusters, clips)
        out["students"][f"seed{s}"] = {"summary": pa.summary(st, clips), "vs_B10": cmp10, "vs_Bstar": cmpb,
                                       "verdict": pa.verdict(cmp10, cmpb, lat, bstar)}
    out["final_pass"] = all(v["verdict"]["pass"] for v in out["students"].values())
    write(REPORTS / "r5-pai-final.json", out)
    print(json.dumps({"bstar": bstar, "final_pass": out["final_pass"]}), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["assemble", "nav-screen", "pai-screen", "select", "seed-diag", "phase-diag", "changed",
                                     "nav-final", "pai-final"])
    p.add_argument("--run"); p.add_argument("--split", default="dev"); p.add_argument("--bench")
    a = p.parse_args()
    if a.stage == "assemble":
        assemble(a.run, a.split)
    elif a.stage == "changed":
        changed(a.bench)
    else:
        {"nav-screen": nav_screen, "pai-screen": pai_screen, "select": select, "seed-diag": seed_diag,
         "phase-diag": phase_diag, "nav-final": nav_final, "pai-final": pai_final}[a.stage]()
