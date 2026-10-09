"""PhysicalAI dev/final analysis (pre-registration 사전등록_PhysicalAI_2026-10-04 §4–§6).

dev  : B* selection (quality CIs vs B10 + decision latency), data-scale choice (4k/20k),
       per-seed student verdict, diagnostics (phase, history cut, fixed prompt), CV reference.
final: B10 reproduction, student (config fixed on dev) vs B10 and B*, verdict.
Student metrics use the upstream ADE/FDE code on assembled action tensors; uncertainty is a
paired bootstrap over NVIDIA data chunks (2,000 resamples). CPU only.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "scripts"))
SRC = ROOT / "work/DriveWAM/src"
sys.path.insert(0, str(SRC))
import runpy                                                       # noqa: E402
UP = runpy.run_path(str(SRC / "physicalai/eval.py"), run_name="pai_frozen_evaluator")
from configs import VA_CONFIGS                                     # noqa: E402

CANDIDATES = ["B10-opt", "B05-opt", "B03-opt", "B02-opt", "B01-opt"]
STEPS = {"B10-opt": 10, "B05-opt": 5, "B03-opt": 3, "B02-opt": 2, "B01-opt": 1}
SEEDS = [17, 29, 43]
ADE_MARGIN, FDE_MARGIN, RATE_MARGIN, LAT_RATIO = 0.05, 0.15, 1.0, 0.80

def light_evaluator():
    cfg = VA_CONFIGS["physicalai_cfg"]
    ev = UP["PhysicalAIEvaluator"].__new__(UP["PhysicalAIEvaluator"])
    ev.config, ev.action_chunk_steps = cfg, cfg.action_chunk_steps
    ev.q01 = torch.tensor(cfg.norm_stat["q01"][:2], dtype=torch.float32)
    ev.q99 = torch.tensor(cfg.norm_stat["q99"][:2], dtype=torch.float32)
    return ev

def clip_metrics(ev, pred_rows, gt_rows):
    """pred_rows/gt_rows: [K, 40, 3] normalized (chunks 1..K) -> upstream ADE/FDE + chunk FDE."""
    K = len(gt_rows)
    pa = torch.zeros(1, 30, K + 1, 40, 1); ga = torch.zeros(1, 30, K + 1, 40, 1)
    pa[0, :3, 1:, :, 0] = torch.as_tensor(np.asarray(pred_rows, dtype=np.float32)).permute(2, 0, 1)
    ga[0, :3, 1:, :, 0] = torch.as_tensor(np.asarray(gt_rows, dtype=np.float32)).permute(2, 0, 1)
    ade, fde = ev.compute_ade_fde(pa, ga)
    pred, gt = ev.actions_to_trajectory_with_reset(pa, ga)
    l2 = np.sqrt(((pred[0] - gt[0]) ** 2).sum(axis=0))
    return {"ade": ade, "fde": fde, "chunk_fde": [float(l2[(c + 1) * 40 - 1]) for c in range(K)]}

def gt_targets(set_name):
    """GT normalized chunks and ego rows per clip from the eval set's normal feature rows."""
    rows = {}
    for f in sorted((ROOT / "caches/pai-features" / f"{set_name}-normal").glob("shard*/*.safetensors")):
        meta = json.loads(Path(str(f) + ".json").read_text())
        with safe_open(str(f), framework="pt", device="cpu") as s:
            tgt, ego = s.get_tensor("target").numpy(), s.get_tensor("ego").numpy()
        for j, r in enumerate(meta["rows"]):
            rows.setdefault(r["clip"], {})[r["chunk"]] = (tgt[j], ego[j])
    return {c: [v[k] for k in sorted(v)] for c, v in rows.items()}

def baselines(run_id):
    out = {}
    for f in (ROOT / "runs" / run_id).glob("shard*/*.json"):
        r = json.loads(f.read_text())
        out[r["clip"]] = {m: {k: v[k] for k in ["ade", "fde", "chunk_fde"]} for m, v in r["models"].items()}
    return out

def student(folder, set_name, ev, gts):
    z = np.load(Path(folder) / f"pred-{set_name}.npz")
    by = {}
    for pred, clip, chunk in zip(z["normalized"], z["clips"], z["chunks"]):
        by.setdefault(str(clip), {})[int(chunk)] = pred
    return {c: clip_metrics(ev, [by[c][k] for k in sorted(by[c])], [g for g, _ in gts[c]]) for c in gts}

def cv_reference(ev, gts):
    """Constant velocity at chunk start, rotated into the clip start frame by the GT heading so far."""
    cfg = ev.config; lo, hi = np.asarray(cfg.norm_stat["q01"][:3]), np.asarray(cfg.norm_stat["q99"][:3])
    out = {}
    for c, rows in gts.items():
        preds, heading = [], 0.0
        for g, ego in rows:
            vx, vy = float(ego[0]), float(ego[1])
            d = np.array([0.1 * (np.cos(heading) * vx - np.sin(heading) * vy),
                          0.1 * (np.sin(heading) * vx + np.cos(heading) * vy), 0.0])
            preds.append(np.repeat(((d - lo) / (hi - lo + 1e-6) * 2 - 1)[None], 40, axis=0))
            heading += float(((g[:, 2] + 1) / 2 * (hi[2] - lo[2] + 1e-6) + lo[2]).sum())
        out[c] = clip_metrics(ev, preds, [g for g, _ in rows])
    return out

def boot(diff_by_clip, cluster_of, n=2000, seed=20261004):
    clips = sorted(diff_by_clip)
    groups = {}
    for c in clips:
        groups.setdefault(cluster_of[c], []).append(diff_by_clip[c])
    keys = sorted(groups)
    sums = np.array([sum(groups[k]) for k in keys]); counts = np.array([len(groups[k]) for k in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(n, len(keys)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return {"mean": float(np.mean([diff_by_clip[c] for c in clips])), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "clips": len(clips), "clusters": len(keys)}

def compare(a, b, clusters, clips):
    out = {}
    for m in ["ade", "fde"]:
        out[m] = boot({c: a[c][m] - b[c][m] for c in clips}, clusters)
    rate = lambda x: 100.0 * np.mean([v > 2.0 for c in clips for v in x[c]["chunk_fde"]])
    out["fde2m_rate_pp"] = float(rate(a) - rate(b))
    return out

def summary(x, clips):
    return {"ade": float(np.mean([x[c]["ade"] for c in clips])), "fde": float(np.mean([x[c]["fde"] for c in clips])),
            "fde2m_rate": float(100.0 * np.mean([v > 2.0 for c in clips for v in x[c]["chunk_fde"]]))}

def latency(run_id):
    vals = {}
    for f in (ROOT / "runs" / run_id).glob("shard*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line); vals.setdefault(r["model"], {"d": [], "t": []})
            vals[r["model"]]["d"].append(r["decision_ms"]); vals[r["model"]]["t"].append(r["decision_ms"] + r["register_ms"])
    return {m: {"decision_p50": float(np.percentile(v["d"], 50)), "decision_p95": float(np.percentile(v["d"], 95)),
                "with_register_p50": float(np.percentile(v["t"], 50)), "n": len(v["d"])} for m, v in vals.items()}

def verdict(cmp_b10, cmp_bstar, lat, bstar):
    s = lat["S-first"]; b = lat[bstar]
    checks = {"ade_vs_B10": cmp_b10["ade"]["hi"] <= ADE_MARGIN, "fde_vs_B10": cmp_b10["fde"]["hi"] <= FDE_MARGIN,
              "ade_vs_Bstar": cmp_bstar["ade"]["hi"] <= ADE_MARGIN, "fde_vs_Bstar": cmp_bstar["fde"]["hi"] <= FDE_MARGIN,
              "fde2m_rate": cmp_b10["fde2m_rate_pp"] <= RATE_MARGIN,
              "latency_p50": s["decision_p50"] <= LAT_RATIO * b["decision_p50"],
              "latency_p95": s["decision_p95"] <= b["decision_p95"]}
    checks = {k: bool(v) for k, v in checks.items()}
    return {"checks": checks, "pass": bool(all(checks.values()))}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True, choices=["dev", "final"])
    p.add_argument("--baselines", required=True)
    p.add_argument("--heads", default="pai_heads_2026-10-04")
    p.add_argument("--latency", required=True)
    p.add_argument("--tag", default="", help="suffix for a non-canonical report (e.g. diagnostics)")
    a = p.parse_args()
    ev = light_evaluator()
    idx = pd.read_parquet(ROOT / "data/physicalai/clip_index.parquet")
    gts = gt_targets(a.set)
    base = baselines(a.baselines)
    clips = sorted(set(gts) & set(base))
    assert len(clips) == len(gts) == len(base), (len(clips), len(gts), len(base))
    clusters = {c: int(idx.loc[c, "chunk"]) for c in clips}
    lat = latency(a.latency)
    b10 = {c: base[c]["B10-opt"] for c in clips}
    report = {"set": a.set, "utc": datetime.now(timezone.utc).isoformat(), "clips": len(clips),
              "clusters": len(set(clusters.values())), "latency": lat,
              "baselines": {m: summary({c: base[c][m] for c in clips}, clips) for m in CANDIDATES},
              "cv": summary(cv_reference(ev, gts), clips)}
    if a.set == "dev":
        elig = {}
        for m in CANDIDATES:
            cmp = compare({c: base[c][m] for c in clips}, b10, clusters, clips)
            elig[m] = {"vs_B10": cmp, "eligible": bool(cmp["ade"]["hi"] <= ADE_MARGIN and cmp["fde"]["hi"] <= FDE_MARGIN),
                       "decision_p50": lat[m]["decision_p50"]}
        ok = [m for m in CANDIDATES if elig[m]["eligible"]]
        bstar = min(ok, key=lambda m: (round(lat[m]["decision_p50"], 3), -STEPS[m]))
        report["bstar_candidates"], report["bstar"] = elig, bstar
    else:
        dev = json.loads((ROOT / "reports/pai-analysis-dev.json").read_text())
        bstar, scale = dev["bstar"], dev["scale"]
        lat = dev["latency"]; report["latency_from"] = "dev"; report["bstar"], report["scale"] = bstar, scale
        report["reproduction"] = {"B10_ade": report["baselines"]["B10-opt"]["ade"], "B10_fde": report["baselines"]["B10-opt"]["fde"],
                                  "paper_ade": 0.83, "paper_fde": 2.47}
    bs = {c: base[c][bstar] for c in clips}
    heads = ROOT / "runs" / a.heads
    students = {}
    for folder in sorted(heads.iterdir()):
        if not (folder / f"pred-{a.set}.npz").exists():
            continue
        st = student(folder, a.set, ev, gts)
        cmp10, cmpb = compare(st, b10, clusters, clips), compare(st, bs, clusters, clips)
        students[folder.name] = {"summary": summary(st, clips), "vs_B10": cmp10, "vs_Bstar": cmpb,
                                 "verdict": verdict(cmp10, cmpb, lat, bstar) if "S-first" in lat else None}
    report["students"] = students
    if a.set == "dev":
        means = {}
        for scale in ["train4k", "train20k"]:
            names = [f"{scale}-normal-first-seed{s}" for s in SEEDS]
            if all(n in students for n in names):
                means[scale] = float(np.mean([students[n]["summary"]["ade"] for n in names]))
        if len(means) == 2:
            report["scale"] = "train20k" if means["train4k"] - means["train20k"] > -0.01 else "train4k"
        report["scale_means"] = means
    scale = report.get("scale")
    if scale:
        names = [f"{scale}-normal-first-seed{s}" for s in SEEDS]
        if all(n in students and students[n]["verdict"] for n in names):
            report["verdict"] = {"config": scale, "per_seed": {n: students[n]["verdict"] for n in names},
                                 "pass": bool(all(students[n]["verdict"]["pass"] for n in names))}
    out = ROOT / "reports" / f"pai-analysis-{a.set}{('-' + a.tag) if a.tag else ''}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ["set", "clips", "clusters", "bstar", "scale", "scale_means"]}))
    print(json.dumps(report["baselines"], indent=1))
    for n, v in students.items():
        print(n, json.dumps(v["summary"]), "dADE", round(v["vs_B10"]["ade"]["mean"], 4), "CIhi", round(v["vs_B10"]["ade"]["hi"], 4),
              "dFDE", round(v["vs_B10"]["fde"]["mean"], 4), "CIhi", round(v["vs_B10"]["fde"]["hi"], 4),
              "pass" if v["verdict"] and v["verdict"]["pass"] else "-")
    if "verdict" in report:
        print("VERDICT", json.dumps(report["verdict"]))
