"""Post-hoc control (not pre-registered): is the student's accuracy edge over B10 just determinism?

B10-opt is run on dev with seeds 0..3 (seed 0 = official run). For K = 1..4 the first K samples are
averaged in normalized delta space (positions are cumulative sums, so this is the mean trajectory)
and scored with the official ADE/FDE. The 20k students are compared with the K-sample mean using
the same NVIDIA-chunk paired bootstrap. ADE(K) ~ a + b / K estimates the infinite-sample mean.
CPU only.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "scripts"))
import remote_pai_analyze as an                                   # noqa: E402

RUNS = {0: "pai_dev_baselines_2026-10-04", 1: "pai_dev_b10seed1_2026-10-04", 2: "pai_dev_b10seed2_2026-10-04",
        3: "pai_dev_b10seed3_2026-10-04"}

def preds(run_id):
    out = {}
    for f in (ROOT / "runs" / run_id).glob("shard*/*.json"):
        r = json.loads(f.read_text())
        out[r["clip"]] = np.asarray(r["models"]["B10-opt"]["pred_norm"], dtype=np.float64)   # [3, K, 40]
    return out

if __name__ == "__main__":
    ev = an.light_evaluator()
    gts = an.gt_targets("dev")
    clips = sorted(gts)
    idx = pd.read_parquet(ROOT / "data/physicalai/clip_index.parquet")
    clusters = {c: int(idx.loc[c, "chunk"]) for c in clips}
    samples = {s: preds(r) for s, r in RUNS.items()}
    for s, d in samples.items():
        assert set(d) == set(clips), (s, len(d))
    rows = lambda arr: [arr[:, k, :].T for k in range(arr.shape[1])]
    single, mean_k = {}, {}
    for s in RUNS:
        single[s] = {c: an.clip_metrics(ev, rows(samples[s][c]), [g for g, _ in gts[c]]) for c in clips}
    for K in range(1, len(RUNS) + 1):
        mean_k[K] = {c: an.clip_metrics(ev, rows(np.mean([samples[s][c] for s in range(K)], axis=0)),
                                        [g for g, _ in gts[c]]) for c in clips}
    spread = float(np.mean([np.abs(samples[1][c] - samples[0][c]).mean() for c in clips]))
    curve = {K: an.summary(mean_k[K], clips) for K in mean_k}
    ks = np.array(sorted(curve)); ade = np.array([curve[k]["ade"] for k in ks]); fde = np.array([curve[k]["fde"] for k in ks])
    A = np.stack([np.ones_like(ks, dtype=float), 1.0 / ks], axis=1)
    fit_ade, fit_fde = np.linalg.lstsq(A, ade, rcond=None)[0], np.linalg.lstsq(A, fde, rcond=None)[0]
    report = {"utc": datetime.now(timezone.utc).isoformat(), "clips": len(clips), "post_hoc": True,
              "single_seed": {s: an.summary(single[s], clips) for s in single},
              "mean_of_k": {int(k): v for k, v in curve.items()},
              "fit": {"ade_inf": float(fit_ade[0]), "ade_b": float(fit_ade[1]), "fde_inf": float(fit_fde[0]), "fde_b": float(fit_fde[1])},
              "mean_abs_normalized_seed_spread": spread, "students": {}}
    heads = ROOT / "runs/pai_heads_2026-10-04"
    for seed in an.SEEDS:
        folder = heads / f"train20k-normal-first-seed{seed}"
        st = an.student(folder, "dev", ev, gts)
        report["students"][folder.name] = {
            "summary": an.summary(st, clips),
            "vs_B10_k1": an.compare(st, mean_k[1], clusters, clips),
            "vs_B10_mean4": an.compare(st, mean_k[len(RUNS)], clusters, clips)}
    (ROOT / "reports/pai-ensemble-dev.json").write_text(json.dumps(report, indent=2) + "\n")
    print("single-seed B10 ADE:", {s: round(v["ade"], 4) for s, v in report["single_seed"].items()})
    print("mean-of-K B10:", {k: (round(v["ade"], 4), round(v["fde"], 4), round(v["fde2m_rate"], 2)) for k, v in report["mean_of_k"].items()})
    print("fit ADE(K)=a+b/K:", {k: round(v, 4) for k, v in report["fit"].items()}, "seed spread", round(spread, 4))
    for n, v in report["students"].items():
        c1, c4 = v["vs_B10_k1"], v["vs_B10_mean4"]
        print(n, "ADE %.4f" % v["summary"]["ade"], "| vs K=1 dADE %+.3f [%+.3f,%+.3f]" % (c1["ade"]["mean"], c1["ade"]["lo"], c1["ade"]["hi"]),
              "| vs mean4 dADE %+.3f [%+.3f,%+.3f] dFDE %+.3f [%+.3f,%+.3f]" % (c4["ade"]["mean"], c4["ade"]["lo"], c4["ade"]["hi"],
                                                                               c4["fde"]["mean"], c4["fde"]["lo"], c4["fde"]["hi"]))
