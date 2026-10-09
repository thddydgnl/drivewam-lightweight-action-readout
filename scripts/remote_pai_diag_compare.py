"""Dev diagnostics (pre-registered, not in the verdict): 4k heads per variant vs the 4k first head.

Variants: phase current / terminal (imagination steps), nohist (memory cut), fixedprompt (no
guidance), clipframe (originally registered output frame). For each seed, paired NVIDIA-chunk
bootstrap of (variant - first) on ADE/FDE, plus the chunk FDE>2m rate difference. CPU only.
Usage: remote_pai_diag_compare.py [variant ...]
"""
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "scripts"))
import remote_pai_analyze as an                                   # noqa: E402

NAMES = {"current": "train4k-normal-current", "terminal": "train4k-normal-terminal", "nohist": "train4k-nohist-first",
         "fixedprompt": "train4k-fixedprompt-first", "clipframe": "train4k-normal-first-clipframe"}

if __name__ == "__main__":
    variants = sys.argv[1:] or list(NAMES)
    ev = an.light_evaluator()
    gts = an.gt_targets("dev")
    clips = sorted(gts)
    idx = pd.read_parquet(ROOT / "data/physicalai/clip_index.parquet")
    clusters = {c: int(idx.loc[c, "chunk"]) for c in clips}
    heads = ROOT / "runs/pai_heads_2026-10-04"
    base = an.baselines("pai_dev_baselines_2026-10-04")
    b10 = {c: base[c]["B10-opt"] for c in clips}
    ref = {s: an.student(heads / f"train4k-normal-first-seed{s}", "dev", ev, gts) for s in an.SEEDS}
    out = {"reference": "train4k-normal-first", "first": {s: an.summary(ref[s], clips) for s in an.SEEDS}, "variants": {}}
    for v in variants:
        res = {}
        for s in an.SEEDS:
            folder = heads / f"{NAMES[v]}-seed{s}"
            if not (folder / "pred-dev.npz").exists():
                continue
            st = an.student(folder, "dev", ev, gts)
            res[s] = {"summary": an.summary(st, clips), "minus_first": an.compare(st, ref[s], clusters, clips),
                      "minus_B10": an.compare(st, b10, clusters, clips)}
        out["variants"][v] = res
        for s, r in res.items():
            d = r["minus_first"]
            print(f"{v:12s} seed{s} ADE {r['summary']['ade']:.4f} FDE {r['summary']['fde']:.4f} | vs first dADE "
                  f"{d['ade']['mean']:+.4f} [{d['ade']['lo']:+.4f},{d['ade']['hi']:+.4f}] dFDE {d['fde']['mean']:+.4f} "
                  f"[{d['fde']['lo']:+.4f},{d['fde']['hi']:+.4f}] rate {d['fde2m_rate_pp']:+.2f}pp", flush=True)
    for s in an.SEEDS:
        print(f"first        seed{s} ADE {out['first'][s]['ade']:.4f} FDE {out['first'][s]['fde']:.4f}")
    path = ROOT / "reports/pai-diag-dev.json"
    old = json.loads(path.read_text()) if path.exists() else {"variants": {}}
    old["variants"].update(out["variants"]); old["first"] = out["first"]; old["reference"] = out["reference"]
    path.write_text(json.dumps(old, indent=2) + "\n")
