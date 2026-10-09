"""Render PhysicalAI result tables (markdown) from the synced analysis reports.

Usage: python3 make_pai_report.py  ->  evidence/physicalai_2026-10-04/tables.md
"""
import json
from pathlib import Path

EV = Path(__file__).resolve().parent / "evidence/physicalai_2026-10-04/reports"
OUT = EV.parent / "tables.md"
MODELS = ["B10-opt", "B05-opt", "B03-opt", "B02-opt", "B01-opt"]

def ci(d, k):
    x = d[k]
    return f"{x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"

def load(name):
    p = EV / name
    return json.loads(p.read_text()) if p.exists() else None

def baseline_table(r, title):
    lat = r["latency"]
    lines = [f"### {title} — 기준 모델 ({r['clips']} clip, {r['clusters']} NVIDIA chunk)", "",
             "| 모델 | ADE (m) | FDE (m) | chunk FDE>2m (%) | 결정 지연 p50 (ms) | p95 (ms) |", "|---|---:|---:|---:|---:|---:|"]
    for m in MODELS:
        b = r["baselines"][m]; l = lat.get(m, {})
        lines.append(f"| {m} | {b['ade']:.3f} | {b['fde']:.3f} | {b['fde2m_rate']:.1f} | "
                     f"{l.get('decision_p50', float('nan')):.1f} | {l.get('decision_p95', float('nan')):.1f} |")
    if "S-first" in lat:
        l = lat["S-first"]
        lines.append(f"| 학생 (first) | — | — | — | {l['decision_p50']:.1f} | {l['decision_p95']:.1f} |")
    c = r["cv"]
    lines.append(f"| CV (참고) | {c['ade']:.3f} | {c['fde']:.3f} | {c['fde2m_rate']:.1f} | — | — |")
    return lines

def bstar_table(r):
    lines = ["", f"### B\\* 선정 (dev) → **{r['bstar']}**", "",
             "| 후보 | ΔADE vs B10 [95% CI] | ΔFDE vs B10 [95% CI] | FDE>2m 증가 (pp) | 자격 | p50 (ms) |",
             "|---|---|---|---:|:---:|---:|"]
    for m, v in r["bstar_candidates"].items():
        lines.append(f"| {m} | {ci(v['vs_B10'], 'ade')} | {ci(v['vs_B10'], 'fde')} | {v['vs_B10']['fde2m_rate_pp']:+.2f} | "
                     f"{'O' if v['eligible'] else 'X'} | {v['decision_p50']:.1f} |")
    return lines

def student_table(r, names, title):
    lines = ["", f"### {title}", "",
             "| head | ADE | FDE | ΔADE vs B10 | ΔFDE vs B10 | ΔADE vs B\\* | ΔFDE vs B\\* | FDE>2m 증가 (pp) | 판정 |",
             "|---|---:|---:|---|---|---|---|---:|:---:|"]
    for n in names:
        s = r["students"].get(n)
        if not s:
            continue
        v = s["verdict"]
        mark = "—" if v is None else ("통과" if v["pass"] else "미달: " + ", ".join(k for k, ok in v["checks"].items() if not ok))
        lines.append(f"| {n} | {s['summary']['ade']:.3f} | {s['summary']['fde']:.3f} | {ci(s['vs_B10'], 'ade')} | "
                     f"{ci(s['vs_B10'], 'fde')} | {ci(s['vs_Bstar'], 'ade')} | {ci(s['vs_Bstar'], 'fde')} | "
                     f"{s['vs_B10']['fde2m_rate_pp']:+.2f} | {mark} |")
    return lines

def diag_table(r):
    groups = {}
    for n, s in r["students"].items():
        key = n.rsplit("-seed", 1)[0]
        groups.setdefault(key, []).append(s)
    lines = ["", "### dev 진단 (seed 평균)", "", "| 설정 | seed 수 | ADE | FDE | ΔADE vs B10 평균 | ΔFDE vs B10 평균 |", "|---|---:|---:|---:|---:|---:|"]
    for k in sorted(groups):
        g = groups[k]
        mean = lambda f: sum(f(s) for s in g) / len(g)
        lines.append(f"| {k} | {len(g)} | {mean(lambda s: s['summary']['ade']):.3f} | {mean(lambda s: s['summary']['fde']):.3f} | "
                     f"{mean(lambda s: s['vs_B10']['ade']['mean']):+.3f} | {mean(lambda s: s['vs_B10']['fde']['mean']):+.3f} |")
    return lines

if __name__ == "__main__":
    out = []
    dev, fin, diag = load("pai-analysis-dev.json"), load("pai-analysis-final.json"), load("pai-analysis-dev-diag.json")
    if dev:
        out += baseline_table(dev, "dev") + bstar_table(dev)
        out += ["", f"규모 선택: {dev.get('scale')} (seed 평균 dev ADE {dev.get('scale_means')})"]
        names = [n for n in dev["students"] if "-normal-first-seed" in n]
        out += student_table(dev, names, "dev 학생 (first, 회전 출력)")
        if "verdict" in dev:
            out += ["", f"**dev 판정({dev['verdict']['config']}): {'통과' if dev['verdict']['pass'] else '미달'}**"]
    if fin:
        rep = fin["reproduction"]
        out += ["", "---", ""] + baseline_table(fin, "final")
        out += ["", f"B10 재현: ADE {rep['B10_ade']:.3f} (논문 {rep['paper_ade']}), FDE {rep['B10_fde']:.3f} (논문 {rep['paper_fde']})"]
        out += student_table(fin, sorted(fin["students"]), f"final 학생 ({fin['scale']}, B\\*={fin['bstar']})")
        if "verdict" in fin:
            out += ["", f"**final 판정: {'통과' if fin['verdict']['pass'] else '미달'}**"]
    if diag:
        out += ["", "---"] + diag_table(diag)
    OUT.write_text("\n".join(out) + "\n")
    print(OUT.read_text())
