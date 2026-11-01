# -*- coding: utf-8 -*-
"""English figures (PDF + PNG) for the JKAIA 2026 poster paper and the TMLR manuscript.

Adapted from docs/report/build/make_figs.py (Korean technical report). All numbers are
copied from evidence/ (see docs/통합_연구기록.md and evidence/r5_2026-11-01/ for the file
behind each value). v2 (2026-11-01): three-row overview, grid heatmap, 20k phase panel,
final-set latencies from the quiet-host session of 2026-11-01.

Usage:  python3 docs/paper/build/make_figs_en.py
Output: docs/paper/figures/*.pdf and *.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import NullLocator

SP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(SP), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "savefig.dpi": 220,
    "figure.dpi": 100,
    "pdf.fonttype": 42,
})

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#7a7974"; GRID = "#e4e3df"
BLUE = "#2a78d6"; BLUE_D = "#1c5cab"; BLUE_L = "#cde2fb"; BLUE_LL = "#9ec5f4"
ORANGE = "#eb6834"; ORANGE_D = "#b8431c"; ORANGE_L = "#fde3d7"
AQUA = "#1baf7a"; AQUA_D = "#0f7a55"; AQUA_L = "#d7f3e8"
GRAY_F = "#f0efec"; GRAY_E = "#b5b3ad"
ARROW = "#52514e"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc=BLUE_L, ec=BLUE_D, fs=8.5, weight="normal", tc=INK, ls="-", lw=1.0, r=0.07, ls_=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, color=tc, linespacing=ls_)


def arrow(ax, x1, y1, x2, y2, color=ARROW, lw=1.0, ls="-", ms=9):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms, shrinkA=0, shrinkB=0, linestyle=ls))


def line(ax, x1, y1, x2, y2, color=ARROW, lw=1.0, ls="-"):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, solid_capstyle="round")


def text(ax, x, y, s, fs=8, color=INK2, ha="center", va="center", weight="normal", ls_=1.2):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, fontweight=weight, linespacing=ls_)


def chain(ax, y, h, specs, x0=0.05, gap=0.1, fs=8.5):
    pos = []; x = x0
    for i, (w, t, fc, ec, kw) in enumerate(specs):
        box(ax, x, y, w, h, t, fc=fc, ec=ec, fs=kw.get("fs", fs), weight=kw.get("weight", "normal"),
            ls=kw.get("ls", "-"), tc=kw.get("tc", INK), lw=kw.get("lw", 1.0))
        pos.append((x, w))
        if i < len(specs) - 1:
            arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2, color=kw.get("ac", ARROW))
        x += w + gap
    return pos


def style_axes(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def bracket(ax, xa, xb, yb, label, color=BLUE_D, fs=7.6):
    line(ax, xa, yb, xb, yb, color=color, lw=1.1)
    line(ax, xa, yb, xa, yb + 0.07, color=color, lw=1.1)
    line(ax, xb, yb, xb, yb + 0.07, color=color, lw=1.1)
    text(ax, (xa + xb) / 2, yb - 0.13, label, fs=fs, color=color)


# ---------------------------------------------------------------------
# Figure 1: B10 vs the training-free V0-A2 vs the proposed readout
# ---------------------------------------------------------------------
def fig_overview():
    W, H = 7.2, 4.15
    fig, ax = canvas(W, H)
    h = 0.6; gap = 0.05; x0 = 0.06; fs = 7.0
    head = [(0.5, "Obs.\nframes", GRAY_F, GRAY_E, {}),
            (0.5, "Video\nVAE", GRAY_F, GRAY_E, {}),
            (0.7, "Obs. pass\n(register)", BLUE_L, BLUE_D, {})]
    # row 1: official B10
    y1 = 3.3
    text(ax, x0, y1 + h + 0.13, "DriveWAM, official inference (B10): 4 future passes, 10 action passes", fs=8.6, color=INK, ha="left", weight="bold")
    specs1 = head + [
        (0.76, "Future pass 1\n(σ = 1, noise)", BLUE_L, BLUE_D, {}),
        (0.6, "Future\npass 2", BLUE_L, BLUE_D, {}),
        (0.6, "Future\npass 3", BLUE_L, BLUE_D, {}),
        (0.78, "Cache pass\n(σ = 0.4)", BLUE_L, BLUE_D, {}),
        (1.07, "Action denoising\n× 10 (+1 cache)", BLUE_L, BLUE_D, {"weight": "bold"}),
        (0.62, "Action\ndecoder", GRAY_F, GRAY_E, {}),
        (0.5, "4 s\ntraj.", AQUA_L, AQUA_D, {}),
    ]
    chain(ax, y1, h, specs1, x0=x0, gap=gap, fs=fs)
    # row 2: training-free V0-A2
    y2 = 1.98
    text(ax, x0, y2 + h + 0.13, "Training-free variant V0-A2: future pass 1 writes the cache, 2 action passes (= B*, keeps B10 quality)", fs=8.6, color=INK, ha="left", weight="bold")
    specs2 = head + [
        (1.26, "Future pass 1 (σ = 1)\nwrites the KV cache", BLUE_L, BLUE_D, {"weight": "bold"}),
        (0.98, "Action denoising\n× 2", BLUE_L, BLUE_D, {"weight": "bold"}),
        (0.62, "Action\ndecoder", GRAY_F, GRAY_E, {}),
        (0.5, "4 s\ntraj.", AQUA_L, AQUA_D, {}),
    ]
    pos2 = chain(ax, y2, h, specs2, x0=x0, gap=gap, fs=fs)
    xg = pos2[-1][0] + pos2[-1][1] + 0.12
    box(ax, xg, y2, W - xg - 0.06, h, "skipped: future passes 2–3,\ncache pass at σ = 0.4,\n8 action passes", fc="white", ec=GRAY_E, ls="--", fs=6.8, tc=MUTED, ls_=1.15)
    # row 3: proposed readout
    y3 = 0.66
    text(ax, x0, y3 + h + 0.13, "Proposed readout (SF3-first): future pass 1 only, trajectory from a 10M-parameter head", fs=8.6, color=INK, ha="left", weight="bold")
    specs3 = head + [
        (1.02, "Future pass 1\n(σ = 1), hooked", ORANGE_L, ORANGE_D, {"weight": "bold", "lw": 1.4}),
        (0.96, "Blocks 10 / 20 / 30\nhidden states", ORANGE_L, ORANGE_D, {}),
        (1.0, "SF3 head\n(trained, 10M)", AQUA_L, AQUA_D, {"weight": "bold", "lw": 1.4}),
        (0.5, "4 s\ntraj.", AQUA_L, AQUA_D, {}),
    ]
    pos3 = chain(ax, y3, h, specs3, x0=x0, gap=gap, fs=fs)
    xg = pos3[-1][0] + pos3[-1][1] + 0.12
    box(ax, xg, y3, W - xg - 0.06, h, "skipped: all later\nvideo and action passes", fc="white", ec=GRAY_E, ls="--", fs=6.8, tc=MUTED, ls_=1.15)
    bracket(ax, pos3[2][0], pos3[3][0] + pos3[3][1], y3 - 0.1, "backbone frozen, no parameter update")
    text(ax, W / 2, 0.12, "Future pass 1 is the same forward in all three paths: the pure-noise future latent at σ = 1 attends to the registered observation.", fs=7.4, color=INK2)
    save(fig, "fig_overview")


# ---------------------------------------------------------------------
# Figure: compact overview for the 2-page JKAIA poster summary (unchanged)
# ---------------------------------------------------------------------
def fig_overview_poster():
    W, H = 7.2, 2.3
    fig, ax = canvas(W, H)
    h = 0.6
    y1 = 1.42
    text(ax, 0.06, y1 + h + 0.14, "DriveWAM, official inference (B10)", fs=10, color=INK, ha="left", weight="bold")
    specs1 = [
        (0.56, "Obs.\nframes", GRAY_F, GRAY_E, {}),
        (0.62, "Video\nVAE", GRAY_F, GRAY_E, {}),
        (0.8, "Obs. pass\n(register)", BLUE_L, BLUE_D, {}),
        (0.64, "Future\npass 1", BLUE_L, BLUE_D, {}),
        (0.64, "Future\npass 2", BLUE_L, BLUE_D, {}),
        (0.64, "Future\npass 3", BLUE_L, BLUE_D, {}),
        (1.26, "Action denoising\n× 10", BLUE_L, BLUE_D, {"weight": "bold"}),
        (0.68, "Action\ndecoder", GRAY_F, GRAY_E, {}),
        (0.5, "4 s\ntraj.", AQUA_L, AQUA_D, {}),
    ]
    chain(ax, y1, h, specs1, x0=0.06, gap=0.06, fs=9)
    y2 = 0.4
    text(ax, 0.06, y2 + h + 0.14, "Proposed: early exit after future pass 1 + SF3 readout head", fs=10, color=INK, ha="left", weight="bold")
    specs2 = [
        (0.56, "Obs.\nframes", GRAY_F, GRAY_E, {}),
        (0.62, "Video\nVAE", GRAY_F, GRAY_E, {}),
        (0.8, "Obs. pass\n(register)", BLUE_L, BLUE_D, {}),
        (0.64, "Future\npass 1", ORANGE_L, ORANGE_D, {"weight": "bold", "lw": 1.4}),
        (0.92, "Blocks\n10 / 20 / 30", ORANGE_L, ORANGE_D, {}),
        (1.1, "SF3 head\n(trained, 10M)", AQUA_L, AQUA_D, {"weight": "bold", "lw": 1.4}),
        (0.5, "4 s\ntraj.", AQUA_L, AQUA_D, {}),
    ]
    pos2 = chain(ax, y2, h, specs2, x0=0.06, gap=0.06, fs=9)
    xg = pos2[-1][0] + pos2[-1][1] + 0.1
    wg = W - xg - 0.04
    box(ax, xg, y2, wg, h, "skipped: future\npasses 2-3, action\ndenoising × 10", fc="white", ec=GRAY_E, ls="--", fs=8, tc=MUTED, ls_=1.1)
    xa = pos2[2][0]; xb = pos2[3][0] + pos2[3][1]
    yb = y2 - 0.08
    line(ax, xa, yb, xb, yb, color=BLUE_D, lw=1.1); line(ax, xa, yb, xa, yb + 0.06, color=BLUE_D, lw=1.1); line(ax, xb, yb, xb, yb + 0.06, color=BLUE_D, lw=1.1)
    text(ax, (xa + xb) / 2, yb - 0.15, "backbone frozen", fs=9, color=BLUE_D)
    save(fig, "fig_overview_poster")


# ---------------------------------------------------------------------
# Figure: readout phases (current / first / terminal)
# ---------------------------------------------------------------------
def fig_phases():
    W, H = 7.2, 2.1
    fig, ax = canvas(W, H)
    y, h = 1.12, 0.6
    specs = [
        (1.0, "Observation pass\n(register obs.)", BLUE_L, BLUE_D, {}),
        (1.42, "Future pass 1\n(input: pure noise, σ = 1)", ORANGE_L, ORANGE_D, {"weight": "bold"}),
        (0.78, "Future\npass 2", BLUE_L, BLUE_D, {}),
        (0.78, "Future\npass 3", BLUE_L, BLUE_D, {}),
        (1.12, "Cache pass (σ = 0.4)\nlatent → KV cache", BLUE_L, BLUE_D, {}),
        (1.1, "Action denoising\n× 10 → trajectory", GRAY_F, GRAY_E, {}),
    ]
    pos = chain(ax, y, h, specs, x0=0.08, gap=0.08, fs=7.0)
    text(ax, 0.08, y + h + 0.16, "Backbone passes for one 4 s chunk →", fs=8, color=MUTED, ha="left")
    marks = [(0, "current", "observation only,\nno future pass", BLUE_D),
             (1, "first", "one future pass\n(proposed readout)", ORANGE_D),
             (4, "terminal", "after all future-\nlatent updates", BLUE_D)]
    for i, name, desc, col in marks:
        x, w = pos[i]
        cx = x + w / 2
        arrow(ax, cx, y - 0.02, cx, y - 0.3, color=col, lw=1.2)
        text(ax, cx, y - 0.42, name, fs=9, color=col, weight="bold")
        text(ax, cx, y - 0.78, desc, fs=7.0, color=INK2)
    text(ax, W - 0.08, y + h + 0.16, "At each phase the head reads the block 10 / 20 / 30 outputs", fs=8, color=MUTED, ha="right")
    save(fig, "fig_phases")


# ---------------------------------------------------------------------
# Figure: SF3 head
# ---------------------------------------------------------------------
def fig_sf3():
    W, H = 7.2, 3.45
    fig, ax = canvas(W, H)
    lane_h = 0.5; gap = 0.14
    ys = [2.85, 2.85 - (lane_h + gap), 2.85 - 2 * (lane_h + gap)]
    names = ["block 10", "block 20", "block 30"]
    for yl, nm in zip(ys, names):
        box(ax, 0.08, yl, 1.1, lane_h, f"{nm} output\n112 tok. × 3072", fc=ORANGE_L, ec=ORANGE_D, fs=7.0)
        arrow(ax, 1.18, yl + lane_h / 2, 1.28, yl + lane_h / 2)
        box(ax, 1.28, yl, 2.22, lane_h, "Query pooling: LN → Linear 3072→512\n→ 16 learned queries, 8-head attn. → weighted sum", fc=AQUA_L, ec=AQUA_D, fs=6.3)
        arrow(ax, 3.5, yl + lane_h / 2, 3.58, yl + lane_h / 2)
        box(ax, 3.58, yl, 0.5, lane_h, "s (512)", fc=AQUA_L, ec=AQUA_D, fs=7.2)
    ye = ys[2] - (lane_h + gap)
    box(ax, 0.08, ye, 1.1, lane_h, "ego state\n(speed, accel., ...)", fc=GRAY_F, ec=GRAY_E, fs=7.0)
    arrow(ax, 1.18, ye + lane_h / 2, 1.28, ye + lane_h / 2)
    box(ax, 1.28, ye, 2.22, lane_h, "Ego MLP: Linear → GELU → Linear (128)", fc=AQUA_L, ec=AQUA_D, fs=6.8)
    arrow(ax, 3.5, ye + lane_h / 2, 3.58, ye + lane_h / 2)
    box(ax, 3.58, ye, 0.5, lane_h, "e (128)", fc=AQUA_L, ec=AQUA_D, fs=7.2)
    xf, wf = 4.38, 2.74
    yf = ys[1] - 0.05; hf = 0.62
    for yl in ys + [ye]:
        line(ax, 4.08, yl + lane_h / 2, 4.22, yl + lane_h / 2)
        line(ax, 4.22, yl + lane_h / 2, 4.22, yf + hf / 2)
    arrow(ax, 4.22, yf + hf / 2, xf, yf + hf / 2)
    box(ax, xf, yf, wf, hf, "Fusion: concat [s10; s20; s30; e] (1,664)\n→ Linear 512, GELU → residual MLP 512→1024→512", fc=AQUA_L, ec=AQUA_D, fs=6.5)
    cx = xf + wf / 2
    yh = yf - 0.46
    arrow(ax, cx, yf, cx, yh + 0.34)
    box(ax, xf + 0.55, yh, wf - 1.1, 0.34, "driving state h (512)", fc=AQUA_L, ec=AQUA_D, fs=8, weight="bold")
    yt = yh - 0.46
    arrow(ax, cx, yh, cx, yt + 0.34)
    box(ax, xf, yt, wf, 0.34, "h + time embedding ψ(t), t = 1 … K (K = 8 or 40)", fc=AQUA_L, ec=AQUA_D, fs=6.8)
    yo = yt - 0.56
    arrow(ax, cx, yt, cx, yo + 0.44)
    box(ax, xf, yo, wf, 0.44, "Output layer (FP32)\nLN → Linear 512→256 → GELU → Linear 256→3", fc=AQUA_L, ec=AQUA_D, fs=6.8)
    yr = yo - 0.5
    arrow(ax, cx, yo, cx, yr + 0.38)
    box(ax, xf, yr, wf, 0.38, "(Δx, Δy, Δθ) per step → de-normalize → cumsum → poses", fc=ORANGE_L, ec=ORANGE_D, fs=5.9, weight="bold")
    text(ax, 0.08, 0.2, "Inputs: three block outputs from future pass 1 and the ego state.\nTrainable parameters: 9.98M (0.16% of the 6.24B backbone).", fs=7.6, color=INK2, ha="left", va="center")
    save(fig, "fig_sf3")


# ---------------------------------------------------------------------
# Figure: PhysicalAI clip / chunk / teacher forcing
# ---------------------------------------------------------------------
def fig_pai():
    W, H = 7.2, 1.85
    fig, ax = canvas(W, H)
    x0, x1 = 0.08, 7.12
    yf = 1.45
    n = 21
    text(ax, x0, yf + 0.22, "Test clip: 21 front-camera frames at 1 fps + ego record + one VLM guidance sentence per chunk", fs=8, color=INK2, ha="left")
    fw = (x1 - x0) / n
    for i in range(n):
        ax.add_patch(Rectangle((x0 + i * fw + 0.01, yf), fw - 0.02, 0.16, fc=GRAY_F, ec=GRAY_E, lw=0.6))
    yc, hc = 0.85, 0.42
    for k in range(5):
        xs = x0 + (1 + 4 * k) * fw + 0.01
        wc = 4 * fw - 0.02
        fc = ORANGE_L if k == 0 else BLUE_L
        ec = ORANGE_D if k == 0 else BLUE_D
        box(ax, xs, yc, wc, hc, f"chunk {k+1}\n4 s · 40 steps", fc=fc, ec=ec, fs=7.4, r=0.05)
        arrow(ax, xs + wc / 2, yf - 0.01, xs + wc / 2, yc + hc + 0.02, color=GRAY_E, lw=0.8)
        if k < 4:
            arrow(ax, xs + wc, yc + hc / 2, xs + wc + 0.02, yc + hc / 2, color=ARROW)
    text(ax, x0 + 0.5 * fw, yc + hc / 2, "init", fs=7, color=MUTED)
    text(ax, W / 2, 0.42, "Teacher forcing: after predicting chunk k the prediction is discarded; the recorded frames and actions are registered before chunk k+1.", fs=7.2, color=INK2)
    text(ax, W / 2, 0.15, "Backbone forwards per clip: B10 87 (video 20 + action 55 + register 12) · V0-A2 27 (5 + 10 + 12) · readout 17 (5 + 0 + 12)", fs=7.0, color=ORANGE_D, weight="bold")
    save(fig, "fig_pai")


# ---------------------------------------------------------------------
# Figure: readout phase results (bars), 4k heads on both benchmarks + 20k heads on PhysicalAI
# ---------------------------------------------------------------------
def _bars(ax, stages, data, cols, fmt, lift):
    means = [sum(data[s]) / len(data[s]) for s in stages]
    ax.bar(range(len(stages)), means, width=0.52, color=cols, zorder=2)
    for i, s in enumerate(stages):
        for v in data[s]:
            ax.plot(i, v, "o", ms=3.5, color=INK2, mec="white", mew=1, zorder=3)
        ax.text(i, max(data[s]) + lift, fmt.format(means[i]), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages)


def fig_phase_results():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.14, wspace=0.38)
    # NAVSIM, 4k head, seeds 17/29
    ax = axes[0]; style_axes(ax)
    nav = {"current": [83.20, 83.61], "first": [86.60, 86.89], "terminal": [87.34, 86.28]}
    _bars(ax, ["current", "first", "terminal"], nav, [BLUE_LL, ORANGE, BLUE_LL], "{:.1f}", 0.3)
    ax.axhline(90.93, color=MUTED, lw=1, ls="--", label="B10 90.93")
    ax.set_ylim(80, 93.2); ax.set_ylabel("PDMS (higher is better)", fontsize=8, color=INK2)
    ax.set_title("NAVSIM dev, 4k head", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    # PhysicalAI, 4k head, seeds 17/29/43
    ax = axes[1]; style_axes(ax)
    pai4 = {"current": [0.922, 0.932, 0.926], "first": [0.809, 0.817, 0.813], "terminal": [0.822, 0.830, 0.825]}
    _bars(ax, ["current", "first", "terminal"], pai4, [BLUE_LL, ORANGE, BLUE_LL], "{:.3f}", 0.007)
    ax.axhline(0.821, color=MUTED, lw=1, ls="--", label="B10-opt 0.821")
    ax.set_ylim(0.70, 0.975); ax.set_ylabel("ADE (m, lower is better)", fontsize=8, color=INK2)
    ax.set_title("PhysicalAI-AV dev, 4k head", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    # PhysicalAI, 20k head, seeds 17/29/43 (+ ego-only control)
    ax = axes[2]; style_axes(ax)
    pai20 = {"ego-only": [0.8732, 0.8732, 0.8732], "current": [0.8649, 0.8683, 0.8602], "first": [0.7616, 0.7568, 0.7665]}
    _bars(ax, ["ego-only", "current", "first"], pai20, [GRAY_E, BLUE_LL, ORANGE], "{:.3f}", 0.007)
    ax.axhline(0.821, color=MUTED, lw=1, ls="--", label="B10-opt 0.821")
    ax.axhline(0.786, color=MUTED, lw=1, ls=":", label="V0-A2 0.786")
    ax.set_ylim(0.70, 0.975); ax.set_ylabel("ADE (m, lower is better)", fontsize=8, color=INK2)
    ax.set_title("PhysicalAI-AV dev, 20k head", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    save(fig, "fig_phase_results")


# ---------------------------------------------------------------------
# Figure: data-scale effect
# ---------------------------------------------------------------------
def fig_scale():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.2, wspace=0.3)
    ax = axes[0]; style_axes(ax)
    sizes = [4000, 20000, 92599]
    s17 = [86.60, 89.36, 90.43]; s29 = [86.89, 90.00, 90.58]
    ax.plot(sizes, s17, "-o", color=BLUE, lw=2, ms=5, mec="white", mew=1.2, label="seed 17", zorder=3)
    ax.plot(sizes, s29, "-o", color=ORANGE, lw=2, ms=5, mec="white", mew=1.2, label="seed 29", zorder=3)
    ax.axhline(90.93, color=MUTED, lw=1, ls="--"); ax.text(4000, 90.93 - 0.12, "B10 90.93", fontsize=7.5, color=MUTED, va="top")
    ax.axhline(91.34, color=MUTED, lw=1, ls=":"); ax.text(92599, 91.34 + 0.12, "V0-A2 91.34", fontsize=7.5, color=MUTED, va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels(["4k", "20k", "92.6k"]); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(3000, 130000); ax.set_ylim(85.5, 92.0)
    ax.set_xlabel("head-training scenes", fontsize=8, color=INK2); ax.set_ylabel("PDMS", fontsize=8, color=INK2)
    ax.set_title("NAVSIM navtest-dev, SF3-first", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    for x, v in zip(sizes, s29):
        ax.text(x, v + 0.18, f"{v:.2f}", fontsize=7.5, color=INK, ha="center", va="bottom")
    for x, v in zip(sizes, s17):
        ax.text(x, v - 0.2, f"{v:.2f}", fontsize=7.5, color=INK, ha="center", va="top")
    ax = axes[1]; style_axes(ax)
    sizes2 = [4000, 20000]
    seeds = {"seed 17": [0.813, 0.762], "seed 29": [0.817, 0.757], "seed 43": [0.809, 0.767]}
    for (nm, v), col in zip(seeds.items(), [BLUE, ORANGE, AQUA]):
        ax.plot(sizes2, v, "-o", color=col, lw=2, ms=5, mec="white", mew=1.2, label=nm, zorder=3)
    ax.axhline(0.821, color=MUTED, lw=1, ls="--"); ax.text(20000, 0.821 + 0.003, "B10-opt 0.821", fontsize=7.5, color=MUTED, va="bottom", ha="right")
    ax.axhline(0.786, color=MUTED, lw=1, ls=":"); ax.text(20000, 0.786 + 0.003, "V0-A2 0.786", fontsize=7.5, color=MUTED, va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_xticks(sizes2); ax.set_xticklabels(["4k", "20k"]); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(3000, 28000); ax.set_ylim(0.74, 0.87)
    ax.set_xlabel("head-training clips", fontsize=8, color=INK2); ax.set_ylabel("ADE (m)", fontsize=8, color=INK2)
    ax.set_title("PhysicalAI-AV dev, SF3-first", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    save(fig, "fig_scale")


# ---------------------------------------------------------------------
# Figure: quality vs latency on the locked final sets (quiet-host session, 2026-11-01)
# ---------------------------------------------------------------------
def fig_tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.2, wspace=0.3)
    ax = axes[0]; style_axes(ax)
    base = [("B10", 796.9, 89.67), ("B02-opt", 419.1, 89.65), ("V0-A2 (B*)", 269.6, 89.72)]
    for nm, x, y in base:
        ax.plot(x, y, "o", ms=8, color=BLUE, mec="white", mew=1.5, zorder=3)
        ax.text(x, y + 0.07, nm, fontsize=7.5, color=INK, ha="center", va="bottom")
    for y in [89.49, 89.25]:
        ax.plot(186.3, y, "o", ms=8, color=ORANGE, mec="white", mew=1.5, zorder=3)
    ax.text(186.3 + 28, 89.49, "SF3-first, seed 29", fontsize=7.5, color=INK, ha="left", va="center")
    ax.text(186.3 + 28, 89.25, "SF3-first, seed 17", fontsize=7.5, color=INK, ha="left", va="center")
    ax.set_xlim(0, 900); ax.set_ylim(88.9, 90.1)
    ax.set_xlabel("policy latency p50 (ms)", fontsize=8, color=INK2); ax.set_ylabel("PDMS", fontsize=8, color=INK2)
    ax.set_title("NAVSIM navtest-final (8,896 scenes)", fontsize=8.5, color=INK, loc="left")
    ax.text(880, 88.95, "better: upper left", fontsize=7, color=MUTED, ha="right", va="bottom")
    ax.plot([], [], "o", color=BLUE, label="DriveWAM, training-free"); ax.plot([], [], "o", color=ORANGE, label="readout (SF3-first)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax = axes[1]; style_axes(ax)
    base = [("B10-opt", 803.9, 0.802), ("B05-opt", 529.3, 0.809), ("B03-opt", 419.5, 0.842), ("V0-A2 (B*)", 175.4, 0.778)]
    for nm, x, y in base:
        ax.plot(x, y, "o", ms=8, color=BLUE, mec="white", mew=1.5, zorder=3)
    ax.text(803.9, 0.802 + 0.006, "B10-opt", fontsize=7.5, color=INK, ha="center", va="bottom")
    ax.text(529.3, 0.809 + 0.006, "B05-opt", fontsize=7.5, color=INK, ha="center", va="bottom")
    ax.text(419.5, 0.842 + 0.006, "B03-opt", fontsize=7.5, color=INK, ha="center", va="bottom")
    ax.text(175.4 + 25, 0.778 + 0.006, "V0-A2 (B*)", fontsize=7.5, color=INK, ha="left", va="bottom")
    for y in [0.750, 0.748, 0.762]:
        ax.plot(67.4, y, "o", ms=8, color=ORANGE, mec="white", mew=1.5, zorder=3)
    ax.text(67.4 + 25, 0.752, "SF3-first (3 seeds)", fontsize=7.5, color=INK, ha="left", va="center")
    ax.set_xlim(0, 900); ax.set_ylim(0.73, 0.87)
    ax.set_xlabel("decision latency p50 (ms)", fontsize=8, color=INK2); ax.set_ylabel("ADE (m, lower is better)", fontsize=8, color=INK2)
    ax.set_title("PhysicalAI-AV official test 1k", fontsize=8.5, color=INK, loc="left")
    ax.text(880, 0.862, "better: lower left", fontsize=7, color=MUTED, ha="right", va="top")
    save(fig, "fig_tradeoff")


# ---------------------------------------------------------------------
# Figure: training-free grid (future denoising steps v × action steps k) on the dev sets
# ---------------------------------------------------------------------
def fig_grid():
    ks = [1, 2, 3, 5, 10]
    vs = [0, 1, 2, 3]
    # NAVSIM navtest-dev: ΔPDMS vs B10 (mean of paired differences); latency p50 (ms) from the quiet session
    nav = {(3, 1): -1.19, (3, 2): -0.05, (3, 3): -0.04, (3, 5): -0.03, (3, 10): 0.0,
           (2, 1): -1.02, (2, 2): -0.00, (2, 3): -0.01, (2, 5): -0.01,
           (1, 1): -1.13, (1, 2): 0.00, (1, 3): 0.10, (1, 5): 0.08,
           (0, 1): -0.61, (0, 2): 0.41, (0, 3): 0.54, (0, 5): 0.53}
    nav_lat = {(3, 10): 797, (3, 5): 545, (3, 3): 461, (3, 2): 419,
               (0, 2): 270, (0, 3): 312, (0, 5): 396, (1, 2): 319, (1, 3): 362, (1, 5): 446,
               (2, 2): 369, (2, 3): 411, (2, 5): 496}
    nav_pass = {(3, 5), (3, 3), (3, 2), (0, 2), (0, 3), (0, 5), (1, 2), (1, 3), (1, 5), (2, 2), (2, 3), (2, 5)}
    # PhysicalAI-AV dev: ΔADE vs B10-opt (m)
    pai = {(3, 1): 0.440, (3, 2): 0.079, (3, 3): 0.031, (3, 5): 0.004, (3, 10): 0.0,
           (2, 1): 0.432, (2, 2): 0.069, (2, 3): 0.021, (2, 5): -0.005,
           (1, 1): 0.415, (1, 2): 0.045, (1, 3): -0.002, (1, 5): -0.026,
           (0, 1): 0.318, (0, 2): -0.035, (0, 3): -0.058, (0, 5): -0.051}
    pai_lat = {(3, 10): 804, (3, 5): 529, (3, 3): 420,
               (0, 2): 175, (0, 3): 230, (0, 5): 340, (1, 3): 294, (1, 5): 403, (2, 3): 357, (2, 5): 466}
    pai_pass = {(3, 10), (3, 5), (3, 3), (0, 2), (0, 3), (0, 5), (1, 3), (1, 5), (2, 3), (2, 5)}
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE_D, "#f6d9cc", "#f2f2f0", "#cfe0f7", BLUE_D])

    def panel(ax, data, lat, passing, better_sign, clip, title, fmt, ylabel):
        norm = Normalize(-clip, clip)
        for r, v in enumerate(vs):
            for c, k in enumerate(ks):
                x, y = c, len(vs) - 1 - r
                if (v, k) not in data:
                    ax.add_patch(Rectangle((x, y), 1, 1, fc="white", ec=GRID, lw=0.8, hatch="////", alpha=0.9))
                    ax.text(x + 0.5, y + 0.5, "not run", ha="center", va="center", fontsize=6.5, color=MUTED)
                    continue
                val = data[(v, k)]
                col = cmap(norm(better_sign * val))
                ax.add_patch(Rectangle((x, y), 1, 1, fc=col, ec="white", lw=1.5))
                label = "ref." if (v, k) == (3, 10) else fmt.format(val)
                ax.text(x + 0.5, y + 0.62, label, ha="center", va="center", fontsize=7.8, color=INK, fontweight="bold" if (v, k) in passing else "normal")
                if (v, k) in lat:
                    ax.text(x + 0.5, y + 0.3, f"{lat[(v, k)]} ms", ha="center", va="center", fontsize=6.6, color=INK2)
                if (v, k) in passing:
                    ax.add_patch(Rectangle((x + 0.06, y + 0.06), 0.88, 0.88, fill=False, ec=INK, lw=1.0))
                if (v, k) == (0, 2):
                    ax.add_patch(Rectangle((x + 0.03, y + 0.03), 0.94, 0.94, fill=False, ec=ORANGE, lw=2.2))
        ax.set_xlim(0, len(ks)); ax.set_ylim(0, len(vs))
        ax.set_xticks([c + 0.5 for c in range(len(ks))]); ax.set_xticklabels([f"k = {k}" for k in ks], fontsize=7.5)
        ax.set_yticks([len(vs) - 1 - r + 0.5 for r in range(len(vs))])
        ax.set_yticklabels([f"v = {v}" + (" (B10)" if v == 3 else "") for v in vs], fontsize=7.5)
        ax.tick_params(length=0, colors=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("action denoising steps k", fontsize=8, color=INK2)
        if ylabel:
            ax.set_ylabel("future denoising steps v", fontsize=8, color=INK2)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.88, bottom=0.3, wspace=0.36)
    panel(axes[0], nav, nav_lat, nav_pass, +1, 0.6, "NAVSIM dev: ΔPDMS vs. B10 (higher is better)", "{:+.2f}", True)
    panel(axes[1], pai, pai_lat, pai_pass, -1, 0.08, "PhysicalAI-AV dev: ΔADE vs. B10-opt (m)", "{:+.3f}", False)
    fig.text(0.5, 0.025, "Black frame: passes the B* rule.  Orange frame: V0-A2, the fastest such variant on both benchmarks.\nSecond line: median latency (ms) in the quiet-host session; only screened-in variants were timed.",
             ha="center", va="bottom", fontsize=6.9, color=INK2, linespacing=1.3)
    save(fig, "fig_grid")


if __name__ == "__main__":
    for f in [fig_overview, fig_overview_poster, fig_phases, fig_sf3, fig_pai, fig_phase_results, fig_scale, fig_tradeoff, fig_grid]:
        f(); print("done", f.__name__)
