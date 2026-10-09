# -*- coding: utf-8 -*-
"""Figures for the DriveWAM lightweight action readout document (matplotlib, Apple SD Gothic Neo)."""
import glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import NullLocator

SP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SP, "fig")
os.makedirs(OUT, exist_ok=True)
for p in glob.glob(os.path.join(SP, "fonts", "*.ttf")):
    fm.fontManager.addfont(p)
plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "savefig.dpi": 220,
    "figure.dpi": 100,
})

# ---- palette (validated dataviz reference palette + light tints) ----
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#7a7974"; GRID = "#e4e3df"
BLUE = "#2a78d6"; BLUE_D = "#1c5cab"; BLUE_L = "#cde2fb"; BLUE_LL = "#9ec5f4"
ORANGE = "#eb6834"; ORANGE_D = "#b8431c"; ORANGE_L = "#fde3d7"
AQUA = "#1baf7a"; AQUA_D = "#0f7a55"; AQUA_L = "#d7f3e8"
GRAY_F = "#f0efec"; GRAY_E = "#b5b3ad"
ARROW = "#52514e"

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
    """specs: list of (width, text, fc, ec, kw). Draw boxes left->right with arrows. Returns list of (x,w)."""
    pos = []; x = x0
    for i, (w, t, fc, ec, kw) in enumerate(specs):
        box(ax, x, y, w, h, t, fc=fc, ec=ec, fs=kw.get("fs", fs), weight=kw.get("weight", "normal"),
            ls=kw.get("ls", "-"), tc=kw.get("tc", INK), lw=kw.get("lw", 1.0))
        pos.append((x, w))
        if i < len(specs) - 1:
            arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2, color=kw.get("ac", ARROW))
        x += w + gap
    return pos

# =====================================================================
# Figure 1: DriveWAM inference flow
# =====================================================================
def fig1():
    W, H = 7.2, 2.55
    fig, ax = canvas(W, H)
    y, h = 1.05, 0.95
    specs = [
        (1.3, "입력\n현재·과거 영상 (VAE latent)\n과거 action, ego state\nVLM 안내 문장", GRAY_F, GRAY_E, {"fs": 7.6}),
        (1.45, "미래 영상 latent 생성\n5B Video DiT × 3 step", BLUE_L, BLUE_D, {"weight": "bold"}),
        (0.85, "미래 latent\n(상상한 미래)", BLUE_L, BLUE_D, {}),
        (1.45, "action denoising\n같은 5B Video DiT × 10 step", BLUE_L, BLUE_D, {"weight": "bold"}),
        (0.85, "Action Decoder\n(MLP)", GRAY_F, GRAY_E, {}),
        (0.75, "4초\ntrajectory", AQUA_L, AQUA_D, {"weight": "bold"}),
    ]
    pos = chain(ax, y, h, specs, x0=0.08, gap=0.09, fs=8.3)
    # shared-backbone bracket under the two DiT boxes
    (x1, w1), (x3, w3) = pos[1], pos[3]
    yb = y - 0.14
    line(ax, x1, yb, x3 + w3, yb, color=BLUE_D, lw=1.2)
    line(ax, x1, yb, x1, yb + 0.08, color=BLUE_D, lw=1.2); line(ax, x3 + w3, yb, x3 + w3, yb + 0.08, color=BLUE_D, lw=1.2)
    text(ax, (x1 + x3 + w3) / 2, yb - 0.17, "동일한 5B Video Diffusion Transformer(30 layer)를 미래 생성과 action 생성에 공유", fs=8, color=BLUE_D, weight="bold")
    # paper latency annotations (H20)
    text(ax, x1 + w1 / 2, y + h + 0.17, "논문 보고 지연(H20): 372 ms", fs=7.8, color=INK2)
    text(ax, x3 + w3 / 2, y + h + 0.17, "논문 보고 지연(H20): 765 ms (5 step: 374 ms)", fs=7.8, color=ORANGE_D, weight="bold")
    text(ax, W / 2, 0.22, "기본 설정 B10: action 생성에 backbone을 10회 반복 실행한다. 출력은 4초 궤적(수십 개 수치)뿐이지만 계산량은 미래 영상 생성보다 크다.", fs=8, color=INK2)
    fig.savefig(os.path.join(OUT, "fig1_drivewam.png")); plt.close(fig)

# =====================================================================
# Figure 2: original vs proposed path
# =====================================================================
def fig2():
    W, H = 7.2, 3.05
    fig, ax = canvas(W, H)
    h = 0.62
    # Row 1: original
    y1 = 2.05
    text(ax, 0.06, y1 + h + 0.13, "기존 DriveWAM (B10)", fs=9, color=INK, ha="left", weight="bold")
    specs1 = [
        (0.55, "관측", GRAY_F, GRAY_E, {}),
        (0.68, "Video VAE", GRAY_F, GRAY_E, {}),
        (0.75, "현재 등록\npass", BLUE_L, BLUE_D, {}),
        (0.66, "미래\npass 1", BLUE_L, BLUE_D, {}),
        (0.66, "미래\npass 2", BLUE_L, BLUE_D, {}),
        (0.66, "미래\npass 3", BLUE_L, BLUE_D, {}),
        (1.02, "action denoising\n× 10", BLUE_L, BLUE_D, {"weight": "bold"}),
        (0.68, "Action\nDecoder", GRAY_F, GRAY_E, {}),
        (0.55, "4초\n궤적", AQUA_L, AQUA_D, {}),
    ]
    pos1 = chain(ax, y1, h, specs1, x0=0.06, gap=0.065, fs=8)
    # Row 2: proposed
    y2 = 0.72
    text(ax, 0.06, y2 + h + 0.13, "제안 모델: 첫 상상 pass 조기 종료 + SF3 head", fs=9, color=INK, ha="left", weight="bold")
    specs2 = [
        (0.55, "관측", GRAY_F, GRAY_E, {}),
        (0.68, "Video VAE", GRAY_F, GRAY_E, {}),
        (0.75, "현재 등록\npass", BLUE_L, BLUE_D, {}),
        (0.66, "미래\npass 1", ORANGE_L, ORANGE_D, {"weight": "bold", "lw": 1.4}),
        (1.0, "layer 10·20·30\nhidden state", ORANGE_L, ORANGE_D, {}),
        (1.05, "SF3 head\n(학습 대상, 작음)", AQUA_L, AQUA_D, {"weight": "bold", "lw": 1.4}),
        (0.55, "4초\n궤적", AQUA_L, AQUA_D, {}),
    ]
    pos2 = chain(ax, y2, h, specs2, x0=0.06, gap=0.065, fs=8)
    # skipped ghost
    xg = pos2[-1][0] + pos2[-1][1] + 0.12
    wg = W - xg - 0.06
    box(ax, xg, y2, wg, h, "건너뜀\n미래 pass 2·3,\naction denoising ×10", fc="white", ec=GRAY_E, ls="--", fs=7.6, tc=MUTED)
    # frozen bracket over row2 DiT boxes
    xa = pos2[2][0]; xb = pos2[3][0] + pos2[3][1]
    yb = y2 - 0.1
    line(ax, xa, yb, xb, yb, color=BLUE_D, lw=1.1); line(ax, xa, yb, xa, yb + 0.07, color=BLUE_D, lw=1.1); line(ax, xb, yb, xb, yb + 0.07, color=BLUE_D, lw=1.1)
    text(ax, (xa + xb) / 2, yb - 0.14, "DriveWAM backbone은 그대로 고정(파라미터 갱신 없음)", fs=7.8, color=BLUE_D)
    # hook arrow note
    text(ax, W / 2, 0.14, "제안 모델은 미래 pass 1 직후 hook으로 중간 layer 출력을 읽고 inference를 끝낸다. 나머지 미래 denoising과 반복 action 생성은 실행하지 않는다.", fs=8, color=INK2)
    fig.savefig(os.path.join(OUT, "fig2_compare.png")); plt.close(fig)

# =====================================================================
# Figure 3: representation timing (current / first / terminal)
# =====================================================================
def fig3():
    W, H = 7.2, 2.15
    fig, ax = canvas(W, H)
    y, h = 1.12, 0.6
    specs = [
        (1.05, "현재 등록 pass\n(관측 처리)", BLUE_L, BLUE_D, {}),
        (1.15, "미래 pass 1\n입력 latent = 순수 noise", ORANGE_L, ORANGE_D, {"weight": "bold"}),
        (0.85, "미래 pass 2", BLUE_L, BLUE_D, {}),
        (0.85, "미래 pass 3", BLUE_L, BLUE_D, {}),
        (1.05, "terminal pass\n(갱신 완료 latent)", BLUE_L, BLUE_D, {}),
        (1.2, "action denoising\n× 10 → 궤적", GRAY_F, GRAY_E, {}),
    ]
    pos = chain(ax, y, h, specs, x0=0.08, gap=0.09, fs=8)
    text(ax, 0.08, y + h + 0.16, "한 chunk(4초)에 대한 backbone 실행 순서 →", fs=8, color=MUTED, ha="left")
    # markers
    marks = [(0, "Current", "관측만 처리한 표현\n상상 없음", BLUE_D),
             (1, "First", "미래를 noise에서 한 번\n상상한 표현 (제안 모델이 사용)", ORANGE_D),
             (4, "Terminal", "미래 latent 갱신 3회를\n마친 뒤의 표현", BLUE_D)]
    for i, name, desc, col in marks:
        x, w = pos[i]
        cx = x + w / 2
        arrow(ax, cx, y - 0.02, cx, y - 0.3, color=col, lw=1.2)
        text(ax, cx, y - 0.42, name, fs=9, color=col, weight="bold")
        text(ax, cx, y - 0.78, desc, fs=7.6, color=INK2)
    # hook note on each: layers
    text(ax, W - 0.08, y + h + 0.16, "각 시점에서 layer 10·20·30 출력(112 token × 3072)을 읽는다", fs=8, color=MUTED, ha="right")
    fig.savefig(os.path.join(OUT, "fig3_timing.png")); plt.close(fig)

# =====================================================================
# Figure 4: SF3 head architecture
# =====================================================================
def fig4():
    W, H = 7.2, 3.45
    fig, ax = canvas(W, H)
    # left lanes
    lane_h = 0.5; gap = 0.14
    ys = [2.85, 2.85 - (lane_h + gap), 2.85 - 2 * (lane_h + gap)]
    names = ["layer 10", "layer 20", "layer 30"]
    for yl, nm in zip(ys, names):
        box(ax, 0.08, yl, 1.05, lane_h, f"{nm} hidden\n112 token × 3072", fc=ORANGE_L, ec=ORANGE_D, fs=7.8)
        arrow(ax, 1.13, yl + lane_h / 2, 1.25, yl + lane_h / 2)
        box(ax, 1.25, yl, 2.1, lane_h, "Query pooling: LayerNorm → Linear 3072→512\n→ 16 learnable query × 8-head attention → 가중 합", fc=AQUA_L, ec=AQUA_D, fs=7.2)
        arrow(ax, 3.35, yl + lane_h / 2, 3.47, yl + lane_h / 2)
        box(ax, 3.47, yl, 0.62, lane_h, "s (512)", fc=AQUA_L, ec=AQUA_D, fs=8)
    # ego lane
    ye = ys[2] - (lane_h + gap)
    box(ax, 0.08, ye, 1.05, lane_h, "ego state\n(속도·가속도 등)", fc=GRAY_F, ec=GRAY_E, fs=7.8)
    arrow(ax, 1.13, ye + lane_h / 2, 1.25, ye + lane_h / 2)
    box(ax, 1.25, ye, 2.1, lane_h, "Ego MLP: Linear → GELU → Linear (128)", fc=AQUA_L, ec=AQUA_D, fs=7.4)
    arrow(ax, 3.35, ye + lane_h / 2, 3.47, ye + lane_h / 2)
    box(ax, 3.47, ye, 0.62, lane_h, "e (128)", fc=AQUA_L, ec=AQUA_D, fs=8)
    # merge lines to fusion
    xf, wf = 4.45, 2.65
    yf = ys[1] - 0.05; hf = 0.62
    for yl in ys + [ye]:
        line(ax, 4.09, yl + lane_h / 2, 4.27, yl + lane_h / 2)
        line(ax, 4.27, yl + lane_h / 2, 4.27, yf + hf / 2)
    arrow(ax, 4.27, yf + hf / 2, xf, yf + hf / 2)
    box(ax, xf, yf, wf, hf, "결합: concat [s10; s20; s30; e] (1,664)\n→ Linear → 512, GELU  → Residual MLP (512→1024→512)", fc=AQUA_L, ec=AQUA_D, fs=7.4, weight="normal")
    # down chain on the right
    cx = xf + wf / 2
    yh = yf - 0.46
    arrow(ax, cx, yf, cx, yh + 0.34)
    box(ax, xf + 0.55, yh, wf - 1.1, 0.34, "driving state h (512)", fc=AQUA_L, ec=AQUA_D, fs=8, weight="bold")
    yt = yh - 0.46
    arrow(ax, cx, yh, cx, yt + 0.34)
    box(ax, xf, yt, wf, 0.34, "h + 시간 임베딩 ψ(t), t = 1 … K  (K = 8 또는 40)", fc=AQUA_L, ec=AQUA_D, fs=7.6)
    yo = yt - 0.56
    arrow(ax, cx, yt, cx, yo + 0.44)
    box(ax, xf, yo, wf, 0.44, "출력층 (FP32)\nLayerNorm → Linear 512→256 → GELU → Linear 256→3", fc=AQUA_L, ec=AQUA_D, fs=7.4)
    yr = yo - 0.5
    arrow(ax, cx, yo, cx, yr + 0.38)
    box(ax, xf, yr, wf, 0.38, "step별 (Δx, Δy, Δθ) → 역정규화·누적합 → 4초 궤적", fc=ORANGE_L, ec=ORANGE_D, fs=7.6, weight="bold")
    text(ax, 0.08, 0.2, "입력은 backbone이 미래 pass 1에서 만든 세 layer의 token 표현과 ego state뿐이다.\nhead의 학습 파라미터는 약 998만 개로, 6.24B backbone의 0.16%다.", fs=7.8, color=INK2, ha="left", va="center")
    fig.savefig(os.path.join(OUT, "fig4_sf3.png")); plt.close(fig)

# =====================================================================
# Figure 5: PhysicalAI clip / chunk / teacher forcing
# =====================================================================
def fig5():
    W, H = 7.2, 1.85
    fig, ax = canvas(W, H)
    # frame strip
    x0, x1 = 0.08, 7.12
    yf = 1.45
    n = 21
    text(ax, x0, yf + 0.22, "test clip: 1 fps × 21 frame (전방 카메라) + ego 기록 + chunk별 VLM 안내 문장", fs=8, color=INK2, ha="left")
    fw = (x1 - x0) / n
    for i in range(n):
        ax.add_patch(Rectangle((x0 + i * fw + 0.01, yf), fw - 0.02, 0.16, fc=GRAY_F, ec=GRAY_E, lw=0.6))
    # 5 chunks of 4 s = 4 frames each, starting at frame 1 (frames 1-4 first chunk)
    yc, hc = 0.85, 0.42
    for k in range(5):
        xs = x0 + (1 + 4 * k) * fw + 0.01
        wc = 4 * fw - 0.02
        fc = ORANGE_L if k == 0 else BLUE_L
        ec = ORANGE_D if k == 0 else BLUE_D
        box(ax, xs, yc, wc, hc, f"chunk {k+1}\n4초 · 40 step 예측", fc=fc, ec=ec, fs=7.6, r=0.05)
        arrow(ax, xs + wc / 2, yf - 0.01, xs + wc / 2, yc + hc + 0.02, color=GRAY_E, lw=0.8)
        if k < 4:
            arrow(ax, xs + wc, yc + hc / 2, xs + wc + 0.02 + 0.0, yc + hc / 2, color=ARROW)
    text(ax, x0 + 0.5 * fw, yc + hc / 2, "입력\n등록", fs=7, color=MUTED)
    text(ax, W / 2, 0.42, "teacher forcing: chunk k를 예측한 뒤 예측 궤적은 버리고, 실제 기록된 영상·action을 backbone에 등록한 다음 chunk k+1을 예측한다.", fs=8, color=INK2)
    text(ax, W / 2, 0.15, "clip 1개당 backbone 실행 횟수(검증값): B10 87회 (영상 20 + action 55 + 등록 12)  /  제안 모델 17회 (영상 5 + 등록 12)", fs=8, color=ORANGE_D, weight="bold")
    fig.savefig(os.path.join(OUT, "fig5_pai.png")); plt.close(fig)

# =====================================================================
# Figure 6: evaluation protocol flow
# =====================================================================
def fig6():
    W, H = 7.2, 1.55
    fig, ax = canvas(W, H)
    steps = ["① 환경·weight 고정", "② 데이터 분할 고정", "③ 공식 DriveWAM 재현\n(비트 단위 parity)", "④ B10·저-step\n기준 평가", "⑤ dev에서 B* 선정", "⑥ 표현 추출\ncurrent/first/terminal",
             "⑦ SF3 head 학습", "⑧ 시점·규모 분석", "⑨ 품질 평가\nPDMS / ADE·FDE", "⑩ 지연 측정\n(교차 3회)", "⑪ 군집 paired\nbootstrap 2,000회", "⑫ 잠금 final\n1회 평가"]
    n = 6; gap = 0.09; x0 = 0.08
    bw = (W - 2 * x0 - (n - 1) * gap) / n
    h = 0.5
    rows = [(0.95, steps[:6]), (0.18, steps[6:])]
    for r, (y, ss) in enumerate(rows):
        for i, s in enumerate(ss):
            x = x0 + i * (bw + gap)
            fc = ORANGE_L if (r == 1 and i == 5) else (BLUE_L if r == 0 else AQUA_L)
            ec = ORANGE_D if (r == 1 and i == 5) else (BLUE_D if r == 0 else AQUA_D)
            box(ax, x, y, bw, h, s, fc=fc, ec=ec, fs=7.4, r=0.05)
            if i < n - 1:
                arrow(ax, x + bw, y + h / 2, x + bw + gap, y + h / 2)
    # connector from end of row 1 to start of row 2
    xe = x0 + (n - 1) * (bw + gap) + bw / 2
    line(ax, xe, 0.95, xe, 0.82); line(ax, xe, 0.82, x0 + bw / 2, 0.82)
    arrow(ax, x0 + bw / 2, 0.82, x0 + bw / 2, 0.18 + h + 0.0)
    fig.savefig(os.path.join(OUT, "fig6_protocol.png")); plt.close(fig)

# =====================================================================
# Figure 7: representation timing results (bars)
# =====================================================================
def style_axes(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.2, wspace=0.3)
    stages = ["Current", "First", "Terminal"]
    cols = [BLUE_LL, BLUE, BLUE_LL]
    # NAVSIM dev, 4k head, seeds 17/29
    nav = {"Current": [83.20, 83.61], "First": [86.60, 86.89], "Terminal": [87.34, 86.28]}
    ax = axes[0]; style_axes(ax)
    means = [sum(nav[s]) / 2 for s in stages]
    ax.bar(stages, means, width=0.5, color=cols, zorder=2)
    for i, s in enumerate(stages):
        for v in nav[s]:
            ax.plot(i, v, "o", ms=3.5, color=INK2, mec="white", mew=1, zorder=3)
        ax.text(i, max(nav[s]) + 0.3, f"{means[i]:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(90.93, color=MUTED, lw=1, ls="--"); ax.text(2.42, 90.93 + 0.15, "B10 90.93", fontsize=7.5, color=MUTED, ha="right", va="bottom")
    ax.set_ylim(80, 92.5); ax.set_ylabel("PDMS (높을수록 좋음)", fontsize=8, color=INK2)
    ax.set_title("NAVSIM navtest-dev · head 4k · seed 17/29", fontsize=8.5, color=INK, loc="left")
    # PhysicalAI dev, 4k, 3 seeds
    pai = {"Current": [0.922, 0.932, 0.926], "First": [0.809, 0.817, 0.813], "Terminal": [0.822, 0.830, 0.825]}
    ax = axes[1]; style_axes(ax)
    means = [sum(pai[s]) / 3 for s in stages]
    ax.bar(stages, means, width=0.5, color=cols, zorder=2)
    for i, s in enumerate(stages):
        for v in pai[s]:
            ax.plot(i, v, "o", ms=3.5, color=INK2, mec="white", mew=1, zorder=3)
        ax.text(i, max(pai[s]) + 0.007, f"{means[i]:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(0.821, color=MUTED, lw=1, ls="--"); ax.text(1.5, 0.821 + 0.004, "B10 0.821", fontsize=7.5, color=MUTED, ha="center", va="bottom")
    ax.set_ylim(0.70, 0.96); ax.set_ylabel("ADE (m, 낮을수록 좋음)", fontsize=8, color=INK2)
    ax.set_title("PhysicalAI dev · head 4k · seed 17/29/43", fontsize=8.5, color=INK, loc="left")
    fig.text(0.5, 0.03, "막대 = seed 평균, 점 = 개별 seed. 두 벤치마크 모두 First가 Current보다 크게 좋고, Terminal은 First와 비슷하다.", ha="center", fontsize=8, color=INK2)
    fig.savefig(os.path.join(OUT, "fig7_timing_results.png")); plt.close(fig)

# =====================================================================
# Figure 8: data-scale effect
# =====================================================================
def fig8():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.22, wspace=0.3)
    ax = axes[0]; style_axes(ax)
    sizes = [4000, 20000, 92599]
    s17 = [86.60, 89.36, 90.43]; s29 = [86.89, 90.00, 90.58]
    ax.plot(sizes, s17, "-o", color=BLUE, lw=2, ms=5, mec="white", mew=1.2, label="seed 17", zorder=3)
    ax.plot(sizes, s29, "-o", color=ORANGE, lw=2, ms=5, mec="white", mew=1.2, label="seed 29", zorder=3)
    ax.axhline(90.93, color=MUTED, lw=1, ls="--"); ax.text(4000, 90.93 + 0.12, "B10 90.93", fontsize=7.5, color=MUTED, va="bottom")
    ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels(["4k", "20k", "92.6k"]); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(3000, 130000); ax.set_ylim(85.5, 91.8)
    ax.set_xlabel("head 학습 장면 수", fontsize=8, color=INK2); ax.set_ylabel("PDMS", fontsize=8, color=INK2)
    ax.set_title("NAVSIM navtest-dev · SF3-first", fontsize=8.5, color=INK, loc="left")
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
    ax.axhline(0.821, color=MUTED, lw=1, ls="--"); ax.text(4000, 0.821 + 0.003, "B10 0.821", fontsize=7.5, color=MUTED, va="bottom")
    ax.axhline(0.852, color=MUTED, lw=1, ls=":"); ax.text(4000, 0.852 + 0.003, "B* (B03-opt) 0.852", fontsize=7.5, color=MUTED, va="bottom")
    ax.set_xscale("log"); ax.set_xticks(sizes2); ax.set_xticklabels(["4k", "20k"]); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(3000, 28000); ax.set_ylim(0.74, 0.87)
    ax.set_xlabel("head 학습 clip 수", fontsize=8, color=INK2); ax.set_ylabel("ADE (m)", fontsize=8, color=INK2)
    ax.set_title("PhysicalAI dev · SF3-first", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.text(20000, 0.762 - 0.006, "0.762 (평균)", fontsize=7.5, color=INK, ha="center", va="top")
    ax.text(4000, 0.813 - 0.006, "0.813 (평균)", fontsize=7.5, color=INK, ha="center", va="top")
    fig.savefig(os.path.join(OUT, "fig8_scale.png")); plt.close(fig)

# =====================================================================
# Figure 9: quality vs latency
# =====================================================================
def fig9():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.27, wspace=0.3)
    ax = axes[0]; style_axes(ax)
    base = [("B10", 797.9, 89.67), ("B02-opt (B*)", 420.7, 89.65)]
    stud = [("제안 seed 29", 189.0, 89.49), ("제안 seed 17", 189.0, 89.25)]
    for nm, x, y in base:
        ax.plot(x, y, "o", ms=8, color=BLUE, mec="white", mew=1.5, zorder=3)
        ax.text(x, y + 0.09, nm, fontsize=7.5, color=INK, ha="center", va="bottom")
    for nm, x, y in stud:
        ax.plot(x, y, "o", ms=8, color=ORANGE, mec="white", mew=1.5, zorder=3)
    ax.text(189.0, 89.49 + 0.09, "제안 모델 seed 29", fontsize=7.5, color=INK, ha="center", va="bottom")
    ax.text(189.0, 89.25 - 0.09, "제안 모델 seed 17", fontsize=7.5, color=INK, ha="center", va="top")
    ax.set_xlim(0, 900); ax.set_ylim(88.8, 90.1)
    ax.set_xlabel("policy p50 지연 (ms, 같은 세션 교차 측정)", fontsize=8, color=INK2); ax.set_ylabel("PDMS", fontsize=8, color=INK2)
    ax.set_title("NAVSIM navtest-final (8,896장면)", fontsize=8.5, color=INK, loc="left")
    ax.plot([], [], "o", color=BLUE, label="DriveWAM 기준 모델"); ax.plot([], [], "o", color=ORANGE, label="제안 모델 (SF3-first)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax = axes[1]; style_axes(ax)
    base = [("B10-opt", 837.8, 0.802), ("B05-opt", 546.8, 0.809), ("B03-opt (B*)", 434.0, 0.842), ("B02-opt", 378.1, 0.891), ("B01-opt", 320.0, 1.272)]
    for nm, x, y in base:
        ax.plot(x, y, "o", ms=8, color=BLUE, mec="white", mew=1.5, zorder=3)
    ax.text(837.8, 0.802 + 0.02, "B10-opt", fontsize=7.5, color=INK, ha="center", va="bottom")
    ax.text(546.8, 0.809 + 0.02, "B05-opt", fontsize=7.5, color=INK, ha="center", va="bottom")
    ax.text(434.0 + 20, 0.842 + 0.02, "B03-opt (B*)", fontsize=7.5, color=INK, ha="left", va="bottom")
    ax.text(378.1 + 20, 0.891 + 0.01, "B02-opt", fontsize=7.5, color=INK, ha="left", va="bottom")
    ax.text(320.0 + 20, 1.272, "B01-opt", fontsize=7.5, color=INK, ha="left", va="center")
    for y in [0.750, 0.748, 0.762]:
        ax.plot(70.9, y, "o", ms=8, color=ORANGE, mec="white", mew=1.5, zorder=3)
    ax.text(70.9 + 25, 0.753, "제안 모델 (seed 3개)", fontsize=7.5, color=INK, ha="left", va="center")
    ax.set_xlim(0, 950); ax.set_ylim(0.7, 1.35)
    ax.set_xlabel("결정 지연 p50 (ms)", fontsize=8, color=INK2); ax.set_ylabel("ADE (m, 낮을수록 좋음)", fontsize=8, color=INK2)
    ax.set_title("PhysicalAI final (test 1k)", fontsize=8.5, color=INK, loc="left")
    fig.text(0.5, 0.025, "왼쪽 아래(빠르고 정확)일수록 좋다. 제안 모델은 두 벤치마크 모두 B*보다 왼쪽에 있다.", ha="center", fontsize=8, color=INK2)
    fig.savefig(os.path.join(OUT, "fig9_tradeoff.png")); plt.close(fig)

if __name__ == "__main__":
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]:
        f(); print("done", f.__name__)
