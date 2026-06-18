"""Fixed publication figures for DeCaF 2026 — overlapping elements corrected."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pickle
import os
import random
import sys
import io
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
})

COLORS = {
    "b3":    "#0072B2",
    "b2":    "#E69F00",
    "ours":  "#D55E00",
    "alpha": "#000000",
    "ref":   "#888888",
}

os.chdir(r"C:\DeCaf")
Path("paper_figures").mkdir(exist_ok=True)

# ─── LOAD ─────────────────────────────────────────────────────────────────────
for p in ["fed_crc_results/fets_final/volume_scores.pkl",
          "fed_crc_results/volume_scores.pkl"]:
    if Path(p).exists():
        with open(p, "rb") as f:
            volume_scores = pickle.load(f)
        print(f"Loaded {len(volume_scores)} subjects from {p}", flush=True)
        break

sample = volume_scores[next(iter(volume_scores))]
if "fnr_fine" in sample:
    CK, SK = "fnr_fine", "str_fine"
    lg = np.linspace(0, 1, len(sample[CK]))
elif "fnr_curve_fine" in sample:
    CK, SK = "fnr_curve_fine", "stretch_curve_fine"
    lg = np.linspace(0, 1, len(sample[CK]))
else:
    old_g = np.array([0.0,0.01,0.02,0.03,0.05,0.08,0.10,0.15,
                      0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,
                      0.80,0.90,0.95,0.99,1.0])
    lg = np.linspace(0, 1, 200)
    for d in volume_scores.values():
        ff = np.interp(lg, old_g[:len(d["fnr_curve"])], d["fnr_curve"])
        fs = np.interp(lg, old_g[:len(d.get("set_size_curve", d["fnr_curve"]))],
                       d.get("set_size_curve", np.ones(len(d["fnr_curve"]))))
        for j in range(1, len(ff)): ff[j] = min(ff[j], ff[j-1])
        d["fnr_fine"] = ff; d["str_fine"] = fs
    CK, SK = "fnr_fine", "str_fine"
    print("Interpolated to 200-pt grid", flush=True)

ss = defaultdict(list)
for s, d in volume_scores.items():
    if d.get("site_id") is not None:
        ss[d["site_id"]].append(s)
vs = {k: v for k, v in ss.items() if len(v) >= 6}
K = len(vs)
alpha, B = 0.10, 1.0

# CORRECT CRC direction: smallest lambda where risk <= alpha
def fls(rc, a):
    for j in range(len(lg)):
        if rc[j] <= a:
            return lg[j], j
    return lg[-1], len(lg) - 1

random.seed(42); np.random.seed(42)
cal, test = {}, {}
for sid, sl in vs.items():
    sh = list(sl); random.shuffle(sh); n = len(sh)
    cal[sid] = sh[:max(n//2, 3)]; test[sid] = sh[max(n//2, 3):]

sc = {}
for sid in cal:
    sc[sid] = np.array([volume_scores[s][CK] for s in cal[sid] if s in volume_scores])

ac = np.vstack([sc[sid] for sid in sc]); N = len(ac); Rg = ac.mean(0)

lb3, _ = fls(Rg + B/(N+1), alpha)

lb2 = {}
for sid in sc:
    nk = len(sc[sid])
    lb2[sid], _ = fls(sc[sid].mean(0) + B/(nk+1), alpha)

def cshrink(n0):
    lam = {}
    for sid in sc:
        nk = len(sc[sid]); w = nk/(nk+n0)
        Rs = w*sc[sid].mean(0) + (1-w)*Rg
        co = w*B/(nk+1) + (1-w)*B/(N+1)
        lam[sid], _ = fls(Rs+co, alpha)
    return lam

slams = {n0: cshrink(n0) for n0 in range(1, 31)}

def ev(lf):
    r = {}
    for sid in test:
        if not test[sid]: continue
        lam = lf(sid); j = int(np.argmin(np.abs(lg - lam)))
        fnrs = [volume_scores[s][CK][j] for s in test[sid] if s in volume_scores]
        strs = [volume_scores[s][SK][j] for s in test[sid] if s in volume_scores]
        r[sid] = {"fnr": float(np.mean(fnrs)) if fnrs else 1.0,
                  "stretch": float(np.mean(strs)) if strs else float("nan")}
    return r

rb3 = ev(lambda s: lb3)
rb2 = ev(lambda s: lb2[s])
rs  = {n0: ev(lambda s, d=slams[n0]: d[s]) for n0 in range(1, 31)}

tv = {sid: np.mean([volume_scores[s].get("tumor_volume", 0)
                    for s in vs[sid] if s in volume_scores])
      for sid in vs}
srt = sorted(vs, key=lambda s: tv[s])
x = np.arange(len(srt))
site_n = [len(vs[s]) for s in srt]

# ─── FIGURE 2: MONEY FIGURE (FIXED) ──────────────────────────────────────────
print("Generating Figure 2...", flush=True)
n0m = 15; ro = rs[n0m]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.8, 2.4))
fig.subplots_adjust(bottom=0.29, wspace=0.38, left=0.10, right=0.98, top=0.91)

for ax, ykey, ylabel, title, ylim, yscale in [
    (ax1, "fnr",    "Test FNR",               "Per-Institution Coverage", (-0.005, 0.22), "linear"),
    (ax2, "stretch", "Stretch $|C_\\lambda|/|Y|$", "Prediction Set Size",  None,           "log"),
]:
    for res, col, mk, lw in [
        (rb3, COLORS["b3"],   "o", 0.9),
        (rb2, COLORS["b2"],   "^", 0.9),
        (ro,  COLORS["ours"], "D", 1.3),
    ]:
        y = [res[sid][ykey] if sid in res and np.isfinite(res[sid][ykey]) else np.nan
             for sid in srt]
        ax.plot(x, y, color=col, marker=mk, linewidth=lw, markersize=3.5, alpha=0.88,
                markeredgewidth=0.3, markeredgecolor="white")

    if yscale == "linear":
        ax.axhline(alpha, color=COLORS["alpha"], linestyle="--", linewidth=1.0, alpha=0.65)
        if ylim: ax.set_ylim(*ylim)
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.1f}"))

    ax.set_title(title, fontweight="bold", pad=3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Institution size $n_k$", fontsize=7)

    # X-axis: show every other label if more than 15 sites
    ax.set_xticks(x)
    if len(srt) > 15:
        xlbls = [str(site_n[i]) if i % 2 == 0 else "" for i in range(len(srt))]
        ax.set_xticklabels(xlbls, rotation=0, fontsize=6)
    else:
        ax.set_xticklabels([str(n) for n in site_n], rotation=45, ha="right", fontsize=6)

# Shared legend, centred below both panels
handles_fig2 = [
    Line2D([0],[0], color=COLORS["b3"],   marker="o", markersize=4, linewidth=0.9,
           label="Naive Pooled (B3)"),
    Line2D([0],[0], color=COLORS["b2"],   marker="^", markersize=4, linewidth=0.9,
           label="Per-site Local (B2)"),
    Line2D([0],[0], color=COLORS["ours"], marker="D", markersize=4, linewidth=1.2,
           label=f"Ours ($n_0$={n0m})"),
    Line2D([0],[0], color=COLORS["alpha"], linestyle="--", linewidth=1.0,
           label=f"$\\alpha$={alpha}"),
]
fig.legend(handles=handles_fig2, loc="lower center", ncol=4, frameon=True,
           bbox_to_anchor=(0.5, 0.01), fontsize=6.5,
           handlelength=1.5, edgecolor="0.8", borderpad=0.3, columnspacing=0.9)

for ext in ["pdf", "png"]:
    fig.savefig(f"paper_figures/figure2_final.{ext}", bbox_inches="tight", pad_inches=0.05)
plt.close()
print("  Saved figure2_final.pdf / .png", flush=True)

# ─── FIGURE 3: n0 SWEEP (FIXED) ───────────────────────────────────────────────
print("Generating Figure 3...", flush=True)
nr   = list(range(1, 31))
viol = [sum(1 for sid in rs[n0] if rs[n0][sid]["fnr"] > alpha) for n0 in nr]
wfnr = [max(rs[n0][sid]["fnr"] for sid in rs[n0]) for n0 in nr]
strt = [np.mean([rs[n0][sid]["stretch"] for sid in rs[n0]
                 if np.isfinite(rs[n0][sid]["stretch"])]) for n0 in nr]

bv = sum(1 for sid in rb3 if rb3[sid]["fnr"] > alpha)
bw = max(rb3[sid]["fnr"] for sid in rb3)
bs = np.mean([rb3[sid]["stretch"] for sid in rb3 if np.isfinite(rb3[sid]["stretch"])])
b2v = sum(1 for sid in rb2 if rb2[sid]["fnr"] > alpha)
b2s = np.mean([rb2[sid]["stretch"] for sid in rb2 if np.isfinite(rb2[sid]["stretch"])])

fig, axes = plt.subplots(1, 3, figsize=(4.8, 2.0))
fig.subplots_adjust(bottom=0.30, wspace=0.50, left=0.09, right=0.98, top=0.88)

panel_cfgs = [
    (axes[0], viol, "Violations /20",  "Coverage",   None,    "linear"),
    (axes[1], wfnr, "Worst-site FNR",  "Worst Case", None,    "linear"),
    (axes[2], strt, "Avg. Stretch",    "Efficiency", None,    "log"),
]

for ax, ydata, ylabel, title, ylim, yscale in panel_cfgs:
    ax.plot(nr, ydata, color=COLORS["ours"], marker="o", markersize=2.5, linewidth=1.0, zorder=3)

    # Shaded sweet-spot and vertical operating-point lines
    ax.axvspan(10, 20, alpha=0.04, color="gray", zorder=0)
    for xv in [9, 15]:
        ax.axvline(xv, color="gray", linestyle=":", linewidth=0.5, alpha=0.6, zorder=1)

    # B3/B2 reference lines
    if "Violations" in ylabel:
        ax.axhline(bv,  color=COLORS["b3"], linestyle="--", linewidth=0.7, alpha=0.6, zorder=2)
        ax.axhline(b2v, color=COLORS["b2"], linestyle="--", linewidth=0.7, alpha=0.6, zorder=2)
    elif "FNR" in ylabel:
        ax.axhline(alpha, color=COLORS["alpha"], linestyle="--", linewidth=0.8, alpha=0.5, zorder=2)
        ax.axhline(bw,    color=COLORS["b3"],    linestyle=":",  linewidth=0.6, alpha=0.4, zorder=2)
    else:
        ax.axhline(bs,  color=COLORS["b3"], linestyle="--", linewidth=0.7, alpha=0.6, zorder=2)
        ax.axhline(b2s, color=COLORS["b2"], linestyle="--", linewidth=0.7, alpha=0.6, zorder=2)

    if yscale == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.0f}" if v >= 10 else f"{v:.1f}"))

    ax.set_title(title, fontweight="bold", fontsize=8, pad=2)
    ax.set_xlabel("$n_0$"); ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, 30.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])

    # Label the operating-point verticals BELOW the x-axis
    for xv, lbl in [(9, "9"), (15, "15")]:
        ax.annotate(lbl, xy=(xv, ax.get_ylim()[0]),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -10), textcoords="offset points",
                    ha="center", fontsize=5.5, color="gray")

# Shared legend below all three panels
handles_fig3 = [
    Line2D([0],[0], color=COLORS["ours"], marker="o", markersize=3, linewidth=1.0,
           label="Shrinkage CRC (ours)"),
    Line2D([0],[0], color=COLORS["b3"],   linestyle="--", linewidth=0.7, label="B3 Naive Pooled"),
    Line2D([0],[0], color=COLORS["b2"],   linestyle="--", linewidth=0.7, label="B2 Per-site Local"),
    Line2D([0],[0], color=COLORS["alpha"], linestyle="--", linewidth=0.8, label=f"$\\alpha$={alpha}"),
    Line2D([0],[0], color="gray", linestyle=":", linewidth=0.5, label="$n_0$=9, 15"),
]
fig.legend(handles=handles_fig3, loc="lower center", ncol=5, frameon=True,
           bbox_to_anchor=(0.5, 0.01), fontsize=6,
           handlelength=1.5, edgecolor="0.8", borderpad=0.3, columnspacing=0.7)

for ext in ["pdf", "png"]:
    fig.savefig(f"paper_figures/n0_sweep_final.{ext}", bbox_inches="tight", pad_inches=0.05)
plt.close()
print("  Saved n0_sweep_final.pdf / .png", flush=True)

# ─── VERIFY ───────────────────────────────────────────────────────────────────
print("\nVerification:")
all_ok = True
for fname in ["figure2_final.pdf", "figure2_final.png",
              "n0_sweep_final.pdf", "n0_sweep_final.png"]:
    p = Path("paper_figures") / fname
    if p.exists():
        print(f"  OK  {fname}  ({p.stat().st_size//1024} KB)")
    else:
        print(f"  MISSING: {fname}"); all_ok = False

n15 = rs[15]
print(f"\nCaption numbers (seed 42):")
print(f"  B3:  {bv}/{K} viol, worst={bw:.3f} (+{(bw-alpha)*100:.1f}pp), stretch={bs:.1f}x")
print(f"  B2:  {b2v}/{K} viol, worst={max(rb2[s]['fnr'] for s in rb2):.3f}, stretch={b2s:.1f}x")
print(f"  Ours(n0=15): {sum(1 for s in n15 if n15[s]['fnr']>alpha)}/{K} viol, "
      f"worst={max(n15[s]['fnr'] for s in n15):.3f}, "
      f"stretch={np.mean([n15[s]['stretch'] for s in n15 if np.isfinite(n15[s]['stretch'])]):.1f}x")
print(f"\n{'ALL OK' if all_ok else 'ERRORS'} — C:\\DeCaf\\paper_figures\\")
