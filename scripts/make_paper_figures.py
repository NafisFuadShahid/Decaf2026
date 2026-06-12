"""
Generate publication-quality figures for DeCaF 2026 (MICCAI workshop).
LNCS Springer camera-ready format.
Figures: Figure 2 (money figure) + Figure 3 (n0 sweep).
"""
import os, sys, io, pickle, random
import numpy as np
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.chdir(r"C:\DeCaf")
Path("paper_figures").mkdir(exist_ok=True)

# ─── LNCS style ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.3,
    "axes.grid": False,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.borderpad": 0.4,
    "legend.handlelength": 1.5,
})

# Okabe-Ito colorblind-safe palette
COLORS = {
    "b3":         "#0072B2",  # blue — naive pooled
    "b2":         "#E69F00",  # orange — per-site local
    "ours":       "#D55E00",  # vermillion — our method (n0=15)
    "ours_alt":   "#009E73",  # green — alternative operating point
    "alpha_line": "#000000",
}
MARKERS = {"b3": "o", "b2": "^", "ours": "D", "ours_alt": "s"}
LNCS_W = 4.8   # full text width in inches for LNCS

def save(fig, stem):
    for ext in ["pdf", "png"]:
        p = f"paper_figures/{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.05)
        kb = Path(p).stat().st_size / 1024
        print(f"  Saved {p}  ({kb:.0f} KB)", flush=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
volume_scores = None
for p in ["fed_crc_results/fets_final/volume_scores.pkl",
          "fed_crc_results/volume_scores.pkl"]:
    if Path(p).exists():
        with open(p, "rb") as f:
            volume_scores = pickle.load(f)
        print(f"  {len(volume_scores)} subjects from {p}", flush=True)
        break
if volume_scores is None:
    print("[FATAL] No volume_scores.pkl found"); sys.exit(1)

sample = volume_scores[next(iter(volume_scores))]

# Use/build fine 200-pt curves
if "fnr_fine" in sample:
    CK, SK = "fnr_fine", "str_fine"
    fine_grid = np.linspace(0, 1, len(sample[CK]))
elif "fnr_curve_fine" in sample:
    CK, SK = "fnr_curve_fine", "stretch_curve_fine"
    fine_grid = np.linspace(0, 1, len(sample[CK]))
else:
    old_grid = np.array([0.0,0.01,0.02,0.03,0.05,0.08,0.10,0.15,
                         0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,
                         0.80,0.90,0.95,0.99,1.0])
    fine_grid = np.linspace(0, 1, 200)
    for d in volume_scores.values():
        f = np.interp(fine_grid, old_grid[:len(d["fnr_curve"])], d["fnr_curve"])
        s = np.interp(fine_grid, old_grid[:len(d.get("set_size_curve", d["fnr_curve"]))],
                      d.get("set_size_curve", np.ones(len(d["fnr_curve"]))))
        for j in range(1, len(f)): f[j] = min(f[j], f[j-1])
        d["fnr_fine"] = f; d["str_fine"] = s
    CK, SK = "fnr_fine", "str_fine"
    print(f"  Interpolated to 200-pt grid", flush=True)

# Sites
site_map = defaultdict(list)
for subj, d in volume_scores.items():
    if d.get("site_id") is not None:
        site_map[d["site_id"]].append(subj)
valid_sites = {k: v for k, v in site_map.items() if len(v) >= 6}
K = len(valid_sites)
print(f"  {K} valid sites", flush=True)

alpha, B = 0.10, 1.0

# ─── CRC (correct direction: smallest lambda where risk <= alpha) ──────────────
def find_lam(risk, alpha):
    for j in range(len(fine_grid)):
        if risk[j] <= alpha:
            return fine_grid[j], j
    return fine_grid[-1], len(fine_grid) - 1

# ─── CAL/TEST SPLIT seed=42 ───────────────────────────────────────────────────
random.seed(42); np.random.seed(42)
cal, tst = {}, {}
for sid, subjs in valid_sites.items():
    sh = list(subjs); random.shuffle(sh)
    n_cal = max(len(sh)//2, 3)
    cal[sid] = sh[:n_cal]; tst[sid] = sh[n_cal:]

sc = {}
for sid in cal:
    curves = [volume_scores[s][CK] for s in cal[sid] if s in volume_scores]
    if curves: sc[sid] = np.array(curves)

all_cc = np.vstack([sc[sid] for sid in sc])
N_cal  = len(all_cc)
Rg     = all_cc.mean(0)

# B3
lb3, _ = find_lam(Rg + B/(N_cal+1), alpha)

# B2
lb2 = {}
for sid in sc:
    n_k = len(sc[sid])
    lb2[sid], _ = find_lam(sc[sid].mean(0) + B/(n_k+1), alpha)

# Shrinkage over n0 = 1..30
def shrink_lams(n0):
    ls = {}
    for sid in sc:
        n_k = len(sc[sid])
        w   = n_k/(n_k+n0)
        Rs  = w*sc[sid].mean(0) + (1-w)*Rg
        c_  = w*B/(n_k+1) + (1-w)*B/(N_cal+1)
        ls[sid], _ = find_lam(Rs+c_, alpha)
    return ls

shrink = {n0: shrink_lams(n0) for n0 in range(1, 31)}

# Evaluate
def evaluate(lam_fn):
    out = {}
    for sid in tst:
        if not tst[sid]: continue
        lam = lam_fn(sid)
        j   = int(np.argmin(np.abs(fine_grid - lam)))
        fnrs = [volume_scores[s][CK][j] for s in tst[sid] if s in volume_scores]
        strs = [volume_scores[s][SK][j] for s in tst[sid] if s in volume_scores]
        out[sid] = {"fnr": float(np.mean(fnrs)) if fnrs else 1.0,
                    "stretch": float(np.mean(strs)) if strs else float("nan"),
                    "n": len(tst[sid])}
    return out

rb3 = evaluate(lambda s: lb3)
rb2 = evaluate(lambda s: lb2[s])
rs  = {n0: evaluate(lambda s, d=shrink[n0]: d[s]) for n0 in range(1, 31)}

# Site ordering: ascending tumor prevalence
tvol = {sid: np.mean([volume_scores[s].get("tumor_volume",0)
                       for s in valid_sites[sid] if s in volume_scores])
        for sid in valid_sites}
sorted_sites = sorted(valid_sites, key=lambda s: tvol[s])
x = np.arange(len(sorted_sites))

# ─── FIGURE 2: MONEY FIGURE ───────────────────────────────────────────────────
print("\nGenerating Figure 2 (money figure)...", flush=True)

n0_main = 15
res_ours = rs[n0_main]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(LNCS_W, 2.2))

# Common plot config
methods_cfg = [
    ("Naive Pooled (B3)", rb3,       COLORS["b3"],       MARKERS["b3"],   3, 1.0),
    ("Per-site Local (B2)", rb2,     COLORS["b2"],       MARKERS["b2"],   2, 0.8),
    (f"Ours ($n_0={n0_main}$)", res_ours, COLORS["ours"], MARKERS["ours"], 4, 1.4),
]

# Left: FNR per site
for label, res, color, marker, zo, lw in methods_cfg:
    y = [res[sid]["fnr"] if sid in res else np.nan for sid in sorted_sites]
    ax1.plot(x, y, color=color, marker=marker, label=label, linewidth=lw,
             markersize=4, alpha=0.88, zorder=zo,
             markeredgewidth=0.3, markeredgecolor="white")

ax1.axhline(alpha, color=COLORS["alpha_line"], linestyle="--", linewidth=0.9,
            alpha=0.55, label=f"$\\alpha$={alpha}", zorder=1)

b3_max_fnr = max(rb3[sid]["fnr"] for sid in sorted_sites if sid in rb3)
ax1.set_ylim(-0.01, max(0.22, b3_max_fnr + 0.025))
ax1.set_xlabel("Institution $n_k$ (sorted by tumor prevalence →)", fontsize=8)
ax1.set_ylabel("Test-set FNR")
ax1.set_title("Per-Institution Coverage", fontweight="bold", pad=4)
ax1.set_xticks(x)
ax1.set_xticklabels([str(len(valid_sites[s])) for s in sorted_sites],
                    rotation=55, ha="right")
ax1.tick_params(axis="x", pad=1)
ax1.legend(loc="upper left", frameon=True, ncol=1, borderaxespad=0.3)

# Add red background for B3-violating sites
for i, sid in enumerate(sorted_sites):
    if sid in rb3 and rb3[sid]["fnr"] > alpha:
        ax1.axvspan(i - 0.45, i + 0.45, ymin=0, ymax=1,
                    color="#FFCCCC", alpha=0.35, zorder=0)

# Right: Stretch per site (log scale)
for label, res, color, marker, zo, lw in methods_cfg:
    y = [res[sid]["stretch"] if (sid in res and np.isfinite(res[sid]["stretch"]))
         else np.nan for sid in sorted_sites]
    ax2.plot(x, y, color=color, marker=marker, label=label, linewidth=lw,
             markersize=4, alpha=0.88, zorder=zo,
             markeredgewidth=0.3, markeredgecolor="white")

ax2.set_yscale("log")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.1f}"))
ax2.set_xlabel("Institution $n_k$ (sorted by tumor prevalence →)", fontsize=8)
ax2.set_ylabel("Stretch $|C_\\lambda|\\,/\\,|Y|$")
ax2.set_title("Prediction Set Size", fontweight="bold", pad=4)
ax2.set_xticks(x)
ax2.set_xticklabels([str(len(valid_sites[s])) for s in sorted_sites],
                    rotation=55, ha="right")
ax2.tick_params(axis="x", pad=1)
ax2.legend(loc="upper right", frameon=True, ncol=1, borderaxespad=0.3)

plt.tight_layout(w_pad=1.5)
save(fig, "figure2_final")
plt.close()

# ─── FIGURE 3: n0 SWEEP ───────────────────────────────────────────────────────
print("\nGenerating Figure 3 (n0 sweep)...", flush=True)

n0_range = list(range(1, 31))
violations = [sum(1 for sid in rs[n0] if rs[n0][sid]["fnr"] > alpha) for n0 in n0_range]
worst_fnrs = [max(rs[n0][sid]["fnr"] for sid in rs[n0]) for n0 in n0_range]
stretches  = [np.mean([rs[n0][sid]["stretch"] for sid in rs[n0] if np.isfinite(rs[n0][sid]["stretch"])])
              for n0 in n0_range]

b3_viol    = sum(1 for sid in rb3 if rb3[sid]["fnr"] > alpha)
b3_worst   = max(rb3[sid]["fnr"] for sid in rb3)
b3_stretch = np.mean([rb3[sid]["stretch"] for sid in rb3 if np.isfinite(rb3[sid]["stretch"])])
b2_viol    = sum(1 for sid in rb2 if rb2[sid]["fnr"] > alpha)
b2_worst   = max(rb2[sid]["fnr"] for sid in rb2)
b2_stretch = np.mean([rb2[sid]["stretch"] for sid in rb2 if np.isfinite(rb2[sid]["stretch"])])

fig, axes = plt.subplots(1, 3, figsize=(LNCS_W, 1.75))

# Shaded sweet spot (n0=10–20)
for ax in axes:
    ax.axvspan(10, 20, alpha=0.08, color="gray", zorder=0)

# Panel 1: Violations
ax = axes[0]
ax.plot(n0_range, violations, color=COLORS["ours"], marker="o",
        markersize=3, linewidth=1.2, zorder=3)
ax.axhline(b3_viol, color=COLORS["b3"], linestyle="--", linewidth=0.8,
           alpha=0.7, label=f"B3 ({b3_viol})", zorder=2)
ax.axhline(b2_viol, color=COLORS["b2"], linestyle="--", linewidth=0.8,
           alpha=0.7, label=f"B2 ({b2_viol:.0f})", zorder=2)
ax.set_xlabel("$n_0$")
ax.set_ylabel("Violations / 20")
ax.set_title("Coverage", fontweight="bold", pad=3)
ax.set_xlim(0.5, 30.5)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.legend(fontsize=6, loc="upper left", handlelength=1.2, borderaxespad=0.3)

# Panel 2: Worst-site FNR
ax = axes[1]
ax.plot(n0_range, worst_fnrs, color=COLORS["ours"], marker="o",
        markersize=3, linewidth=1.2, zorder=3)
ax.axhline(alpha, color=COLORS["alpha_line"], linestyle="--", linewidth=0.9,
           alpha=0.5, label=f"$\\alpha$={alpha}", zorder=1)
ax.axhline(b3_worst, color=COLORS["b3"], linestyle=":", linewidth=0.8,
           alpha=0.5, label=f"B3 ({b3_worst:.2f})", zorder=2)
ax.set_xlabel("$n_0$")
ax.set_ylabel("Worst-site FNR")
ax.set_title("Worst Case", fontweight="bold", pad=3)
ax.set_xlim(0.5, 30.5)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.legend(fontsize=6, loc="lower right", handlelength=1.2, borderaxespad=0.3)

# Panel 3: Stretch (log)
ax = axes[2]
ax.semilogy(n0_range, stretches, color=COLORS["ours"], marker="o",
            markersize=3, linewidth=1.2, zorder=3)
ax.axhline(b3_stretch, color=COLORS["b3"], linestyle="--", linewidth=0.8,
           alpha=0.7, label=f"B3 ({b3_stretch:.1f}×)", zorder=2)
ax.axhline(b2_stretch, color=COLORS["b2"], linestyle="--", linewidth=0.8,
           alpha=0.7, label=f"B2 ({b2_stretch:.0f}×)", zorder=2)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{v:.0f}" if v >= 10 else f"{v:.1f}"))
ax.set_xlabel("$n_0$")
ax.set_ylabel("Avg. Stretch")
ax.set_title("Efficiency", fontweight="bold", pad=3)
ax.set_xlim(0.5, 30.5)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.legend(fontsize=6, loc="upper right", handlelength=1.2, borderaxespad=0.3)

# Annotate operating points on all panels
for ax, yvals in [(axes[0], violations), (axes[1], worst_fnrs), (axes[2], stretches)]:
    for n0_ann in [9, 15]:
        idx = n0_range.index(n0_ann)
        ax.annotate(f"$n_0$={n0_ann}", (n0_ann, yvals[idx]),
                    fontsize=5.5, textcoords="offset points", xytext=(3, 4),
                    color="#444444")

plt.tight_layout(w_pad=0.7)
save(fig, "n0_sweep_final")
plt.close()

# ─── VERIFY ───────────────────────────────────────────────────────────────────
print("\nVerification:")
expected = ["figure2_final.pdf", "figure2_final.png",
            "n0_sweep_final.pdf", "n0_sweep_final.png"]
all_ok = True
for fname in expected:
    p = Path("paper_figures") / fname
    if p.exists():
        print(f"  OK  {fname}  ({p.stat().st_size//1024} KB)")
    else:
        print(f"  MISSING: {fname}")
        all_ok = False

# Print key numbers for caption writing
print(f"\nCaption data (seed 42):")
print(f"  B3:  {b3_viol}/{K} violations, worst FNR={b3_worst:.3f} (+{(b3_worst-alpha)*100:.1f}pp), stretch={b3_stretch:.1f}x")
print(f"  B2:  {b2_viol}/{K} violations, worst FNR={b2_worst:.3f}, stretch={b2_stretch:.1f}x")
n15 = rs[15]
print(f"  Ours(n0=15): {sum(1 for s in n15 if n15[s]['fnr']>alpha)}/{K} violations, "
      f"worst FNR={max(n15[s]['fnr'] for s in n15):.3f}, "
      f"stretch={np.mean([n15[s]['stretch'] for s in n15]):.1f}x")
print(f"\n{'OK' if all_ok else 'ERRORS'}: figures in C:\\DeCaf\\paper_figures\\")
