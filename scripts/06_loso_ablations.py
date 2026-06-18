"""
Fed-CRC-Seg Phase 6: LOSO-CV for n0 Selection + lambda_fed Stretch Computation
- Task 1: Leave-One-Site-Out CV over n0 in {1,3,5,7,9,11,13,15,17,19,21,25,30,40,50}
- Task 2: lambda_fed = max(lambda_pool, max_k lambda_k_shrink) -- Theorem 1 conservative threshold
- Task 3a: corr_k ablation (with vs without correction term)
- Task 3b: lambda-grid resolution ablation (G in {50,100,200,500}) at n0=15
Pure CPU calibration math on existing FeTS volume scores. No re-inference.
"""
import os, sys, io, pickle, random, datetime, csv
import numpy as np
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.chdir(r"C:\DeCaf")
OUT = Path("fed_crc_results/phase6_loso")
OUT.mkdir(parents=True, exist_ok=True)

alpha = 0.10
B = 1.0
SEEDS = [42, 1337, 2024]
MIN_SITE = 6

print("=" * 60)
print("PHASE 6: LOSO-CV n0 SELECTION + LAMBDA_FED STRETCH")
print("=" * 60, flush=True)

# ─── LOAD DATA ──────────────────────────────────────────────────────────────
volume_scores = None
for p in ["fed_crc_results/fets_final/volume_scores.pkl",
          "fed_crc_results/volume_scores.pkl"]:
    if Path(p).exists():
        with open(p, "rb") as f:
            volume_scores = pickle.load(f)
        print(f"Loaded {len(volume_scores)} subjects from {p}", flush=True)
        break
if volume_scores is None:
    print("[FATAL] No volume_scores.pkl found"); sys.exit(1)

sample = volume_scores[next(iter(volume_scores))]
old_len = len(sample["fnr_curve"])
old_grid = np.array([0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15,
                      0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70,
                      0.80, 0.90, 0.95, 0.99, 1.0])
if old_len != len(old_grid):
    old_grid = np.linspace(0, 1, old_len)


def build_fine(G):
    """Interpolate raw fnr_curve/set_size_curve to a G-point uniform grid."""
    grid = np.linspace(0.0, 1.0, G)
    fnr_d, str_d = {}, {}
    for subj, d in volume_scores.items():
        old_fnr = np.array(d["fnr_curve"])
        old_str = np.array(d.get("set_size_curve", np.ones(old_len)))
        fnr = np.interp(grid, old_grid[:len(old_fnr)], old_fnr)
        srt = np.interp(grid, old_grid[:len(old_str)], old_str)
        for j in range(1, len(fnr)):
            fnr[j] = min(fnr[j], fnr[j - 1])
        fnr_d[subj] = fnr
        str_d[subj] = srt
    return grid, fnr_d, str_d


G_MAIN = 200
grid200, fnr200, str200 = build_fine(G_MAIN)
print(f"Main grid: G={G_MAIN}", flush=True)

# ─── SITE GROUPING ────────────────────────────────────────────────────────────
site_map = defaultdict(list)
for subj, d in volume_scores.items():
    if d.get("site_id") is not None:
        site_map[d["site_id"]].append(subj)
valid_sites = {k: v for k, v in site_map.items() if len(v) >= MIN_SITE}
K = len(valid_sites)
print(f"Sites: {K}, Subjects: {sum(len(v) for v in valid_sites.values())}", flush=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def find_lam(curve, grid, a=alpha):
    """Smallest lambda where curve[j] <= alpha (correct CRC direction)."""
    for j in range(len(grid)):
        if curve[j] <= a:
            return grid[j], j
    return grid[-1], len(grid) - 1


def eval_lambda(lam, subs, grid, fnr_d, str_d):
    j = int(np.argmin(np.abs(grid - lam)))
    fnrs = [fnr_d[s][j] for s in subs if s in fnr_d]
    strs = [str_d[s][j] for s in subs if s in str_d]
    return (float(np.mean(fnrs)) if fnrs else 1.0,
            float(np.mean(strs)) if strs else float("nan"))


def summarize(fnrs, strs):
    return {"viol": int(sum(1 for f in fnrs if f > alpha)),
            "worst": float(max(fnrs)),
            "stretch": float(np.mean(strs))}


def make_splits(seed):
    random.seed(seed); np.random.seed(seed)
    cal, tst = {}, {}
    for sid, subjs in valid_sites.items():
        sh = list(subjs); random.shuffle(sh)
        n_cal = max(len(sh) // 2, 3)
        cal[sid] = sh[:n_cal]; tst[sid] = sh[n_cal:]
    return cal, tst


# ════════════════════════════════════════════════════════════════════════════
# TASK 1: LOSO-CV for n0 selection
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 1: LOSO-CV n0 SELECTION")
print("=" * 60, flush=True)

n0_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 30, 40, 50]
task1_per_seed = {n0: [] for n0 in n0_list}

for seed in SEEDS:
    print(f"\n=== Seed {seed} ===", flush=True)
    cal, tst = make_splits(seed)
    site_cc = {sid: np.array([fnr200[s] for s in cal[sid] if s in fnr200]) for sid in cal}

    # Precompute leave-one-site-out global curves (n0-independent)
    loo_global = {}
    for h in valid_sites:
        others = [sid for sid in valid_sites if sid != h]
        other_cc = np.vstack([site_cc[sid] for sid in others])
        loo_global[h] = (other_cc.mean(0), len(other_cc))

    for n0 in n0_list:
        site_fnr, site_str = [], []
        for h in valid_sites:
            n_h = len(site_cc[h])
            R_h_cal = site_cc[h].mean(0)
            R_glob_noh, N_noh = loo_global[h]
            w_h = n_h / (n_h + n0)
            corr_h = w_h * B / (n_h + 1) + (1 - w_h) * B / (N_noh + 1)
            R_shrink = w_h * R_h_cal + (1 - w_h) * R_glob_noh + corr_h
            lam_h, _ = find_lam(R_shrink, grid200)
            mfnr, mstr = eval_lambda(lam_h, tst[h], grid200, fnr200, str200)
            site_fnr.append(mfnr); site_str.append(mstr)

        task1_per_seed[n0].append({
            "viol": int(sum(1 for f in site_fnr if f > alpha)),
            "worst": float(max(site_fnr)),
            "mean_stretch": float(np.mean(site_str)),
            "median_stretch": float(np.median(site_str)),
        })
    print(f"  Done {len(n0_list)} n0 values", flush=True)

# Aggregate over seeds
agg1 = {}
for n0 in n0_list:
    rl = task1_per_seed[n0]
    agg1[n0] = {
        "viol_mean": float(np.mean([r["viol"] for r in rl])),
        "viol_std":  float(np.std([r["viol"] for r in rl])),
        "worst_mean": float(np.mean([r["worst"] for r in rl])),
        "worst_std":  float(np.std([r["worst"] for r in rl])),
        "stretch_mean": float(np.mean([r["mean_stretch"] for r in rl])),
        "stretch_std":  float(np.std([r["mean_stretch"] for r in rl])),
        "medstretch_mean": float(np.mean([r["median_stretch"] for r in rl])),
        "medstretch_std":  float(np.std([r["median_stretch"] for r in rl])),
    }

print(f"\n{'n0':>4} {'Violations':>14} {'Worst FNR':>16} {'Mean Stretch':>16} {'Median Stretch':>16}")
for n0 in n0_list:
    a = agg1[n0]
    print(f"{n0:>4} {a['viol_mean']:>6.1f}+-{a['viol_std']:.1f}   "
          f"{a['worst_mean']:>7.4f}+-{a['worst_std']:.4f}  "
          f"{a['stretch_mean']:>7.2f}+-{a['stretch_std']:.2f}   "
          f"{a['medstretch_mean']:>7.2f}+-{a['medstretch_std']:.2f}", flush=True)

# LOSO-selected n0
candidates = [n0 for n0 in n0_list if agg1[n0]["viol_mean"] <= 3]
if candidates:
    loso_n0 = min(candidates, key=lambda n0: agg1[n0]["stretch_mean"])
    select_reason = "lowest mean stretch among n0 with violations <= 3 (mean over seeds)"
else:
    loso_n0 = min(n0_list, key=lambda n0: (agg1[n0]["viol_mean"], agg1[n0]["stretch_mean"]))
    select_reason = "fewest violations (no n0 achieved <= 3 violations)"

print(f"\nLOSO-selected n0 = {loso_n0}  ({select_reason})", flush=True)
print(f"Paper's reported n0 = 15", flush=True)


# ════════════════════════════════════════════════════════════════════════════
# TASK 2: lambda_fed conservative threshold (Theorem 1)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 2: LAMBDA_FED CONSERVATIVE THRESHOLD")
print("=" * 60, flush=True)

n0_task2 = [9, 15, 20]
task2_per_seed = defaultdict(list)
task3a_per_seed = defaultdict(list)

for seed in SEEDS:
    print(f"\n=== Seed {seed} ===", flush=True)
    cal, tst = make_splits(seed)
    site_cc = {sid: np.array([fnr200[s] for s in cal[sid] if s in fnr200]) for sid in cal}
    all_cc = np.vstack([site_cc[sid] for sid in site_cc])
    N_pool = len(all_cc)
    R_global = all_cc.mean(0)

    # B3-style pooled threshold
    lam_pool, _ = find_lam(R_global + B / (N_pool + 1), grid200)

    for n0 in n0_task2:
        # Per-site shrinkage thresholds (with corr_k) -- "already have this"
        lam_k = {}
        lam_k_nocorr = {}
        for sid in site_cc:
            n_k = len(site_cc[sid])
            w_k = n_k / (n_k + n0)
            corr_k = w_k * B / (n_k + 1) + (1 - w_k) * B / (N_pool + 1)
            R_shrink = w_k * site_cc[sid].mean(0) + (1 - w_k) * R_global
            lam_k[sid], _ = find_lam(R_shrink + corr_k, grid200)
            lam_k_nocorr[sid], _ = find_lam(R_shrink, grid200)  # Task 3a: corr_k = 0

        lam_fed = max(lam_pool, max(lam_k.values()))

        # lambda_fed: same single threshold deployed at every site
        fnrs, strs = [], []
        for sid in valid_sites:
            f, s = eval_lambda(lam_fed, tst[sid], grid200, fnr200, str200)
            fnrs.append(f); strs.append(s)
        task2_per_seed[f"lambda_fed n0={n0}"].append(summarize(fnrs, strs))

        # Per-site lambda_k (with corr_k)
        fnrs, strs = [], []
        for sid in valid_sites:
            f, s = eval_lambda(lam_k[sid], tst[sid], grid200, fnr200, str200)
            fnrs.append(f); strs.append(s)
        task2_per_seed[f"per_site n0={n0}"].append(summarize(fnrs, strs))

        # Task 3a: per-site lambda_k without corr_k
        fnrs, strs = [], []
        for sid in valid_sites:
            f, s = eval_lambda(lam_k_nocorr[sid], tst[sid], grid200, fnr200, str200)
            fnrs.append(f); strs.append(s)
        task3a_per_seed[f"n0={n0} nocorr"].append(summarize(fnrs, strs))

    print(f"  lambda_pool={lam_pool:.4f}", flush=True)

# Aggregate
agg2 = {}
for m, rl in task2_per_seed.items():
    agg2[m] = {
        "viol_mean": float(np.mean([r["viol"] for r in rl])),
        "viol_std":  float(np.std([r["viol"] for r in rl])),
        "worst_mean": float(np.mean([r["worst"] for r in rl])),
        "worst_std":  float(np.std([r["worst"] for r in rl])),
        "stretch_mean": float(np.mean([r["stretch"] for r in rl])),
        "stretch_std":  float(np.std([r["stretch"] for r in rl])),
    }

agg3a = {}
for m, rl in task3a_per_seed.items():
    agg3a[m] = {
        "viol_mean": float(np.mean([r["viol"] for r in rl])),
        "viol_std":  float(np.std([r["viol"] for r in rl])),
        "worst_mean": float(np.mean([r["worst"] for r in rl])),
        "worst_std":  float(np.std([r["worst"] for r in rl])),
        "stretch_mean": float(np.mean([r["stretch"] for r in rl])),
        "stretch_std":  float(np.std([r["stretch"] for r in rl])),
    }

print(f"\n{'Method':<22} {'Violations':>14} {'Worst FNR':>16} {'Mean Stretch':>16}")
for m in [f"lambda_fed n0={n0}" for n0 in n0_task2] + [f"per_site n0={n0}" for n0 in n0_task2]:
    a = agg2[m]
    print(f"{m:<22} {a['viol_mean']:>6.1f}+-{a['viol_std']:.1f}   "
          f"{a['worst_mean']:>7.4f}+-{a['worst_std']:.4f}  "
          f"{a['stretch_mean']:>7.2f}+-{a['stretch_std']:.2f}", flush=True)


# ════════════════════════════════════════════════════════════════════════════
# TASK 3b: lambda-grid resolution ablation (n0=15)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 3b: GRID RESOLUTION ABLATION (n0=15)")
print("=" * 60, flush=True)

G_list = [50, 100, 200, 500]
fine_cache = {200: (grid200, fnr200, str200)}
for G in G_list:
    if G not in fine_cache:
        fine_cache[G] = build_fine(G)

task3b_per_seed = defaultdict(list)
n0_3b = 15

for seed in SEEDS:
    cal, tst = make_splits(seed)
    for G in G_list:
        grid, fnr_d, str_d = fine_cache[G]
        site_cc = {sid: np.array([fnr_d[s] for s in cal[sid] if s in fnr_d]) for sid in cal}
        all_cc = np.vstack([site_cc[sid] for sid in site_cc])
        N_pool = len(all_cc)
        R_global = all_cc.mean(0)

        lam_k = {}
        for sid in site_cc:
            n_k = len(site_cc[sid])
            w_k = n_k / (n_k + n0_3b)
            corr_k = w_k * B / (n_k + 1) + (1 - w_k) * B / (N_pool + 1)
            R_shrink = w_k * site_cc[sid].mean(0) + (1 - w_k) * R_global + corr_k
            lam_k[sid], _ = find_lam(R_shrink, grid)

        fnrs, strs = [], []
        for sid in valid_sites:
            f, s = eval_lambda(lam_k[sid], tst[sid], grid, fnr_d, str_d)
            fnrs.append(f); strs.append(s)
        task3b_per_seed[G].append(summarize(fnrs, strs))
    print(f"  Seed {seed} done", flush=True)

agg3b = {}
for G, rl in task3b_per_seed.items():
    agg3b[G] = {
        "viol_mean": float(np.mean([r["viol"] for r in rl])),
        "viol_std":  float(np.std([r["viol"] for r in rl])),
        "worst_mean": float(np.mean([r["worst"] for r in rl])),
        "worst_std":  float(np.std([r["worst"] for r in rl])),
        "stretch_mean": float(np.mean([r["stretch"] for r in rl])),
        "stretch_std":  float(np.std([r["stretch"] for r in rl])),
    }

print(f"\n{'G':>5} {'Violations':>14} {'Worst FNR':>16} {'Stretch':>16}")
for G in G_list:
    a = agg3b[G]
    print(f"{G:>5} {a['viol_mean']:>6.1f}+-{a['viol_std']:.1f}   "
          f"{a['worst_mean']:>7.4f}+-{a['worst_std']:.4f}  "
          f"{a['stretch_mean']:>7.2f}+-{a['stretch_std']:.2f}", flush=True)


# ════════════════════════════════════════════════════════════════════════════
# WRITE MARKDOWN REPORT
# ════════════════════════════════════════════════════════════════════════════
lines = []
lines.append("# LOSO-CV and lambda_fed Ablation Results")
lines.append("")
lines.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append(f"**Data:** FeTS-2022, {len(volume_scores)} subjects, {K} institutions")
lines.append(f"**Seeds:** {SEEDS}, alpha={alpha}, B={B}")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Table 1: LOSO-CV n0 Selection")
lines.append("")
lines.append("| n0 | Violations (mean+-std) | Worst FNR (mean+-std) | Mean Stretch (mean+-std) | Median Stretch (mean+-std) |")
lines.append("|----|------------------------|------------------------|---------------------------|------------------------------|")
for n0 in n0_list:
    a = agg1[n0]
    marker = " **<-- LOSO-selected**" if n0 == loso_n0 else ""
    lines.append(f"| {n0}{marker} | {a['viol_mean']:.1f}+-{a['viol_std']:.1f} | "
                  f"{a['worst_mean']:.4f}+-{a['worst_std']:.4f} | "
                  f"{a['stretch_mean']:.2f}+-{a['stretch_std']:.2f} | "
                  f"{a['medstretch_mean']:.2f}+-{a['medstretch_std']:.2f} |")
lines.append("")
lines.append(f"**LOSO-selected n0: {loso_n0}** (criteria: {select_reason})")
lines.append("")
lines.append("**Paper's reported n0: 15**")
lines.append("")
match = "Yes" if loso_n0 == 15 else "No"
lines.append(f"**Match: {match}**")
if loso_n0 != 15:
    a15 = agg1[15]
    aloso = agg1[loso_n0]
    lines.append("")
    lines.append(f"  n0=15:  {a15['viol_mean']:.1f}+-{a15['viol_std']:.1f} viol, "
                  f"stretch={a15['stretch_mean']:.2f}+-{a15['stretch_std']:.2f}")
    lines.append(f"  n0={loso_n0}: {aloso['viol_mean']:.1f}+-{aloso['viol_std']:.1f} viol, "
                  f"stretch={aloso['stretch_mean']:.2f}+-{aloso['stretch_std']:.2f}")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Table 2: lambda_fed Conservative Threshold (Theorem 1)")
lines.append("")
lines.append("| Method | Violations | Worst FNR | Mean Stretch |")
lines.append("|--------|-----------|-----------|--------------|")
for n0 in n0_task2:
    a = agg2[f"lambda_fed n0={n0}"]
    lines.append(f"| lambda_fed (n0={n0}) | {a['viol_mean']:.1f}+-{a['viol_std']:.1f} | "
                  f"{a['worst_mean']:.4f}+-{a['worst_std']:.4f} | {a['stretch_mean']:.2f}+-{a['stretch_std']:.2f} |")
for n0 in n0_task2:
    a = agg2[f"per_site n0={n0}"]
    lines.append(f"| Per-site lambda_k (n0={n0}) | {a['viol_mean']:.1f}+-{a['viol_std']:.1f} | "
                  f"{a['worst_mean']:.4f}+-{a['worst_std']:.4f} | {a['stretch_mean']:.2f}+-{a['stretch_std']:.2f} |")
lines.append("")
lines.append("lambda_fed shows the cost of the conservative, single-threshold guarantee: "
              "much worse stretch than per-site deployment for the same n0.")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Table 3: corr_k Ablation")
lines.append("")
lines.append("| n0 | With corr_k: Viol/Worst/Stretch | Without corr_k: Viol/Worst/Stretch |")
lines.append("|----|----------------------------------|--------------------------------------|")
for n0 in n0_task2:
    w = agg2[f"per_site n0={n0}"]
    wo = agg3a[f"n0={n0} nocorr"]
    lines.append(f"| {n0} | {w['viol_mean']:.1f}+-{w['viol_std']:.1f} / "
                  f"{w['worst_mean']:.4f} / {w['stretch_mean']:.2f}+-{w['stretch_std']:.2f} | "
                  f"{wo['viol_mean']:.1f}+-{wo['viol_std']:.1f} / "
                  f"{wo['worst_mean']:.4f} / {wo['stretch_mean']:.2f}+-{wo['stretch_std']:.2f} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Table 4: Grid Resolution Ablation (n0=15)")
lines.append("")
lines.append("| G | Violations | Worst FNR | Stretch |")
lines.append("|---|-----------|-----------|---------|")
for G in G_list:
    a = agg3b[G]
    lines.append(f"| {G} | {a['viol_mean']:.1f}+-{a['viol_std']:.1f} | "
                  f"{a['worst_mean']:.4f}+-{a['worst_std']:.4f} | {a['stretch_mean']:.2f}+-{a['stretch_std']:.2f} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append(f"Output files: {OUT}/")

report = "\n".join(lines)
with open("LOSO_AND_ABLATION_RESULTS.md", "w", encoding="utf-8") as f:
    f.write(report)

# CSV dumps
with open(str(OUT / "task1_loso_n0.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["n0"] + list(agg1[n0_list[0]].keys()))
    w.writeheader()
    for n0 in n0_list:
        w.writerow({"n0": n0, **agg1[n0]})

with open(str(OUT / "task2_lambda_fed.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["method"] + list(next(iter(agg2.values())).keys()))
    w.writeheader()
    for m, a in agg2.items():
        w.writerow({"method": m, **a})

with open(str(OUT / "task3a_corr_ablation.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["method"] + list(next(iter(agg3a.values())).keys()))
    w.writeheader()
    for m, a in agg3a.items():
        w.writerow({"method": m, **a})

with open(str(OUT / "task3b_grid_ablation.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["G"] + list(agg3b[G_list[0]].keys()))
    w.writeheader()
    for G in G_list:
        w.writerow({"G": G, **agg3b[G]})

print("\n" + "=" * 60)
print(report, flush=True)
print("=" * 60)
print(f"\nSaved LOSO_AND_ABLATION_RESULTS.md and CSVs in {OUT}/", flush=True)
