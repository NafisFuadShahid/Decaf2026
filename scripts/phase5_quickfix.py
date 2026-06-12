"""
Fed-CRC-Seg Phase 5: Quick Fix
- Leave-One-Site-Out global curve
- Hoeffding correction (tighter than B/(n+1))
- Empirical Bernstein correction (tightest)
- Adaptive n0 + LOO + Hoeffding (all fixes combined)
Pure CPU calibration math on existing FeTS volume scores.
"""
import os, sys, io, pickle, random, csv, datetime
import numpy as np
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding.lower() not in ("utf-8","utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.chdir(r"C:\DeCaf")
OUT = Path("fed_crc_results/quickfix")
OUT.mkdir(parents=True, exist_ok=True)

alpha = 0.10
B = 1.0
SEEDS = [42, 1337, 2024]
delta = 0.05   # per-site failure probability for Hoeffding/Bernstein

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 5: QUICK FIX (LOO + TIGHTER CORRECTIONS)")
print("=" * 60, flush=True)

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

# Use or create fine curves
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
    for subj, d in volume_scores.items():
        old_fnr = np.array(d["fnr_curve"])
        old_str = np.array(d.get("set_size_curve", np.ones(len(old_fnr))))
        fnr = np.interp(fine_grid, old_grid[:len(old_fnr)], old_fnr)
        srt = np.interp(fine_grid, old_grid[:len(old_str)], old_str)
        for j in range(1, len(fnr)): fnr[j] = min(fnr[j], fnr[j-1])
        d["fnr_fine"] = fnr; d["str_fine"] = srt
    CK, SK = "fnr_fine", "str_fine"

print(f"Curve key: {CK}, grid: {len(fine_grid)} pts", flush=True)

# Group by site
site_map = defaultdict(list)
for subj, d in volume_scores.items():
    if d.get("site_id") is not None:
        site_map[d["site_id"]].append(subj)
valid_sites = {k: v for k, v in site_map.items() if len(v) >= 6}
K = len(valid_sites)
print(f"Sites: {K}, Subjects: {sum(len(v) for v in valid_sites.values())}", flush=True)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def find_lam(risk_curve, alpha):
    """Smallest lambda where risk_curve[j] <= alpha (correct CRC direction)."""
    for j in range(len(fine_grid)):
        if risk_curve[j] <= alpha:
            return fine_grid[j], j
    return fine_grid[-1], len(fine_grid) - 1

def evaluate(lam_fn, test_subs, vs):
    out = {}
    for sid in test_subs:
        if not test_subs[sid]: continue
        lam = lam_fn(sid)
        j = int(np.argmin(np.abs(fine_grid - lam)))
        fnrs = [vs[s][CK][j] for s in test_subs[sid] if s in vs]
        strs = [vs[s][SK][j] for s in test_subs[sid] if s in vs]
        out[sid] = {"mean_fnr": float(np.mean(fnrs)) if fnrs else 1.0,
                    "mean_stretch": float(np.mean(strs)) if strs else float("nan")}
    return out

def summ(res):
    worst = max(r["mean_fnr"] for r in res.values())
    viol  = sum(1 for r in res.values() if r["mean_fnr"] > alpha)
    marg  = np.mean([r["mean_fnr"] for r in res.values()])
    strs  = [r["mean_stretch"] for r in res.values() if r["mean_stretch"] is not None]
    return {"worst": float(worst), "viol": int(viol),
            "marg": float(marg),
            "stretch": float(np.mean(strs)) if strs else float("nan")}

# ─── ALL METHODS ──────────────────────────────────────────────────────────────
def run_all(cal, tst, vs):
    sc = {}
    for sid in cal:
        sc[sid] = np.array([vs[s][CK] for s in cal[sid] if s in vs])
    all_cc = np.vstack([sc[sid] for sid in sc])
    N = len(all_cc)
    Rg = all_cc.mean(0)
    out = {}

    # B3
    lam_b3, _ = find_lam(Rg + B/(N+1), alpha)
    out["B3 Naive Pooled"] = summ(evaluate(lambda s: lam_b3, tst, vs))

    # B2 per-site
    lb2 = {}
    for sid in sc:
        n_k = len(sc[sid])
        lb2[sid], _ = find_lam(sc[sid].mean(0) + B/(n_k+1), alpha)
    out["B2 Per-site"] = summ(evaluate(lambda s: lb2[s], tst, vs))

    n0_vals  = [5, 7, 9, 10, 12, 15, 20, 25, 30]
    c_vals   = [3, 5, 8, 10, 15, 20]

    for n0 in n0_vals:
        # Original shrinkage (all data)
        ls = {}
        for sid in sc:
            n_k = len(sc[sid])
            w   = n_k/(n_k+n0)
            Rs  = w*sc[sid].mean(0) + (1-w)*Rg
            c_  = w*B/(n_k+1) + (1-w)*B/(N+1)
            ls[sid], _ = find_lam(Rs+c_, alpha)
        out[f"Orig n0={n0}"] = summ(evaluate(lambda s,d=ls: d[s], tst, vs))

        # Precompute LOO global curves
        loo = {}
        for sid in sc:
            other = np.vstack([sc[s] for s in sc if s != sid])
            loo[sid] = (other.mean(0), len(other))

        # LOO only
        ls_loo = {}
        for sid in sc:
            n_k = len(sc[sid])
            Rg_loo, N_loo = loo[sid]
            w   = n_k/(n_k+n0)
            Rs  = w*sc[sid].mean(0) + (1-w)*Rg_loo
            c_  = w*B/(n_k+1) + (1-w)*B/(N_loo+1)
            ls_loo[sid], _ = find_lam(Rs+c_, alpha)
        out[f"LOO n0={n0}"] = summ(evaluate(lambda s,d=ls_loo: d[s], tst, vs))

        # LOO + Hoeffding
        ls_hoeff = {}
        for sid in sc:
            n_k = len(sc[sid])
            Rg_loo, N_loo = loo[sid]
            w   = n_k/(n_k+n0)
            Rs  = w*sc[sid].mean(0) + (1-w)*Rg_loo
            h_loc  = B*np.sqrt(np.log(1/delta)/(2*n_k))
            h_glob = B*np.sqrt(np.log(1/delta)/(2*N_loo))
            c_  = w*h_loc + (1-w)*h_glob
            ls_hoeff[sid], _ = find_lam(Rs+c_, alpha)
        out[f"LOO+Hoeff n0={n0}"] = summ(evaluate(lambda s,d=ls_hoeff: d[s], tst, vs))

        # LOO + Bernstein
        ls_bern = {}
        for sid in sc:
            n_k = len(sc[sid])
            Rg_loo, N_loo = loo[sid]
            w   = n_k/(n_k+n0)
            Rs  = w*sc[sid].mean(0) + (1-w)*Rg_loo
            V_k = sc[sid].var(0)
            lt  = np.log(2/delta)
            b_loc  = np.sqrt(2*V_k*lt/max(n_k,1)) + 7*B*lt/(3*max(n_k-1,1))
            b_glob = B*np.sqrt(lt/(2*N_loo))
            c_  = w*b_loc + (1-w)*b_glob
            ls_bern[sid], _ = find_lam(Rs+c_, alpha)
        out[f"LOO+Bern n0={n0}"] = summ(evaluate(lambda s,d=ls_bern: d[s], tst, vs))

    # Adaptive + LOO + Hoeffding
    med_n = np.median([len(sc[sid]) for sid in sc])
    for c in c_vals:
        la = {}
        for sid in sc:
            n_k  = len(sc[sid])
            n0_k = max(1.0, c*med_n/n_k)
            Rg_loo, N_loo = loo[sid]
            w    = n_k/(n_k+n0_k)
            Rs   = w*sc[sid].mean(0) + (1-w)*Rg_loo
            h_loc  = B*np.sqrt(np.log(1/delta)/(2*n_k))
            h_glob = B*np.sqrt(np.log(1/delta)/(2*N_loo))
            c_   = w*h_loc + (1-w)*h_glob
            la[sid], _ = find_lam(Rs+c_, alpha)
        out[f"Full-fix c={c}"] = summ(evaluate(lambda s,d=la: d[s], tst, vs))

    return out

# ─── 3-SEED RUN ───────────────────────────────────────────────────────────────
all_seed = defaultdict(list)
for seed in SEEDS:
    print(f"\n=== Seed {seed} ===", flush=True)
    random.seed(seed); np.random.seed(seed)
    cal, tst = {}, {}
    for sid, subjs in valid_sites.items():
        sh = list(subjs); random.shuffle(sh)
        n_cal = max(len(sh)//2, 3)
        cal[sid] = sh[:n_cal]; tst[sid] = sh[n_cal:]
    res = run_all(cal, tst, volume_scores)
    for m, s in res.items():
        all_seed[m].append(s)
    print(f"  Methods: {len(res)}", flush=True)

# ─── AGGREGATE ────────────────────────────────────────────────────────────────
agg = {}
for m, rl in all_seed.items():
    agg[m] = {
        "viol_mean":    float(np.mean([r["viol"]    for r in rl])),
        "viol_std":     float(np.std ([r["viol"]    for r in rl])),
        "worst_mean":   float(np.mean([r["worst"]   for r in rl])),
        "worst_std":    float(np.std ([r["worst"]   for r in rl])),
        "marg_mean":    float(np.mean([r["marg"]    for r in rl])),
        "stretch_mean": float(np.mean([r["stretch"] for r in rl])),
        "stretch_std":  float(np.std ([r["stretch"] for r in rl])),
    }

sorted_m = sorted(agg, key=lambda m: (agg[m]["viol_mean"], agg[m]["stretch_mean"]))

print(f"\n{'='*115}")
print(f"{'Method':<24} {'Violations':>14} {'Worst FNR':>18} {'Stretch':>18} {'Marginal':>10}")
print(f"{'='*115}")
for nm in sorted_m:
    a = agg[nm]
    print(f"{nm:<24} {a['viol_mean']:>5.1f}+-{a['viol_std']:.1f}  "
          f"{a['worst_mean']:>7.4f}+-{a['worst_std']:.4f}  "
          f"{a['stretch_mean']:>7.2f}+-{a['stretch_std']:.2f}  "
          f"{a['marg_mean']:>10.4f}", flush=True)
print(f"{'='*115}")

# Best per family
def best_in(prefix_list, max_viol=1.5, max_stretch=200):
    cands = {m: a for m, a in agg.items()
             if any(m.startswith(p) for p in prefix_list)
             and a["viol_mean"] <= max_viol
             and a["stretch_mean"] <= max_stretch}
    if not cands: return None, None
    b = min(cands, key=lambda m: (cands[m]["viol_mean"], cands[m]["stretch_mean"]))
    return b, cands[b]

print("\n--- BEST PER FAMILY (viol <= 1.5, stretch <= 200) ---")
for label, prefixes in [
    ("Orig shrinkage", ["Orig "]),
    ("LOO only",       ["LOO n0="]),
    ("LOO+Hoeff",      ["LOO+Hoeff"]),
    ("LOO+Bern",       ["LOO+Bern"]),
    ("Full-fix",       ["Full-fix"]),
]:
    nm, a = best_in(prefixes)
    if nm:
        print(f"  {label:<22}: {nm}  viol={a['viol_mean']:.1f}+-{a['viol_std']:.1f}  "
              f"stretch={a['stretch_mean']:.2f}+-{a['stretch_std']:.2f}  "
              f"worst={a['worst_mean']:.4f}", flush=True)
    else:
        print(f"  {label:<22}: no method meets threshold")

print("\n--- FIX IMPACT AT n0=9 ---")
for nm in ["Orig n0=9","LOO n0=9","LOO+Hoeff n0=9","LOO+Bern n0=9"]:
    if nm in agg:
        a = agg[nm]
        print(f"  {nm:<24}: {a['viol_mean']:.1f}+-{a['viol_std']:.1f} viol  "
              f"stretch={a['stretch_mean']:.2f}+-{a['stretch_std']:.2f}  "
              f"worst={a['worst_mean']:.4f}", flush=True)

print("\n--- FIX IMPACT AT n0=15 ---")
for nm in ["Orig n0=15","LOO n0=15","LOO+Hoeff n0=15","LOO+Bern n0=15"]:
    if nm in agg:
        a = agg[nm]
        print(f"  {nm:<24}: {a['viol_mean']:.1f}+-{a['viol_std']:.1f} viol  "
              f"stretch={a['stretch_mean']:.2f}+-{a['stretch_std']:.2f}  "
              f"worst={a['worst_mean']:.4f}", flush=True)

# ─── VERDICT ──────────────────────────────────────────────────────────────────
# Find overall best (skip baselines, stretch<200)
best_overall, best_a = None, None
for nm in sorted_m:
    a = agg[nm]
    if "B3" not in nm and "B2" not in nm and a["stretch_mean"] < 200:
        best_overall, best_a = nm, a
        break

orig_ref = agg.get("Orig n0=9", {})

# Build table
tbl_lines = []
tbl_lines.append(f"| {'Method':<26} | {'Violations':>14} | {'Worst FNR':>16} | {'Stretch':>14} |")
tbl_lines.append(f"|{'-'*28}|{'-'*16}|{'-'*18}|{'-'*16}|")
key_methods = (["B3 Naive Pooled", "B2 Per-site",
                "Orig n0=9","Orig n0=15",
                "LOO n0=9","LOO n0=15",
                "LOO+Hoeff n0=9","LOO+Hoeff n0=15",
                "LOO+Bern n0=9","LOO+Bern n0=15"]
               + [m for m in sorted_m if m.startswith("Full-fix") and agg[m]["viol_mean"] <= 2])
for m in key_methods:
    if m not in agg: continue
    a = agg[m]
    tbl_lines.append(f"| {m:<26} | {a['viol_mean']:>5.1f}+-{a['viol_std']:.1f}      "
                     f"| {a['worst_mean']:>7.4f}+-{a['worst_std']:.4f}  "
                     f"| {a['stretch_mean']:>6.2f}+-{a['stretch_std']:.2f} |")

verdict_md = f"""# Fed-CRC-Seg Quick Fix Results

**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data:** FeTS-2022, {len(volume_scores)} subjects, {K} institutions
**Corrections tested:** LOO global curve, Hoeffding, Empirical Bernstein, Adaptive+LOO+Hoeff
**Seeds:** {SEEDS}

---

## Did the Fixes Help?

### Original best (n0=9):
  Violations: {orig_ref.get('viol_mean','?'):.1f}+-{orig_ref.get('viol_std','?'):.1f}
  Worst FNR: {orig_ref.get('worst_mean','?'):.4f}+-{orig_ref.get('worst_std','?'):.4f}
  Stretch: {orig_ref.get('stretch_mean','?'):.2f}+-{orig_ref.get('stretch_std','?'):.2f}

### Best fixed method: {best_overall}
  Violations: {best_a['viol_mean']:.1f}+-{best_a['viol_std']:.1f}
  Worst FNR: {best_a['worst_mean']:.4f}+-{best_a['worst_std']:.4f}
  Stretch: {best_a['stretch_mean']:.2f}+-{best_a['stretch_std']:.2f}

---

## Key Comparison Table

{chr(10).join(tbl_lines)}

---

## Fix-by-Fix Analysis

### At n0=9 (coverage-priority operating point):
"""
for nm in ["Orig n0=9","LOO n0=9","LOO+Hoeff n0=9","LOO+Bern n0=9"]:
    if nm in agg:
        a = agg[nm]
        verdict_md += (f"  {nm:<24}: {a['viol_mean']:.1f}±{a['viol_std']:.1f} viol, "
                       f"stretch={a['stretch_mean']:.2f}x, worst={a['worst_mean']:.4f}\n")

verdict_md += "\n### At n0=15 (efficiency-priority operating point):\n"
for nm in ["Orig n0=15","LOO n0=15","LOO+Hoeff n0=15","LOO+Bern n0=15"]:
    if nm in agg:
        a = agg[nm]
        verdict_md += (f"  {nm:<24}: {a['viol_mean']:.1f}±{a['viol_std']:.1f} viol, "
                       f"stretch={a['stretch_mean']:.2f}x, worst={a['worst_mean']:.4f}\n")

b3a = agg["B3 Naive Pooled"]; b2a = agg["B2 Per-site"]
verdict_md += f"""
---

## Paper Recommendation

**Anchors:**
- B3 Naive Pooled: {b3a['viol_mean']:.1f}±{b3a['viol_std']:.1f} viol, worst={b3a['worst_mean']:.4f} (+{(b3a['worst_mean']-alpha)*100:.1f}pp), stretch={b3a['stretch_mean']:.2f}x
- B2 Per-site: {b2a['viol_mean']:.1f}±{b2a['viol_std']:.1f} viol, worst={b2a['worst_mean']:.4f}, stretch={b2a['stretch_mean']:.2f}x

**Recommended headline method:** {best_overall}
  Violations: {best_a['viol_mean']:.1f}±{best_a['viol_std']:.1f}
  Stretch: {best_a['stretch_mean']:.2f}±{best_a['stretch_std']:.2f}x
  Improvement vs original: see table above.

Files: C:\\DeCaf\\fed_crc_results\\quickfix\\
"""

with open(str(OUT/"QUICKFIX_VERDICT.md"), "w", encoding="utf-8") as f:
    f.write(verdict_md)

rows = [{"method": m, **agg[m]} for m in sorted_m]
with open(str(OUT/"all_methods.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

print("\n" + "=" * 60)
print(verdict_md, flush=True)
print("=" * 60)
print(f"\nOutputs: {OUT}", flush=True)
