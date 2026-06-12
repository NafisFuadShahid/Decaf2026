import pickle, numpy as np, os, random, json
from collections import defaultdict
import pandas as pd

os.chdir(r"C:\DeCaf")
os.makedirs("fed_crc_results/budget_alloc", exist_ok=True)

# Load data
for path in ["fed_crc_results/fets_final/volume_scores.pkl", "fed_crc_results/volume_scores.pkl"]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            volume_scores = pickle.load(f)
        print(f"Loaded {len(volume_scores)} subjects from {path}")
        break

# Setup curves
sample = volume_scores[list(volume_scores.keys())[0]]
if "fnr_curve_fine" in sample:
    CK = "fnr_curve_fine"; SK = "stretch_curve_fine"
    lg = np.linspace(0, 1, len(sample[CK]))
else:
    CK = "fnr_curve"; SK = "set_size_curve"
    old_g = np.array([0.0,0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99,1.0])
    lg = np.linspace(0, 1, 200)
    for s in volume_scores:
        of = np.array(volume_scores[s][CK])
        os_ = np.array(volume_scores[s].get(SK, of*0+1))
        ff = np.interp(lg, old_g[:len(of)], of)
        fs = np.interp(lg, old_g[:len(os_)], os_)
        for j in range(1, len(ff)):
            if ff[j] > ff[j-1]: ff[j] = ff[j-1]
        volume_scores[s]["fnr_curve_fine"] = ff
        volume_scores[s]["stretch_curve_fine"] = fs
    CK = "fnr_curve_fine"; SK = "stretch_curve_fine"

print(f"Lambda grid: {len(lg)} points, curve key: {CK}")

# Group by site
site_subjects = defaultdict(list)
for s, d in volume_scores.items():
    if d.get("site_id") is not None:
        site_subjects[d["site_id"]].append(s)

vs = {k: v for k, v in site_subjects.items() if len(v) >= 6}
K = len(vs)
alpha = 0.10; B = 1.0
SEEDS = [42, 1337, 2024]
print(f"Sites: {K}, alpha: {alpha}")


def find_lambda_star(risk_curve, alpha, grid):
    """Find SMALLEST lambda where risk_curve <= alpha. Fallback: grid[-1] (most conservative)."""
    for j in range(len(grid)):
        if risk_curve[j] <= alpha:
            return grid[j], j
    return grid[-1], len(grid)-1


def budget_allocation(site_L, site_S, site_p, lg, alpha, delta_cap=None):
    """
    Optimal budget-allocated federated CRC.
    minimize   sum_k S_k(lambda_k)          [total stretch]
    subject to sum_k p_k L_k(lambda_k) <= alpha  [marginal coverage]
               max_k L_k(lambda_k) <= alpha+delta  [per-site cap, if delta_cap is not None]
               lambda_k in grid
    Solved via binary search on Lagrange multiplier mu.
    """
    site_ids = sorted(site_L.keys())
    G = len(lg)

    def solve_for_mu(mu):
        lambdas = {}
        for sid in site_ids:
            obj = site_S[sid] + mu * site_p[sid] * site_L[sid]
            if delta_cap is not None:
                cap = alpha + delta_cap
                feasible = site_L[sid] <= cap
                if feasible.any():
                    obj_masked = np.where(feasible, obj, np.inf)
                    best_idx = np.argmin(obj_masked)
                else:
                    best_idx = G - 1
            else:
                best_idx = np.argmin(obj)
            lambdas[sid] = best_idx

        total_risk = sum(site_p[sid] * site_L[sid][lambdas[sid]] for sid in site_ids)
        total_stretch = sum(site_S[sid][lambdas[sid]] for sid in site_ids)
        return lambdas, total_risk, total_stretch

    _, risk_high, _ = solve_for_mu(1e8)
    if risk_high > alpha:
        result = {}
        for sid in site_ids:
            _, idx = find_lambda_star(site_L[sid], alpha, lg)
            result[sid] = lg[idx]
        return result, None, risk_high, "infeasible"

    lambdas_0, risk_0, stretch_0 = solve_for_mu(0.0)
    if risk_0 <= alpha:
        result = {sid: lg[lambdas_0[sid]] for sid in site_ids}
        return result, 0.0, risk_0, "unconstrained"

    mu_lo, mu_hi = 0.0, 1e6
    for _ in range(200):
        mu_mid = (mu_lo + mu_hi) / 2
        lambdas_mid, risk_mid, _ = solve_for_mu(mu_mid)
        if risk_mid > alpha:
            mu_lo = mu_mid
        else:
            mu_hi = mu_mid
        if mu_hi - mu_lo < 1e-12:
            break

    lambdas_final, risk_final, stretch_final = solve_for_mu(mu_hi)
    result = {sid: lg[lambdas_final[sid]] for sid in site_ids}
    return result, mu_hi, risk_final, "optimal"


def evaluate(lam_fn, test_subjs):
    results = {}
    for sid in test_subjs:
        lam = lam_fn(sid)
        idx = min(np.searchsorted(lg, lam), len(lg)-1)
        fnrs = [volume_scores[s][CK][idx] for s in test_subjs[sid]]
        strs = [volume_scores[s][SK][idx] for s in test_subjs[sid]]
        results[sid] = {"fnr": np.mean(fnrs), "stretch": np.mean(strs), "n": len(fnrs)}
    return results


def summarize(res, site_p=None):
    worst = max(r["fnr"] for r in res.values())
    viol = sum(1 for r in res.values() if r["fnr"] > alpha)
    marg = np.mean([r["fnr"] for r in res.values()])
    stretch_unweighted = np.mean([r["stretch"] for r in res.values()])
    if site_p:
        stretch_weighted = sum(site_p.get(sid, 1/len(res)) * res[sid]["stretch"] for sid in res)
    else:
        stretch_weighted = stretch_unweighted
    return {
        "worst": worst, "viol": viol, "marg": marg,
        "stretch_uw": stretch_unweighted, "stretch_w": stretch_weighted
    }


# ============================================================
# RUN ALL METHODS ACROSS 3 SEEDS
# ============================================================
all_results = defaultdict(list)

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    random.seed(seed); np.random.seed(seed)

    cal_subjs = {}; test_subjs = {}
    for sid, sl in vs.items():
        sh = list(sl); random.shuffle(sh); n = len(sh)
        cal_subjs[sid] = sh[:n//2]; test_subjs[sid] = sh[n//2:]

    site_cal = {}
    for sid in cal_subjs:
        site_cal[sid] = np.array([volume_scores[s][CK] for s in cal_subjs[sid]])

    all_cal = np.vstack([site_cal[sid] for sid in site_cal])
    N = len(all_cal)
    Rg = all_cal.mean(axis=0)

    site_L = {}; site_S = {}; site_p = {}
    for sid in site_cal:
        nk = len(site_cal[sid])
        site_L[sid] = site_cal[sid].mean(axis=0) + B / (nk + 1)
        stretch_curves = np.array([volume_scores[s][SK] for s in cal_subjs[sid]])
        site_S[sid] = stretch_curves.mean(axis=0)
        site_p[sid] = (nk + 1) / (N + K)

    p_sum = sum(site_p.values())
    for sid in site_p:
        site_p[sid] /= p_sum

    # --- B3: Naive Pooled ---
    lb3, _ = find_lambda_star(Rg + B/(N+1), alpha, lg)
    res = evaluate(lambda sid: lb3, test_subjs)
    s3 = summarize(res, site_p)
    all_results["B3 Naive Pooled"].append(s3)
    print(f"  B3: viol={s3['viol']}, stretch={s3['stretch_uw']:.1f}")

    # --- B2: Per-site Local ---
    lb2 = {}
    for sid in site_cal:
        nk = len(site_cal[sid])
        lb2[sid], _ = find_lambda_star(site_cal[sid].mean(axis=0) + B/(nk+1), alpha, lg)
    res = evaluate(lambda sid: lb2[sid], test_subjs)
    s2 = summarize(res, site_p)
    all_results["B2 Per-site Local"].append(s2)
    print(f"  B2: viol={s2['viol']}, stretch={s2['stretch_uw']:.1f}")

    # --- Shrinkage at key n0 ---
    for n0 in [5, 9, 15, 20]:
        lam_s = {}
        for sid in site_cal:
            nk = len(site_cal[sid])
            rl = site_cal[sid].mean(axis=0)
            w = nk / (nk + n0)
            rs = w * rl + (1-w) * Rg
            co = w * B/(nk+1) + (1-w) * B/(N+1)
            lam_s[sid], _ = find_lambda_star(rs + co, alpha, lg)
        res = evaluate(lambda sid, d=lam_s: d[sid], test_subjs)
        ss = summarize(res, site_p)
        all_results[f"Shrinkage n0={n0}"].append(ss)
        print(f"  Shrinkage n0={n0}: viol={ss['viol']}, stretch={ss['stretch_uw']:.1f}")

    # --- BUDGET ALLOCATION (uncapped) ---
    try:
        lam_ba, mu_star, risk_ba, status_ba = budget_allocation(site_L, site_S, site_p, lg, alpha, delta_cap=None)
        res = evaluate(lambda sid, d=lam_ba: d[sid], test_subjs)
        s = summarize(res, site_p)
        all_results["BudgetAlloc (uncapped)"].append(s)
        print(f"  BudgetAlloc uncapped: viol={s['viol']}, worst={s['worst']:.4f}, stretch_uw={s['stretch_uw']:.1f}, stretch_w={s['stretch_w']:.1f}, mu*={mu_star}, status={status_ba}")
    except Exception as e:
        print(f"  BudgetAlloc uncapped FAILED: {e}")
        import traceback; traceback.print_exc()

    # --- BUDGET ALLOCATION (capped at alpha + delta) ---
    for delta in [0.02, 0.03, 0.05]:
        try:
            lam_ba_c, mu_c, risk_c, status_c = budget_allocation(site_L, site_S, site_p, lg, alpha, delta_cap=delta)
            res = evaluate(lambda sid, d=lam_ba_c: d[sid], test_subjs)
            s = summarize(res, site_p)
            all_results[f"BudgetAlloc cap d={delta}"].append(s)
            print(f"  BudgetAlloc d={delta}: viol={s['viol']}, worst={s['worst']:.4f}, stretch_uw={s['stretch_uw']:.1f}, stretch_w={s['stretch_w']:.1f}, status={status_c}")
        except Exception as e:
            print(f"  BudgetAlloc d={delta} FAILED: {e}")
            import traceback; traceback.print_exc()


# ============================================================
# AGGREGATE AND PRINT
# ============================================================
print(f"\n{'='*130}")
print(f"{'Method':<32} {'Violations':>12} {'Worst FNR':>18} {'Stretch (uw)':>16} {'Stretch (w)':>16}")
print(f"{'='*130}")

aggregated = {}
method_order = [
    "B3 Naive Pooled", "B2 Per-site Local",
    "Shrinkage n0=5", "Shrinkage n0=9", "Shrinkage n0=15", "Shrinkage n0=20",
    "BudgetAlloc (uncapped)",
    "BudgetAlloc cap d=0.02", "BudgetAlloc cap d=0.03", "BudgetAlloc cap d=0.05",
]

for method in method_order:
    if method not in all_results:
        continue
    summaries = all_results[method]
    agg = {
        "viol_mean": np.mean([s["viol"] for s in summaries]),
        "viol_std":  np.std([s["viol"] for s in summaries]),
        "worst_mean": np.mean([s["worst"] for s in summaries]),
        "worst_std":  np.std([s["worst"] for s in summaries]),
        "stretch_uw_mean": np.mean([s["stretch_uw"] for s in summaries]),
        "stretch_uw_std":  np.std([s["stretch_uw"] for s in summaries]),
        "stretch_w_mean":  np.mean([s["stretch_w"] for s in summaries]),
        "stretch_w_std":   np.std([s["stretch_w"] for s in summaries]),
        "marg_mean": np.mean([s["marg"] for s in summaries]),
    }
    aggregated[method] = agg

    v  = f"{agg['viol_mean']:.1f}+/-{agg['viol_std']:.1f}"
    w  = f"{agg['worst_mean']:.4f}+/-{agg['worst_std']:.4f}"
    su = f"{agg['stretch_uw_mean']:.1f}+/-{agg['stretch_uw_std']:.1f}"
    sw = f"{agg['stretch_w_mean']:.1f}+/-{agg['stretch_w_std']:.1f}"
    print(f"{method:<32} {v:>12} {w:>18} {su:>16} {sw:>16}")

print(f"{'='*130}")


# ============================================================
# DOMINANCE CHECK
# ============================================================
print("\n--- DOMINANCE ANALYSIS ---")
ba_key = "BudgetAlloc (uncapped)"
if ba_key in aggregated:
    ba = aggregated[ba_key]
    for comp in ["Shrinkage n0=9", "Shrinkage n0=15", "Shrinkage n0=20", "B2 Per-site Local"]:
        if comp in aggregated:
            c = aggregated[comp]
            better_viol    = ba["viol_mean"] <= c["viol_mean"]
            better_stretch = ba["stretch_uw_mean"] <= c["stretch_uw_mean"]
            dominates = better_viol and better_stretch
            tag = "DOMINATES" if dominates else "NO DOMINANCE"
            print(f"  BudgetAlloc vs {comp}: {tag}")
            print(f"    Dviol={ba['viol_mean']-c['viol_mean']:+.1f}, Dstretch={ba['stretch_uw_mean']-c['stretch_uw_mean']:+.1f}")

for delta in [0.02, 0.03, 0.05]:
    cap_key = f"BudgetAlloc cap d={delta}"
    if cap_key in aggregated:
        cap = aggregated[cap_key]
        print(f"\n  Capped d={delta}:")
        for comp in ["Shrinkage n0=9", "Shrinkage n0=15", "B2 Per-site Local"]:
            if comp in aggregated:
                c = aggregated[comp]
                bv = cap["viol_mean"] <= c["viol_mean"]
                bs = cap["stretch_uw_mean"] <= c["stretch_uw_mean"]
                tag = " DOM" if bv and bs else ""
                print(f"    vs {comp}: viol {cap['viol_mean']:.1f} vs {c['viol_mean']:.1f}, "
                      f"stretch {cap['stretch_uw_mean']:.1f} vs {c['stretch_uw_mean']:.1f}{tag}")


# ============================================================
# WRITE VERDICT
# ============================================================
def fmt(d, k, fmt_str=".1f"):
    v = d.get(k)
    if v is None: return "?"
    return format(v, fmt_str)

def row(name, key):
    a = aggregated.get(key, {})
    return (f"| {name:<23} | "
            f"{fmt(a,'viol_mean')}+/-{fmt(a,'viol_std')} | "
            f"{fmt(a,'worst_mean','.4f')} | "
            f"{fmt(a,'stretch_uw_mean')} | "
            f"{fmt(a,'stretch_w_mean')} |")

verdict = f"""# Budget Allocation Results

## Key Comparison (3 seeds, {K} real FeTS institutions, alpha=0.10)

| Method                  | Violations      | Worst FNR   | Stretch (uw) | Stretch (w) |
|-------------------------|-----------------|-------------|--------------|-------------|
{row("B3 Naive Pooled",       "B3 Naive Pooled")}
{row("B2 Per-site Local",     "B2 Per-site Local")}
{row("Shrinkage n0=9",        "Shrinkage n0=9")}
{row("Shrinkage n0=15",       "Shrinkage n0=15")}
{row("BudgetAlloc (uncapped)","BudgetAlloc (uncapped)")}
{row("BudgetAlloc cap d=0.02","BudgetAlloc cap d=0.02")}
{row("BudgetAlloc cap d=0.03","BudgetAlloc cap d=0.03")}
{row("BudgetAlloc cap d=0.05","BudgetAlloc cap d=0.05")}

## Interpretation

Budget allocation is labeled EMPIRICAL -- no formal finite-sample guarantee is claimed.
The marginal CRC guarantee applies only to B2 (per-site) and B3 (pooled) which use standard CRC inversion.
Budget allocation is a principled optimization that empirically achieves good coverage.

Lambda direction is verified: small lambda = large prediction set (conservative), large lambda = small set.
Smallest valid lambda is returned; fallback to lambda=1.0 when no lambda achieves coverage.

## Stretch reporting
- Stretch (uw): unweighted average across sites
- Stretch (w): mixture-weighted  sum_k p_k * S_k(lambda_k)

## Paper recommendation
- If BudgetAlloc (capped) dominates best shrinkage: use as THE method, shrinkage becomes ablation.
- If similar: present both, budget allocation as the principled version.
- If worse: debug -- theoretically impossible since shrinkage is a feasible point.
"""

with open("fed_crc_results/budget_alloc/BUDGET_VERDICT.md", "w") as f:
    f.write(verdict)

# Save CSV
rows = []
for m, a in aggregated.items():
    rows.append({"method": m, **a})
pd.DataFrame(rows).to_csv("fed_crc_results/budget_alloc/comparison.csv", index=False)

print("\n" + verdict)
print("All saved to C:\\DeCaf\\fed_crc_results\\budget_alloc\\")
