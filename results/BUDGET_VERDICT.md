# Budget Allocation Results

## Key Comparison (3 seeds, 20 real FeTS institutions, alpha=0.10)

| Method                  | Violations (mean+/-std) | Worst FNR   | Stretch (uw) | Stretch (w) |
|-------------------------|-------------------------|-------------|--------------|-------------|
| B3 Naive Pooled         | 8.0+/-2.4  | 0.1779 | 1.5  | 1.6  |
| B2 Per-site Local       | 1.3+/-1.2  | 0.1110 | 83.2 | 20.7 |
| Shrinkage n0=5          | 1.3+/-1.2  | 0.1086 | 63.5 | 13.6 |
| Shrinkage n0=9          | 1.3+/-1.2  | 0.1118 | 28.8 | 7.6  |
| Shrinkage n0=15         | 2.7+/-1.7  | 0.1185 | 4.4  | 3.1  |
| Shrinkage n0=20         | 3.3+/-2.6  | 0.1248 | 2.0  | 2.1  |
| BudgetAlloc (uncapped)  | 12.3+/-0.9 | 0.3508 | 1.4  | 2.7  |
| BudgetAlloc cap d=0.02  | 2.7+/-1.2  | 0.1323 | 74.7 | 16.7 |
| BudgetAlloc cap d=0.03  | 2.7+/-1.2  | 0.1422 | 71.6 | 15.5 |
| BudgetAlloc cap d=0.05  | 4.0+/-0.0  | 0.1517 | 63.8 | 13.2 |

Violations = number of sites (out of 20) with test FNR > alpha=0.10.
Stretch (uw) = unweighted average across sites.
Stretch (w) = mixture-weighted sum_k p_k * S_k(lambda_k).

## Interpretation

Budget allocation is labeled EMPIRICAL -- no formal finite-sample guarantee is claimed.
The marginal CRC guarantee applies only to B2 (per-site) and B3 (pooled) which use standard CRC inversion.

Lambda direction: small lambda = large prediction set (conservative), large lambda = small set.

## Key Findings

**BudgetAlloc (uncapped) FAILS badly:**
- 12.3/20 sites violated (vs 1.3 for per-site, 8.0 for pooled)
- Worst FNR = 0.35 (vs alpha=0.10)
- Achieves minimal stretch (1.4) by sacrificing per-site coverage on hard sites
- The mixture constraint (sum_k p_k L_k <= alpha) allows concentration of risk on
  small/hard sites while satisfying the aggregate. This is not a bug -- it is what
  the Lagrangian optimizer does when unconstrained.

**BudgetAlloc capped (d=0.02/0.03):**
- 2.7/20 violations -- same as shrinkage n0=9, but more stable (lower std)
- Stretch 71-75 -- WORSE than shrinkage n0=9 (28.8) and much worse than n0=15 (4.4)
- Closest competitor: per-site local (1.3 viol, 83.2 stretch) -- cap version has more
  violations AND less stretch, an ambiguous trade-off

**No budget allocation variant dominates any shrinkage variant** on both dimensions.

## Root cause: shrinkage is NOT a feasible point for budget allocation

The claim "budget allocation is theoretically optimal because shrinkage is feasible" is WRONG.
Shrinkage uses a blended risk estimator: R_shrink = w*R_local + (1-w)*R_global + correction.
Budget allocation uses: L_k = mean_k + B/(nk+1).
These are different risk estimators. Shrinkage's lambda may or may not satisfy
sum_k p_k L_k(lambda_k) <= alpha -- there is no guarantee. So budget allocation can
legitimately be worse than shrinkage in practice.

Furthermore, the Lagrangian approach to a discrete non-convex problem (lambda on a grid,
piecewise-constant objectives) can have duality gaps, causing the binary search to converge
to a suboptimal solution.

## Paper recommendation (UPDATED)

Budget allocation does NOT replace shrinkage -- it performs worse on both violations and stretch.

Recommended framing for DeCaF 2026:
1. Shrinkage (n0=9) remains the headline method: 1.3 violations, stretch=28.8
2. Budget allocation is presented as a "natural but failing extension" -- shows that
   naive mixture-coverage optimization without proper regularization hurts badly.
3. The uncapped result (12.3 violations) is a useful negative to include: demonstrates
   that the optimization pressure to minimize stretch drives lambda too high on hard sites
   unless explicitly protected by a per-site cap.
4. Capped variant (d=0.02) is included for completeness as the "principled but suboptimal"
   version -- it shows that even with the fix, implicit regularization from shrinkage wins.

This actually STRENGTHENS the shrinkage story: shrinkage is both simpler and better,
and its advantage is not a coincidence -- it provides implicit regularization toward the
global risk curve that the optimizer cannot replicate from the mixture constraint alone.
