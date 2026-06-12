# Fed-CRC-Seg: FINAL REFINED Results

**Generated:** 2026-05-31 16:17:15
**Data:** FeTS-2022 — 1251 subjects, 20 institutions
**Lambda grid:** 200 points (interpolated fine grid)
**Seeds:** [42, 1337, 2024]
**Methods:** 53 total

---

## Main Results (mean +- std across 3 seeds, alpha=0.1)

| Method                         |       Violations |          Worst FNR |          Stretch |
|--------------------------------|------------------|--------------------|------------------|
| B3 Naive Pooled                |   8.0+-2.4    |  0.1779+-0.0146  |   1.46+-0.05 |
| B2 Per-site Local              |   1.3+-1.2    |  0.1110+-0.0101  |  83.23+-3.48 |
| Shrinkage n0=1                 |   1.3+-1.2    |  0.1099+-0.0090  |  81.05+-6.29 |
| Shrinkage n0=3                 |   1.3+-1.2    |  0.1094+-0.0084  |  75.26+-3.51 |
| Shrinkage n0=5                 |   1.3+-1.2    |  0.1086+-0.0077  |  63.50+-4.64 |
| Shrinkage n0=7                 |   1.3+-1.2    |  0.1083+-0.0075  |  37.93+-9.56 |
| Shrinkage n0=10                |   2.0+-0.8    |  0.1129+-0.0098  |  20.10+-10.15 |
| Shrinkage n0=15                |   2.7+-1.7    |  0.1185+-0.0049  |   4.38+-1.75 |
| Shrinkage n0=20                |   3.3+-2.6    |  0.1248+-0.0092  |   1.98+-0.09 |
| Shrinkage n0=30                |   4.3+-4.0    |  0.1340+-0.0099  |   1.86+-0.09 |
| Adaptive-sqrt c=10             |   1.3+-1.2    |  0.1094+-0.0084  |  71.71+-7.21 |
| Shrinkage n0=9                 |   1.3+-1.2    |  0.1118+-0.0111  |  28.75+-7.73 |

---

## Key Findings

**B3 Naive Pooled:**
  Violations = 8.0+-2.4 / 20
  Worst-site FNR = 0.1779+-0.0146  (+7.8pp above alpha)
  Stretch = 1.46x

**B2 Per-site Local:**
  Violations = 1.3+-1.2 / 20
  Worst-site FNR = 0.1110+-0.0101
  Stretch = 83.23x

**Best overall: Shrinkage n0=9**
  Violations = 1.3+-1.2 / 20
  Worst-site FNR = 0.1118+-0.0111
  Stretch = 28.75+-7.73x

**Best fixed-n0: Shrinkage n0=9**
  Violations = 1.3, Stretch = 28.75x

**Best adaptive: Adaptive-sqrt c=10**
  Violations = 1.3, Stretch = 71.71x

---

## n0 Pareto Trade-off (coverage <-> efficiency dial)

  n0=  1: 1.3+-1.2 viol., stretch=81.0+-6.3x, worst_FNR=0.1099
  n0=  2: 1.3+-1.2 viol., stretch=79.2+-7.1x, worst_FNR=0.1094
  n0=  3: 1.3+-1.2 viol., stretch=75.3+-3.5x, worst_FNR=0.1094
  n0=  5: 1.3+-1.2 viol., stretch=63.5+-4.6x, worst_FNR=0.1086
  n0=  7: 1.3+-1.2 viol., stretch=37.9+-9.6x, worst_FNR=0.1083
  n0= 10: 2.0+-0.8 viol., stretch=20.1+-10.1x, worst_FNR=0.1129
  n0= 12: 2.3+-1.2 viol., stretch=10.1+-5.8x, worst_FNR=0.1157
  n0= 15: 2.7+-1.7 viol., stretch=4.4+-1.8x, worst_FNR=0.1185
  n0= 20: 3.3+-2.6 viol., stretch=2.0+-0.1x, worst_FNR=0.1248
  n0= 25: 4.0+-3.6 viol., stretch=1.9+-0.1x, worst_FNR=0.1311
  n0= 30: 4.3+-4.0 viol., stretch=1.9+-0.1x, worst_FNR=0.1340

---

## Output Files — C:\DeCaf\fed_crc_results\final_refined\
- FINAL_REFINED_VERDICT.md
- figure2_final.png/.pdf  (money figure, fine grid)
- pareto_all_methods.png/.pdf
- n0_sweep_fine.png/.pdf
- all_methods_aggregated.csv
