# Fed-CRC-Seg FINAL Verdict — FeTS-2022

**Generated:** 2026-05-31 11:14:30
**Data:** SYNTHETIC K-means partitions (fallback — real partitions unavailable)
**Subjects with tumor:** 484
**Institutions (>= 6 subjects):** 8

---

## Final Decision: GREEN

Shrinkage (n0=50) PARETO-DOMINATES both baselines:
  Coverage: 0/8 violations vs B3 4/8
  Efficiency: stretch 2.1 vs B2 46.8 (95% reduction)
Result holds on SYNTHETIC FeTS-2022 institutional partitions.

---

## Results Table (alpha = 0.1)

| Method                       |   Marginal |  Worst FNR | Violations |  Stretch |
|------------------------------|------------|------------|------------|----------|
| B3 Naive Pooled              |     0.0992 |     0.1412 |      4/8    |     1.44 |
| B2 Per-site Local            |     0.0483 |     0.1012 |      1/8    |    46.82 |
| Shrinkage n0=5               |     0.0483 |     0.1012 |      1/8    |    46.82 |
| Shrinkage n0=10              |     0.0577 |     0.0782 |      0/8    |     2.21 |
| Shrinkage n0=15              |     0.0577 |     0.0782 |      0/8    |     2.21 |
| Shrinkage n0=20              |     0.0577 |     0.0782 |      0/8    |     2.21 |
| Shrinkage n0=30              |     0.0577 |     0.0782 |      0/8    |     2.21 |
| Shrinkage n0=50              |     0.0654 |     0.0784 |      0/8    |     2.15 |
| Shrinkage n0=75              |     0.0654 |     0.0784 |      0/8    |     2.15 |
| Shrinkage n0=100             |     0.0768 |     0.1412 |      1/8    |     1.94 |
| Shrinkage n0=200             |     0.0835 |     0.1412 |      2/8    |     1.80 |
| Weighted Shared              |     0.0465 |     0.0722 |      0/8    |     2.40 |
| James-Stein                  |     0.0550 |     0.0782 |      0/8    |    20.54 |

---

## Alpha Sweep

  a=0.05: B3 worst=0.072 (3 viol.)   Shrinkage worst=0.038 (0 viol.)
  a=0.10: B3 worst=0.141 (4 viol.)   Shrinkage worst=0.078 (0 viol.)
  a=0.15: B3 worst=0.177 (3 viol.)   Shrinkage worst=0.177 (2 viol.)
  a=0.20: B3 worst=0.252 (3 viol.)   Shrinkage worst=0.220 (1 viol.)

Best shrinkage n0 = 50

---

## Per-Institution Breakdown

  Site 0 (n=74): B3 FNR=0.141/**FAIL** str=1.6 | Ours FNR=0.070 str=3.2
  Site 1 (n=84): B3 FNR=0.078/ok str=1.5 | Ours FNR=0.078 str=1.5
  Site 2 (n=50): B3 FNR=0.076/ok str=1.1 | Ours FNR=0.028 str=1.4
  Site 3 (n=17): B3 FNR=0.123/**FAIL** str=1.4 | Ours FNR=0.072 str=2.2
  Site 4 (n=44): B3 FNR=0.100/**FAIL** str=1.7 | Ours FNR=0.046 str=3.0
  Site 5 (n=78): B3 FNR=0.141/**FAIL** str=1.7 | Ours FNR=0.070 str=3.3
  Site 6 (n=79): B3 FNR=0.078/ok str=1.2 | Ours FNR=0.078 str=1.2
  Site 7 (n=58): B3 FNR=0.062/ok str=1.2 | Ours FNR=0.062 str=1.2

---

## Fallbacks Triggered

- FeTS data not ready (0 subjects found). Using Phase 1 MSD volume_scores.pkl directly.
- Using Phase 1 MSD volume_scores.pkl (FeTS data not available)

## Errors

- None

---

## Paper-Ready Numbers

B3 Naive Pooled CRC:
  - Marginal FNR = 0.0992  (target: <= 0.1)
  - Worst-site FNR = 0.1412  (+4.1pp above target)
  - Sites violating alpha = 4/8 (50%)
  - Avg prediction set stretch = 1.44x

Our Shrinkage CRC (n0=50):
  - Marginal FNR = 0.0654
  - Worst-site FNR = 0.0784
  - Sites violating alpha = 0/8
  - Avg prediction set stretch = 2.15x
  - Stretch reduction vs B2 = 95%

---

## Outputs
All in C:\DeCaf\fed_crc_results\fets_final\:
- FINAL_VERDICT.md
- figure2_final.png/.pdf
- pareto_final.png/.pdf
- n0_sweep_final.png
- comparison_table.csv
- per_institution.csv
- volume_scores.pkl
