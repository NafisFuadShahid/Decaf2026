# Fed-CRC-Seg FINAL Verdict — FeTS-2022 REAL DATA

**Generated:** 2026-05-31 12:53:30
**Data:** REAL FeTS-2022 multi-institutional partitions (partitioning_1.csv)
**Subjects with tumor:** 1251
**Institutions (>= 6 subjects):** 20 / 23 total

---

## Final Decision: GREEN

Shrinkage (n0=5) improves on both axes with real FeTS-2022 data:
  0/20 violations (vs B3: 8/20)
  stretch 79.0x (vs B2: 98.4x, 20% reduction)

---

## Results Table (alpha = 0.1)

| Method                       |   Marginal |  Worst FNR | Violations |  Stretch |
|------------------------------|------------|------------|------------|----------|
| B3 Naive Pooled              |     0.0966 |     0.1681 |      8/20   |     1.52 |
| B2 Per-site Local            |     0.0630 |     0.0821 |      0/20   |    98.40 |
| Shrinkage n0=5               |     0.0651 |     0.0821 |      0/20   |    79.02 |
| Shrinkage n0=10              |     0.0668 |     0.1008 |      1/20   |    54.05 |
| Shrinkage n0=15              |     0.0684 |     0.1008 |      1/20   |     8.55 |
| Shrinkage n0=20              |     0.0707 |     0.1008 |      1/20   |     2.43 |
| Shrinkage n0=30              |     0.0707 |     0.1008 |      1/20   |     2.43 |
| Shrinkage n0=50              |     0.0707 |     0.1008 |      1/20   |     2.43 |
| Shrinkage n0=75              |     0.0627 |     0.1008 |      1/20   |     2.44 |
| Shrinkage n0=100             |     0.0627 |     0.1008 |      1/20   |     2.44 |
| Shrinkage n0=200             |     0.0636 |     0.1008 |      1/20   |     2.40 |
| Weighted Shared              |     0.0000 |     0.0000 |      0/20   |   138.43 |
| James-Stein                  |     0.0630 |     0.0821 |      0/20   |    98.40 |

---

## Alpha Sweep

  a=0.05: B3 worst=0.101 (7 viol.)   Shrinkage worst=0.027 (0 viol.)
  a=0.10: B3 worst=0.168 (8 viol.)   Shrinkage worst=0.082 (0 viol.)
  a=0.15: B3 worst=0.203 (5 viol.)   Shrinkage worst=0.150 (0 viol.)
  a=0.20: B3 worst=0.273 (7 viol.)   Shrinkage worst=0.203 (1 viol.)
Best shrinkage n0 = 5

---

## Per-Institution Breakdown

  Inst  1 (n= 511): B3 FNR=0.063/ok str=1.6 | Ours FNR=0.082 str=1.4
  Inst  2 (n=   6): B3 FNR=0.011/ok str=1.4 | Ours FNR=0.000 str=98.5
  Inst  3 (n=  15): B3 FNR=0.032/ok str=2.0 | Ours FNR=0.000 str=201.7
  Inst  4 (n=  47): B3 FNR=0.136/**FAIL** str=1.5 | Ours FNR=0.000 str=125.1
  Inst  5 (n=  22): B3 FNR=0.067/ok str=1.5 | Ours FNR=0.029 str=2.5
  Inst  6 (n=  34): B3 FNR=0.109/**FAIL** str=1.5 | Ours FNR=0.045 str=2.4
  Inst  7 (n=  12): B3 FNR=0.050/ok str=1.6 | Ours FNR=0.000 str=125.6
  Inst  8 (n=   8): B3 FNR=0.067/ok str=1.3 | Ours FNR=0.000 str=95.6
  Inst 10 (n=   8): B3 FNR=0.134/**FAIL** str=1.0 | Ours FNR=0.000 str=51.7
  Inst 11 (n=  14): B3 FNR=0.063/ok str=1.3 | Ours FNR=0.024 str=1.9
  Inst 12 (n=  11): B3 FNR=0.168/**FAIL** str=1.8 | Ours FNR=0.000 str=129.5
  Inst 13 (n=  35): B3 FNR=0.107/**FAIL** str=1.3 | Ours FNR=0.054 str=2.1
  Inst 14 (n=   6): B3 FNR=0.068/ok str=1.5 | Ours FNR=0.000 str=248.5
  Inst 15 (n=  13): B3 FNR=0.055/ok str=1.9 | Ours FNR=0.000 str=159.9
  Inst 16 (n=  30): B3 FNR=0.084/ok str=1.7 | Ours FNR=0.033 str=2.6
  Inst 17 (n=   9): B3 FNR=0.128/**FAIL** str=1.9 | Ours FNR=0.000 str=250.9
  Inst 18 (n= 382): B3 FNR=0.150/**FAIL** str=1.7 | Ours FNR=0.080 str=3.3
  Inst 20 (n=  33): B3 FNR=0.070/ok str=1.4 | Ours FNR=0.036 str=2.2
  Inst 21 (n=  35): B3 FNR=0.049/ok str=1.4 | Ours FNR=0.049 str=1.4
  Inst 22 (n=   7): B3 FNR=0.116/**FAIL** str=1.2 | Ours FNR=0.000 str=73.9

---

## Paper-Ready Numbers

**B3 Naive Pooled CRC:**
  Marginal FNR = 0.0966  (target: <= 0.1)
  Worst-site FNR = 0.1681  (+6.8pp above alpha)
  Sites violating alpha = 8/20 (40%)
  Avg stretch = 1.52x

**Our Shrinkage CRC (n0=5):**
  Marginal FNR = 0.0651
  Worst-site FNR = 0.0821
  Sites violating alpha = 0/20
  Avg stretch = 79.02x
  Stretch reduction vs B2 = 20%
  Pareto domination: partial

---

## Errors
- None

---

## Outputs — C:\DeCaf\fed_crc_results\fets_final\
- FINAL_VERDICT_FETS.md
- figure2_fets_real.png/.pdf  (publication-ready money figure)
- pareto_frontier_fets.png/.pdf
- n0_sweep_fets.png
- comparison_table_fets.csv
- per_institution_fets.csv
- volume_scores.pkl
