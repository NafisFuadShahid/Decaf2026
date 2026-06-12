# Fed-CRC-Seg Quick Fix Results

**Generated:** 2026-05-31 16:50:05
**Data:** FeTS-2022, 1251 subjects, 20 institutions
**Corrections tested:** LOO global curve, Hoeffding, Empirical Bernstein, Adaptive+LOO+Hoeff
**Seeds:** [42, 1337, 2024]

---

## Did the Fixes Help?

### Original best (n0=9):
  Violations: 1.3+-1.2
  Worst FNR: 0.1118+-0.0111
  Stretch: 28.75+-7.73

### Best fixed method: LOO+Bern n0=5
  Violations: 0.0+-0.0
  Worst FNR: 0.0500+-0.0105
  Stretch: 118.45+-6.02

---

## Key Comparison Table

| Method                     |     Violations |        Worst FNR |        Stretch |
|----------------------------|----------------|------------------|----------------|
| B3 Naive Pooled            |   8.0+-2.4      |  0.1779+-0.0146  |   1.46+-0.05 |
| B2 Per-site                |   1.3+-1.2      |  0.1110+-0.0101  |  83.23+-3.48 |
| Orig n0=9                  |   1.3+-1.2      |  0.1118+-0.0111  |  28.75+-7.73 |
| Orig n0=15                 |   2.7+-1.7      |  0.1185+-0.0049  |   4.38+-1.75 |
| LOO n0=9                   |   1.3+-1.2      |  0.1118+-0.0111  |  28.76+-7.73 |
| LOO n0=15                  |   2.7+-1.7      |  0.1220+-0.0096  |   4.38+-1.76 |
| LOO+Hoeff n0=9             |   0.0+-0.0      |  0.0147+-0.0017  | 127.92+-5.30 |
| LOO+Hoeff n0=15            |   0.0+-0.0      |  0.0147+-0.0017  | 127.92+-5.30 |
| LOO+Bern n0=9              |   0.0+-0.0      |  0.0475+-0.0086  | 118.46+-6.02 |
| LOO+Bern n0=15             |   0.0+-0.0      |  0.0475+-0.0086  | 118.46+-6.02 |
| Full-fix c=3               |   0.0+-0.0      |  0.0147+-0.0017  | 127.92+-5.30 |
| Full-fix c=5               |   0.0+-0.0      |  0.0147+-0.0017  | 127.92+-5.30 |
| Full-fix c=8               |   0.0+-0.0      |  0.0147+-0.0017  | 127.92+-5.30 |
| Full-fix c=10              |   0.0+-0.0      |  0.0147+-0.0017  | 127.92+-5.30 |
| Full-fix c=15              |   0.0+-0.0      |  0.0147+-0.0017  | 127.92+-5.30 |
| Full-fix c=20              |   0.0+-0.0      |  0.0147+-0.0017  | 127.92+-5.30 |

---

## Fix-by-Fix Analysis

### At n0=9 (coverage-priority operating point):
  Orig n0=9               : 1.3±1.2 viol, stretch=28.75x, worst=0.1118
  LOO n0=9                : 1.3±1.2 viol, stretch=28.76x, worst=0.1118
  LOO+Hoeff n0=9          : 0.0±0.0 viol, stretch=127.92x, worst=0.0147
  LOO+Bern n0=9           : 0.0±0.0 viol, stretch=118.46x, worst=0.0475

### At n0=15 (efficiency-priority operating point):
  Orig n0=15              : 2.7±1.7 viol, stretch=4.38x, worst=0.1185
  LOO n0=15               : 2.7±1.7 viol, stretch=4.38x, worst=0.1220
  LOO+Hoeff n0=15         : 0.0±0.0 viol, stretch=127.92x, worst=0.0147
  LOO+Bern n0=15          : 0.0±0.0 viol, stretch=118.46x, worst=0.0475

---

## Paper Recommendation

**Anchors:**
- B3 Naive Pooled: 8.0±2.4 viol, worst=0.1779 (+7.8pp), stretch=1.46x
- B2 Per-site: 1.3±1.2 viol, worst=0.1110, stretch=83.23x

**Recommended headline method:** LOO+Bern n0=5
  Violations: 0.0±0.0
  Stretch: 118.45±6.02x
  Improvement vs original: see table above.

Files: C:\DeCaf\fed_crc_results\quickfix\
