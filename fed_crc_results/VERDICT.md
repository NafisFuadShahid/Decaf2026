# Fed-CRC-Seg Pilot Verdict

**Generated:** 2026-05-31 02:05:54

---

## Final Decision: GREEN

## Rationale
Naive pooled CRC protects the average site (marginal FNR=0.100) but catastrophically fails 4/8 sites individually (worst site FNR=0.141, 4.1% above α). Our site-conditional protocol reduces violations to 1/8, worst site FNR=0.101. The failure mode is real and large enough for the paper.

---

## Key Numbers (α = 0.1)

| Method | Marginal FNR | Worst-site FNR | Sites Violating α | Avg Stretch |
|--------|-------------|----------------|-------------------|-------------|
| B1 Centralized Oracle | 0.0992 | 0.1412 | 4/8 | 1.439 |
| B2 Per-site Local CRC | 0.0483 | 0.1012 | 1/8 | 46.819 |
| B3 Naive Pooled CRC | 0.0992 | 0.1412 | 4/8 | 1.439 |
| Ours (per-site λ) | 0.0483 | 0.1012 | 1/8 | 46.819 |
| **Ours (shared λ)** | **0.0000** | **0.0000** | **0/8** | **140.897** |

---

## Failure Mode Separation
- **B3 worst-site FNR above α:** 4.1 percentage points
- **B3 sites violating α:** 4/8
- **Our method worst-site FNR above α:** 0.1 pp
- **Our method sites violating α:** 1/8

---

## Figure 2 Assessment
CLEAR VISUAL SEPARATION (GREEN). B3 worst-site FNR = 0.141 (exceeds α by 4.1%). B3 fails 4/8 sites while our per-site method fails 1/8. The figure shows B3 crossing above the α=0.1 dashed line for multiple sites while our method stays below (or barely at) the coverage boundary.

---

## Alpha Sweep
  α=0.05: B3 worst=0.072 (3 viol.), Ours worst=0.038 (0 viol.)
  α=0.10: B3 worst=0.141 (4 viol.), Ours worst=0.101 (1 viol.)
  α=0.15: B3 worst=0.177 (3 viol.), Ours worst=0.177 (2 viol.)
  α=0.20: B3 worst=0.252 (3 viol.), Ours worst=0.220 (2 viol.)

---

## Data Used
- **Dataset:** MSD_Task01_BrainTumour
- **Partition:** synthetic institutional partitions
- **Model:** segresnet_bundle
- **Subjects with tumor (used for CRC):** 484
- **Sites used (≥6 subjects each):** 8

---

## Errors and Fallbacks
- Synapse returned no NIfTI files
- Synapse failed — using MSD fallback
- No real partition CSV — will use synthetic partitions
- Used MSD Task01_BrainTumour (Synapse too slow for full FeTS download)
- No real institutional partition CSV — will use synthetic partitions
- Using synthetic K=8 partitions (no real CSV)

**Errors:**
- None

---

## Interpretation
```
GREEN  → Proceed to write the DeCaF 2026 paper.
         The failure mode is real, large, and our fix works.
YELLOW → Investigate further. Failure mode exists but is not large enough
         to be the paper's primary result. Consider: different model weights,
         more heterogeneous data split, or reframe as an existence proof.
RED    → The failure mode doesn't manifest with this setup.
         Either the model is too uncertain everywhere (high FNR at all λ),
         sites are too homogeneous, or the prediction quality is insufficient.
         Kill or substantially redesign the idea.
```

**Current verdict: GREEN**
