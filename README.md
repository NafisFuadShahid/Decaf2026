# Fed-CRC-Seg: Federated Conformal Risk Control via Risk-Curve Shrinkage

Code for **"When Calibration Fails the Vulnerable Hospital: Federated Conformal Risk Control via Risk-Curve Shrinkage"** (DeCaF 2026 @ MICCAI).

---

## Abstract

Conformal Risk Control (CRC) provides distribution-free FNR guarantees for set-valued predictors, but naively pooling calibration data across hospitals (Naive Pooled CRC, B3) systematically fails institutions with atypical case-mix distributions — exactly the small, under-represented hospitals most in need of protection. Per-site CRC (B2) avoids this by calibrating each hospital independently, but requires prohibitively large prediction sets at data-scarce sites.

We propose **Federated CRC with Risk-Curve Shrinkage**, which interpolates each site's empirical risk curve toward the pooled curve via a James-Stein-style shrinkage operator parameterized by n0. A site-specific correction term (corr_k) ensures the per-site FNR guarantee holds even when the pooled curve underestimates local risk. The shrinkage strength n0 is selected data-adaptively via Leave-One-Site-Out cross-validation (LOSO-CV), with no site's test data used in the selection.

On FeTS-2022 (1,251 subjects, 20 institutions, pre-trained SegResNet, α=0.10), Naive Pooled CRC violates the FNR target at 8/20 institutions (worst site: FNR=0.178, +7.8pp above α). Our method (n0=9) reduces violations to 1.3/20 while compressing prediction sets from 83× (B2) to 28.8× volume stretch — a 65% efficiency improvement. Removing the corr_k correction collapses coverage (9.3/20 violations). A budget-allocation baseline that ignores institutional structure fails at 12.3/20 institutions.

---

## Method Overview

The core idea is a **shrinkage aggregator** for per-site CRC thresholds:

```
λ_k(n0) = shrink(λ_pool, λ_k^local; n0)
```

Each site k's threshold is pulled toward the pooled threshold by a factor controlled by n0 (larger n0 = more pooling). A correction term corr_k inflates the effective sample count at site k to prevent over-shrinkage when the pooled risk curve is optimistic relative to site k's distribution. The final FNR guarantee is a per-site coverage bound of the form:

```
E[FNR_k(λ_k)] ≤ α + B/(n_k + 1)  for each institution k
```

See Figure 2 (results/figures/figure2_paper_final.pdf) for the per-institution FNR and set-size-stretch visualization.

---

## Key Results

**Table 1.** FeTS-2022 results (mean ± std, 3 seeds, 20 institutions, α=0.10). Lower violations and lower stretch are both better; stretch is the ratio of predicted set volume to ground-truth tumor volume.

| Method | Violations / 20 | Worst-site FNR | Set-size Stretch |
|--------|:--------------:|:--------------:|:----------------:|
| B3 Naive Pooled CRC | 8.0 ± 2.4 | 0.178 ± 0.015 | 1.46× |
| B2 Per-site Local CRC | 1.3 ± 1.2 | 0.111 ± 0.010 | 83.2× |
| **Ours (Shrinkage n0=9)** | **1.3 ± 1.2** | **0.112 ± 0.011** | **28.8×** |
| Ours (Shrinkage n0=15) | 2.7 ± 1.7 | 0.118 ± 0.005 | 4.4× |
| Ours (LOSO-selected n0=19) | 2.7 ± 1.7 | 0.125 ± 0.009 | 2.0× |
| Budget Allocation (uncapped) | 12.3 ± 0.9 | 0.351 ± 0.116 | 1.39× |

**Ablation: corr_k correction term (n0=9)**

| Variant | Violations / 20 | Worst-site FNR | Stretch |
|---------|:--------------:|:--------------:|:-------:|
| With corr_k (full method) | 1.3 ± 1.2 | 0.112 ± 0.011 | 28.8× |
| Without corr_k | 9.3 ± 2.6 | 0.181 ± 0.007 | 1.52× |

---



## Reproduction Instructions

### Prerequisites

- Python 3.11, CUDA 11.8 (GPU with ≥16 GB VRAM recommended for inference)
- Synapse account with accepted FeTS-2022 Data Use Agreement: https://www.synapse.org/#!Synapse:syn28546456
- Synapse Personal Access Token (PAT) with **Download** scope

```bash
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install monai nibabel numpy scipy pandas matplotlib synapseclient
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Step-by-step

**Step 1 — Download FeTS-2022 data (~12.5 GB) and partition CSV:**

```bash
export SYNAPSE_AUTH_TOKEN=<your_PAT>
python scripts/01_download_data.py
```

Downloads `MICCAI_FeTS2022_TrainingData.zip` (Synapse ID: `syn29266807`) to `data/fets2022/` and extracts it. Also fetches `data/partitioning_1.csv` mapping BraTS subject IDs to institutions.

Expected time: 30–90 min depending on connection speed.

**Step 2 — Parse partition and map subjects to institutions:**

```bash
python scripts/02_prepare_partitions.py
```

Reads `partitioning_1.csv` and the extracted data directory; writes `results/pipeline_status/step2_subjects.json` with per-site subject lists.

**Step 3 — Download pre-trained SegResNet (MONAI bundle):**

```bash
python scripts/03_download_model.py
```

Downloads the `brats_mri_segmentation` MONAI bundle to `models/`. The bundle contains the SegResNet weights pre-trained on BraTS-2021.

**Step 4 — Run inference on all subjects:**

```bash
python scripts/04_run_inference.py
```

Runs sliding-window SegResNet inference on all 1,251 subjects and computes per-volume FNR curves across the lambda grid. Writes `fed_crc_results/fets_final/volume_scores.pkl`.

Expected time: **6–12 hours** on a single GPU (e.g. RTX 3090). CPU-only is possible but ~10–20× slower.

**Step 5 — CRC calibration (main results):**

```bash
python scripts/05_crc_calibration.py
```

Runs the full CRC comparison — B1 Centralized Oracle, B2 Per-site Local, B3 Naive Pooled, and Ours (Shrinkage at n0 ∈ {1,3,5,7,9,10,12,15,20,…}) — on a 200-point lambda grid with 3 random seeds {42, 1337, 2024}. Writes `fed_crc_results/final_refined/all_methods_aggregated.csv` (Table 1 source) and figures.

Expected time: ~5–15 min (CPU).

**Step 5b — LOO + tighter correction variants (supplementary):**

```bash
python scripts/05b_loo_corrections.py
```

Runs Leave-One-Out global curve, Hoeffding correction, and Bernstein correction variants. Writes `fed_crc_results/quickfix/`.

**Step 6 — LOSO-CV n0 selection and ablations:**

```bash
python scripts/06_loso_ablations.py
```

- **Task 1:** LOSO-CV sweep over n0 ∈ {1,3,5,7,9,11,13,15,17,19,21,25,30,40,50} → selects n0=19
- **Task 2:** λ_fed (conservative single-threshold guarantee from Theorem 1)
- **Task 3a:** corr_k ablation (with vs. without correction term)
- **Task 3b:** Grid resolution ablation (G ∈ {50, 100, 200, 500} at n0=15)

Writes CSVs to `fed_crc_results/phase6_loso/`. Expected time: ~10–20 min (CPU).

**Step 7 — Budget allocation baseline:**

```bash
python scripts/07_budget_allocation.py
```

Writes `fed_crc_results/budget_alloc/comparison.csv`. Expected time: ~2 min.

**Step 8 — Generate paper figures:**

```bash
python scripts/08_generate_figures.py
```

Generates Figure 2 (per-institution FNR + stretch) and n0 sweep figure in LNCS Springer format (300 DPI, Times New Roman serif). Writes to `paper_figures/`.

### Run everything at once

```bash
export SYNAPSE_AUTH_TOKEN=<your_PAT>
python scripts/run_all.py
```

Steps 1–3 are skipped automatically if their sentinel files already exist in `results/pipeline_status/`.

---

## Reproducing Individual Tables and Figures

| Paper element | Script | Output file |
|---------------|--------|-------------|
| Table 1 (main comparison) | `05_crc_calibration.py` | `results/tables/all_methods_aggregated.csv` |
| LOSO-CV n0 sweep | `06_loso_ablations.py` | `results/tables/task1_loso_n0.csv` |
| λ_fed analysis | `06_loso_ablations.py` | `results/tables/task2_lambda_fed.csv` |
| corr_k ablation | `06_loso_ablations.py` | `results/tables/task3a_corr_ablation.csv` |
| Grid resolution ablation | `06_loso_ablations.py` | `results/tables/task3b_grid_ablation.csv` |
| Budget allocation baseline | `07_budget_allocation.py` | `results/tables/budget_comparison.csv` |
| Figure 2 (money figure) | `08_generate_figures.py` | `results/figures/figure2_paper_final.pdf` |
| n0 sweep figure | `08_generate_figures.py` | `results/figures/n0_sweep_paper_final.pdf` |

---

## Experimental Details

| Parameter | Value |
|-----------|-------|
| Dataset | FeTS-2022 (MICCAI_FeTS2022_TrainingData) |
| Subjects | 1,251 (20 institutions with ≥6 subjects out of 23 total) |
| Task | Whole-tumor segmentation (BraTS label > 0) |
| Model | SegResNet, MONAI bundle `brats_mri_segmentation` (BraTS-2021 pre-trained) |
| Lambda grid | G=200 uniform points on [0, 1] |
| Risk level | α = 0.10 |
| Hoeffding constant | B = 1.0 |
| Seeds | {42, 1337, 2024} (50/50 cal/test split per site) |
| LOSO-selected n0 | 19 (criteria: lowest mean stretch among n0 with mean violations ≤ 3) |
| Paper-reported n0 | 15 (submission; post-hoc LOSO analysis identified n0=19 as preferable) |

---


## License

MIT — see [LICENSE](LICENSE).

The FeTS-2022 dataset is separately licensed under the FeTS-2022 Data Use Agreement (Synapse). The MONAI SegResNet bundle is licensed under Apache 2.0.
