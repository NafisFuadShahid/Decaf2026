# Archive

This directory contains development-phase scripts that are **not needed to reproduce the paper results**. They are kept for historical reference only.

| File | What it was | Superseded by |
|------|-------------|---------------|
| `step1_download_data.py` | Early data downloader with multiple Synapse ID fallbacks and MSD fallback | `scripts/01_download_data.py` |
| `step1_msd_fallback.py` | Downloaded MSD Task01_BrainTumour as a fallback dataset (not the paper data) | — |
| `step4_inference.py` | Inference step for the early MSD-based pipeline | `scripts/04_run_inference.py` |
| `step5_crc_calibration.py` | Early CRC calibration on MSD data with coarse lambda grid | `scripts/05_crc_calibration.py` |
| `step6_figures.py` | Figure generation for the early MSD pipeline | `scripts/08_generate_figures.py` |
| `step7_verdict.py` | Verdict writer for the early pipeline | — |
| `phase2_shrinkage.py` | First shrinkage prototype (pre-corr_k formulation) | `scripts/05_crc_calibration.py` |
| `phase3_fets_final.py` | Earlier Phase 3 with MSD fallback; `phase3_fets_real.py` supersedes it | `scripts/04_run_inference.py` |
| `make_paper_figures.py` | First figure generation attempt (overlapping legend elements) | `scripts/08_generate_figures.py` |
| `run_pipeline_from_step2.py` | Partial runner (steps 2–7) used during iterative development | `scripts/run_all.py` |
