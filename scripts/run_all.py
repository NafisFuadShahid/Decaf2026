"""
Fed-CRC-Seg end-to-end pipeline runner (DeCaF 2026).

Runs all 8 steps in sequence. Steps 01–03 are skipped automatically if their
outputs already exist (idempotent). Steps 04–08 always run to allow re-running
calibration experiments without repeating inference.

Usage:
  python scripts/run_all.py

Prerequisites: see README.md — data and model must be downloaded first, or set
SYNAPSE_AUTH_TOKEN and let step 01 handle it.
"""
import subprocess
import sys
import os
from pathlib import Path

PY = sys.executable  # use whichever python launched this script
SCRIPTS = Path(__file__).parent
ROOT = SCRIPTS.parent
PIPELINE_STATUS = ROOT / "results" / "pipeline_status"
PIPELINE_STATUS.mkdir(parents=True, exist_ok=True)

env = {**os.environ, "TF_ENABLE_ONEDNN_OPTS": "0"}


def run_step(script_name, step_label, fatal=True):
    print(f"\n{'='*70}")
    print(f"RUNNING: {step_label}")
    print(f"{'='*70}", flush=True)
    result = subprocess.run([PY, str(SCRIPTS / script_name)], env=env)
    if result.returncode != 0:
        print(f"[{'FATAL' if fatal else 'WARNING'}] {step_label} exited with code {result.returncode}")
        if fatal:
            sys.exit(result.returncode)
        return False
    return True


def already_done(sentinel_file):
    """Return True if a sentinel output file from a previous run exists."""
    return (PIPELINE_STATUS / sentinel_file).exists()


if __name__ == "__main__":
    print("=" * 70)
    print("FED-CRC-SEG — MASTER PIPELINE RUNNER")
    print("DeCaF 2026: Federated Conformal Risk Control via Risk-Curve Shrinkage")
    print("=" * 70)

    # 01: Download FeTS-2022 data + partition CSV
    if already_done("step1_status.json"):
        print("01 already done (step1_status.json found) — skipping data download")
    else:
        run_step("01_download_data.py", "01: Download FeTS-2022 data + partition CSV")

    # 02: Parse partition CSV and map subjects to institutions
    if already_done("step2_subjects.json"):
        print("02 already done (step2_subjects.json found) — skipping partition parse")
    else:
        run_step("02_prepare_partitions.py", "02: Parse partition and map subjects to sites")

    # 03: Download pre-trained SegResNet (MONAI bundle)
    if already_done("step3_model.json"):
        print("03 already done (step3_model.json found) — skipping model download")
    else:
        run_step("03_download_model.py", "03: Download SegResNet MONAI bundle")

    # 04: Run inference on all subjects → produces fets_final/volume_scores.pkl
    run_step("04_run_inference.py", "04: Run SegResNet inference on FeTS-2022 (~6–12 h on GPU)")

    # 05: CRC calibration — fine 200-pt grid, 3 seeds, B1/B2/B3/Ours
    run_step("05_crc_calibration.py", "05: CRC calibration (200-pt grid, 3 seeds)")

    # 05b: LOO + Hoeffding/Bernstein correction variants (supplementary)
    run_step("05b_loo_corrections.py", "05b: LOO + tighter correction variants", fatal=False)

    # 06: LOSO-CV n0 selection + lambda_fed + corr_k ablation + grid ablation
    run_step("06_loso_ablations.py", "06: LOSO-CV and ablation experiments")

    # 07: Budget allocation baseline
    run_step("07_budget_allocation.py", "07: Budget allocation baseline")

    # 08: Generate all paper figures
    run_step("08_generate_figures.py", "08: Generate paper figures (paper_figures/)")

    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETE")
    print("=" * 70)
    print(f"Tables:  {ROOT / 'results' / 'tables'}/")
    print(f"Figures: {ROOT / 'results' / 'figures'}/")
    print(f"Paper figures (final): paper_figures/")
