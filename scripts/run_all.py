"""Master runner: executes all pipeline steps in sequence with error recovery."""
import subprocess
import sys
import os
import json
from pathlib import Path

PY = r"C:\Users\fuadn\anaconda3\python.exe"
SCRIPTS = Path("C:/DeCaf/scripts")
RESULTS = Path("C:/DeCaf/fed_crc_results")
RESULTS.mkdir(parents=True, exist_ok=True)

errors_log = []

def run_step(script_name, step_label):
    """Run a pipeline step and handle errors."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {step_label}")
    print(f"{'='*70}")

    result = subprocess.run(
        [PY, str(SCRIPTS / script_name)],
        env={**os.environ, "TF_ENABLE_ONEDNN_OPTS": "0"},
        capture_output=False,  # Let output stream to console
    )

    if result.returncode != 0:
        msg = f"{step_label} exited with code {result.returncode}"
        print(f"[WARNING] {msg}")
        errors_log.append(msg)
        return False

    return True


def check_prerequisite(json_file):
    """Check if a prerequisite step completed successfully."""
    path = RESULTS / json_file
    if not path.exists():
        return False
    try:
        with open(path) as f:
            json.load(f)
        return True
    except:
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("FED-CRC-SEG PILOT — MASTER RUNNER")
    print("=" * 70)

    # Step 1: Download data (only if not already done)
    if not check_prerequisite("step1_status.json"):
        ok = run_step("step1_download_data.py", "Step 1: Data Download")
        if not ok:
            print("Step 1 failed — check logs above")
            sys.exit(1)
    else:
        print("Step 1 already complete — skipping")

    # Step 2: Parse partition
    if not check_prerequisite("step2_subjects.json"):
        ok = run_step("step2_parse_partition.py", "Step 2: Parse Partition & Find Subjects")
        if not ok:
            sys.exit(1)
    else:
        print("Step 2 already complete — skipping")

    # Step 3: Download model
    if not check_prerequisite("step3_model.json"):
        ok = run_step("step3_download_model.py", "Step 3: Download Model")
        if not ok:
            sys.exit(1)
    else:
        print("Step 3 already complete — skipping")

    # Step 4: Inference
    if not check_prerequisite("step4_status.json"):
        ok = run_step("step4_inference.py", "Step 4: Inference + CRC Score Computation")
        if not ok:
            sys.exit(1)
    else:
        print("Step 4 already complete — skipping")

    # Step 5: CRC calibration
    if not check_prerequisite("step5_crc_results.json"):
        ok = run_step("step5_crc_calibration.py", "Step 5: CRC Calibration & Evaluation")
        if not ok:
            sys.exit(1)
    else:
        print("Step 5 already complete — skipping")

    # Step 6: Figures
    ok = run_step("step6_figures.py", "Step 6: Generate Figures")
    if not ok:
        print("Figure generation failed — continuing to verdict")

    # Step 7: Verdict
    ok = run_step("step7_verdict.py", "Step 7: Write VERDICT.md")
    if not ok:
        print("Verdict failed")
        sys.exit(1)

    print("\n" + "="*70)
    print("ALL STEPS COMPLETE")
    print("="*70)
    print(f"Results in: C:/DeCaf/fed_crc_results/")
