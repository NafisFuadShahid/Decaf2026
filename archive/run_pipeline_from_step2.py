"""Run steps 2-7 of the pipeline, assuming step1 and step3 are already done."""
import subprocess
import sys
import os
import json
from pathlib import Path

PY = r"C:\Users\fuadn\anaconda3\python.exe"
SCRIPTS = Path("C:/DeCaf/scripts")
RESULTS = Path("C:/DeCaf/fed_crc_results")
RESULTS.mkdir(parents=True, exist_ok=True)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def run_step(script_name, step_label, check_output=None):
    """Run a pipeline step."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {step_label}")
    print(f"{'='*70}", flush=True)

    result = subprocess.run(
        [PY, str(SCRIPTS / script_name)],
        env={**os.environ},
    )

    if result.returncode != 0:
        print(f"[WARNING] {step_label} exited with code {result.returncode}", flush=True)
        return False

    if check_output and not (RESULTS / check_output).exists():
        print(f"[WARNING] Expected output {check_output} not created", flush=True)
        return False

    return True


def check_ok(json_file):
    p = RESULTS / json_file
    if not p.exists():
        return False
    try:
        with open(p) as f:
            json.load(f)
        return True
    except:
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("FED-CRC-SEG: RUNNING STEPS 2-7")
    print("=" * 70, flush=True)

    # Step 2: Parse partition
    ok = run_step("step2_parse_partition.py", "Step 2: Parse Partition & Find Subjects",
                  check_output="step2_subjects.json")
    if not ok:
        sys.exit(1)

    # Step 4: Inference (step3 already done in parallel)
    if not check_ok("step4_status.json"):
        ok = run_step("step4_inference.py", "Step 4: Inference + CRC Score Computation",
                      check_output="step4_status.json")
        if not ok:
            sys.exit(1)
    else:
        print("Step 4 already complete")

    # Step 5: CRC calibration
    ok = run_step("step5_crc_calibration.py", "Step 5: CRC Calibration & Evaluation",
                  check_output="step5_crc_results.json")
    if not ok:
        sys.exit(1)

    # Step 6: Figures
    run_step("step6_figures.py", "Step 6: Generate Figures")

    # Step 7: Verdict
    ok = run_step("step7_verdict.py", "Step 7: Write VERDICT.md",
                  check_output="VERDICT.md")
    if not ok:
        sys.exit(1)

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("Results: C:/DeCaf/fed_crc_results/")
