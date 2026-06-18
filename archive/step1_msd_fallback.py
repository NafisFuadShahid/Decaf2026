"""Step 1 (MSD fallback): Download Medical Segmentation Decathlon Task01_BrainTumour.
Runs only if step1_status.json doesn't already have valid data."""
import os
import sys
import json
import tarfile
import urllib.request
import urllib.error
from pathlib import Path
from time import time

RESULTS = Path("C:/DeCaf/fed_crc_results")
DATA_DIR = Path("C:/DeCaf/data")
RESULTS.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

ERRORS = []

def log_error(msg):
    ERRORS.append(msg)
    print(f"[ERROR] {msg}", file=sys.stderr)


def step1_already_done():
    """Check if step1 completed successfully with a valid data dir."""
    p = RESULTS / "step1_status.json"
    if not p.exists():
        return False
    try:
        with open(p) as f:
            s = json.load(f)
        data_dir = s.get("data_dir")
        if data_dir and Path(data_dir).exists():
            # Check there are NIfTI files
            niis = list(Path(data_dir).rglob("*.nii.gz"))
            if len(niis) >= 5:
                print(f"Step 1 already done with {len(niis)} NIfTI files in {data_dir}")
                return True
    except Exception:
        pass
    return False


def try_partition_csv_github():
    """Download partition CSV from GitHub."""
    urls = [
        "https://raw.githubusercontent.com/FeTS-AI/Challenge/main/Task_1/partitioning_1.csv",
        "https://raw.githubusercontent.com/FeTS-AI/Challenge/master/Task_1/partitioning_1.csv",
        "https://raw.githubusercontent.com/FETS-AI/Challenge/main/Task_1/FeTS2022_partitioning_1.csv",
    ]
    out = DATA_DIR / "partitioning_1.csv"
    for url in urls:
        try:
            print(f"Trying partition CSV: {url}")
            urllib.request.urlretrieve(url, str(out))
            import pandas as pd
            df = pd.read_csv(out)
            if len(df) > 10:
                print(f"Got partition CSV: {len(df)} rows, cols: {df.columns.tolist()}")
                return str(out)
        except Exception as e:
            print(f"  Failed: {e}")
    return None


def download_msd():
    """Download MSD Task01_BrainTumour."""
    msd_dir = DATA_DIR / "Task01_BrainTumour"
    tar_path = DATA_DIR / "Task01_BrainTumour.tar"

    if msd_dir.exists():
        niis = list(msd_dir.rglob("*.nii.gz"))
        if len(niis) > 100:
            print(f"MSD already extracted: {len(niis)} NIfTI files")
            return str(msd_dir)

    # Try to download
    url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
    print(f"Downloading MSD Task01_BrainTumour from AWS S3...")
    print(f"  URL: {url}")
    print(f"  This is ~1.8GB — please wait...")

    start = time()
    last_report = start

    def progress(count, block_size, total_size):
        nonlocal last_report
        now = time()
        if now - last_report >= 30:
            if total_size > 0:
                pct = min(100, count * block_size / total_size * 100)
                elapsed = now - start
                print(f"  {pct:.1f}% ({count*block_size/1e9:.2f}GB) — {elapsed:.0f}s elapsed")
            last_report = now

    try:
        urllib.request.urlretrieve(url, str(tar_path), reporthook=progress)
        elapsed = time() - start
        print(f"  Download complete in {elapsed:.0f}s ({tar_path.stat().st_size/1e9:.2f}GB)")
    except urllib.error.URLError as e:
        log_error(f"MSD download failed: {e}")
        return None

    # Extract
    print("Extracting tar archive...")
    start_ext = time()
    try:
        with tarfile.open(str(tar_path), "r") as tar:
            tar.extractall(str(DATA_DIR))
        print(f"Extracted in {time()-start_ext:.0f}s")
    except Exception as e:
        log_error(f"MSD extraction failed: {e}")
        return None

    # Verify
    niis = list(msd_dir.rglob("*.nii.gz"))
    print(f"Verified: {len(niis)} NIfTI files in {msd_dir}")

    return str(msd_dir) if len(niis) > 0 else None


def main():
    print("=" * 60)
    print("STEP 1 (MSD FALLBACK): DATA DOWNLOAD")
    print("=" * 60)

    if step1_already_done():
        print("Step 1 already complete — nothing to do")
        return

    # Try partition CSV from GitHub (independent of data download)
    partition_csv = try_partition_csv_github()

    # Download MSD
    data_dir = download_msd()

    if data_dir is None:
        log_error("MSD download failed — cannot proceed")
        sys.exit(1)

    fallbacks = ["Used MSD Task01_BrainTumour (Synapse too slow for full FeTS download)"]
    if partition_csv is None:
        fallbacks.append("No real institutional partition CSV — will use synthetic partitions")

    status = {
        "data_source": "MSD_Task01_BrainTumour",
        "data_dir": data_dir,
        "partition_source": "github" if partition_csv else "synthetic",
        "partition_csv": partition_csv,
        "fallbacks": fallbacks,
        "errors": ERRORS,
    }

    with open(RESULTS / "step1_status.json", "w") as f:
        json.dump(status, f, indent=2)

    print("\nSTEP 1 (MSD) COMPLETE")
    print(f"  Data: {data_dir}")
    print(f"  Partition: {partition_csv or 'synthetic (TBD)'}")
    print(f"  Fallbacks: {fallbacks}")


if __name__ == "__main__":
    main()
