"""
Step 1: Download FeTS-2022 data and partition CSV.

Part A downloads MICCAI_FeTS2022_TrainingData.zip from Synapse (syn29266807)
and extracts it. Requires SYNAPSE_AUTH_TOKEN env var and an accepted FeTS-2022
Data Use Agreement at https://www.synapse.org/access.

Part B fetches partitioning_1.csv (subject→institution mapping) from the
FeTS-AI GitHub repo. If all GitHub URLs fail, the CSV is typically bundled
inside the zip — check data/fets2022/MICCAI_FeTS2022_TrainingData/.

Prerequisites:
  pip install synapseclient pandas
  export SYNAPSE_AUTH_TOKEN=<your Synapse PAT with Download scope>
"""
import synapseclient
import urllib.request
import os
import sys
import io
import zipfile
from pathlib import Path
from time import time

if sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─── PART A: Download and extract FeTS-2022 zip ───────────────────────────────

syn = synapseclient.Synapse()
syn.login(authToken=os.environ["SYNAPSE_AUTH_TOKEN"], silent=True)
print("Synapse login OK", flush=True)

FETS_DATA_DIR = Path("C:/DeCaf/data/fets2022")
FETS_DATA_DIR.mkdir(parents=True, exist_ok=True)

ZIP_SYN_ID = "syn29266807"
zip_path = FETS_DATA_DIR / "MICCAI_FeTS2022_TrainingData.zip"

# Check if already extracted
subject_dirs = [d for d in FETS_DATA_DIR.iterdir()
                if d.is_dir() and (d.name.startswith("FeTS2022_") or d.name.startswith("BraTS"))]
if len(subject_dirs) > 100:
    print(f"Already extracted: {len(subject_dirs)} subject directories found", flush=True)
else:
    # Download the zip
    if zip_path.exists() and zip_path.stat().st_size > 1e9:
        print(f"Zip already exists: {zip_path.stat().st_size/1e9:.2f}GB", flush=True)
    else:
        print(f"Downloading {ZIP_SYN_ID} to {zip_path}...", flush=True)
        print("Expected size: ~12.47 GB", flush=True)
        t0 = time()
        try:
            ent = syn.get(ZIP_SYN_ID, downloadLocation=str(FETS_DATA_DIR))
            print(f"Download complete in {time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"[ERROR] Download failed: {e}", flush=True)
            sys.exit(1)

    # Find the zip
    zips = list(FETS_DATA_DIR.glob("*.zip"))
    if not zips:
        print("[ERROR] No zip file found after download!", flush=True)
        sys.exit(1)
    zip_path = zips[0]
    print(f"Zip: {zip_path}, size={zip_path.stat().st_size/1e9:.2f}GB", flush=True)

    # Extract
    print("Extracting zip...", flush=True)
    t0 = time()
    try:
        with zipfile.ZipFile(str(zip_path), "r") as z:
            names = z.namelist()
            print(f"  Zip contains {len(names)} entries", flush=True)
            for n in names[:10]:
                print(f"  {n}", flush=True)
            csv_entries = [n for n in names if "partition" in n.lower() and n.endswith(".csv")]
            print(f"  Partition CSVs in zip: {csv_entries}", flush=True)
            z.extractall(str(FETS_DATA_DIR))
        print(f"Extraction complete in {time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}", flush=True)
        sys.exit(1)

    subject_dirs = [d for d in FETS_DATA_DIR.rglob("*")
                    if d.is_dir() and (d.name.startswith("FeTS2022_") or d.name.startswith("BraTS"))]
    print(f"Extracted {len(subject_dirs)} subject directories", flush=True)

    for csv_path in FETS_DATA_DIR.rglob("*partition*.csv"):
        print(f"Partition CSV found in zip: {csv_path}", flush=True)

print("PART A COMPLETE", flush=True)

# ─── PART B: Fetch partitioning_1.csv ────────────────────────────────────────
# The CSV maps each BraTS subject ID to its FeTS-2022 institution (1–23).
# It may already exist inside the extracted zip; if not, try GitHub mirrors.

DATA_DIR = Path("C:/DeCaf/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
csv_dest = DATA_DIR / "partitioning_1.csv"

# Check if already available inside the extracted archive
extracted_csvs = list(FETS_DATA_DIR.rglob("*partition*.csv"))
if extracted_csvs:
    import shutil
    shutil.copy(str(extracted_csvs[0]), str(csv_dest))
    print(f"Copied partition CSV from zip: {extracted_csvs[0]}", flush=True)
else:
    print("Partition CSV not in zip — trying GitHub mirrors...", flush=True)
    urls = [
        "https://raw.githubusercontent.com/FeTS-AI/Challenge/main/Task_1/partitioning_1.csv",
        "https://raw.githubusercontent.com/FeTS-AI/Challenge/master/Task_1/partitioning_1.csv",
        "https://raw.githubusercontent.com/FETS-AI/Challenge/main/Task_1/partitioning_1.csv",
        "https://raw.githubusercontent.com/FETS-AI/Challenge/main/Task_1/FeTS2022_partitioning_1.csv",
        "https://raw.githubusercontent.com/FETS-AI/Challenge/main/partitioning_1.csv",
        "https://raw.githubusercontent.com/cbica/CaPTk/master/src/applications/FeTS/data/partitioning_1.csv",
    ]
    found = False
    for url in urls:
        try:
            urllib.request.urlretrieve(url, str(csv_dest))
            with open(csv_dest) as f:
                content = f.read()
            if len(content) > 100 and "," in content:
                import pandas as pd
                df = pd.read_csv(csv_dest)
                if len(df) > 50:
                    print(f"SUCCESS: {url}", flush=True)
                    print(f"  Rows: {len(df)}, Cols: {df.columns.tolist()}", flush=True)
                    found = True
                    break
                else:
                    print(f"  Too small ({len(df)} rows): {url}", flush=True)
            else:
                print(f"  Not a CSV: {url}", flush=True)
        except Exception as e:
            print(f"  FAIL {url}: {e}", flush=True)

    if not found:
        print("\n[WARNING] All GitHub attempts failed.", flush=True)
        print("The partition CSV requires Synapse DUA acceptance.", flush=True)
        print("  1. Accept the FeTS-2022 DUA at https://www.synapse.org/access", flush=True)
        print("  2. Re-run this script with a valid SYNAPSE_AUTH_TOKEN", flush=True)
        print("  The CSV is bundled inside the zip upon successful download.", flush=True)

print("PART B COMPLETE", flush=True)
print("DATA DOWNLOAD DONE", flush=True)
