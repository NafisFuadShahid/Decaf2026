"""Download MICCAI_FeTS2022_TrainingData.zip from Synapse and extract."""
import synapseclient
import os
import sys
import zipfile
from pathlib import Path
from time import time

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
    sys.exit(0)

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
        # First, list top-level members to understand structure
        names = z.namelist()
        print(f"  Zip contains {len(names)} entries", flush=True)
        # Show first 10 and look for partition CSV
        for n in names[:10]:
            print(f"  {n}", flush=True)
        csv_entries = [n for n in names if "partition" in n.lower() and n.endswith(".csv")]
        print(f"  Partition CSVs in zip: {csv_entries}", flush=True)
        # Extract everything
        z.extractall(str(FETS_DATA_DIR))
    print(f"Extraction complete in {time()-t0:.0f}s", flush=True)
except Exception as e:
    print(f"[ERROR] Extraction failed: {e}", flush=True)
    sys.exit(1)

# Verify
subject_dirs = [d for d in FETS_DATA_DIR.rglob("*")
                if d.is_dir() and (d.name.startswith("FeTS2022_") or d.name.startswith("BraTS"))]
print(f"Extracted {len(subject_dirs)} subject directories", flush=True)

# Find partition CSV
for csv_path in FETS_DATA_DIR.rglob("*partition*.csv"):
    print(f"Partition CSV: {csv_path}", flush=True)

print("DOWNLOAD AND EXTRACTION COMPLETE", flush=True)
