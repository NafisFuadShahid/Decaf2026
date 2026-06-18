"""Step 1: Download FeTS-2022 data from Synapse with fallbacks."""
import os
import sys
import json
import urllib.request
from pathlib import Path

ERRORS = []

def log_error(msg):
    ERRORS.append(msg)
    print(f"[ERROR] {msg}", file=sys.stderr)

def try_synapse_download():
    import synapseclient

    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    syn = synapseclient.Synapse()

    try:
        if token:
            syn.login(authToken=token, silent=True)
            print("Synapse login successful with token")
        else:
            syn.login()
            print("Synapse login via .synapseConfig")
    except Exception as e:
        log_error(f"Synapse login failed: {e}")
        return False, None

    data_dir = Path("C:/DeCaf/data/fets_raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    syn_ids = ["syn28546456", "syn51514105", "syn25829067"]

    for syn_id in syn_ids:
        try:
            print(f"Trying Synapse ID: {syn_id}")
            entity = syn.get(syn_id, downloadLocation=str(data_dir))
            print(f"  Got entity: {entity.name}, type: {type(entity).__name__}")
            return True, syn, syn_id
        except Exception as e:
            print(f"  Failed {syn_id}: {e}")
            continue

    return False, syn, None


def download_synapse_children_recursive(syn, syn_id, dest_dir, depth=0, max_depth=4):
    """Recursively download NIfTI files from a Synapse container."""
    if depth > max_depth:
        return 0

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    try:
        children = list(syn.getChildren(syn_id))
        print(f"{'  '*depth}Container {syn_id}: {len(children)} children")

        for child in children:
            name = child.get("name", "")
            child_id = child.get("id", "")
            child_type = child.get("type", "")

            if "Folder" in child_type or "Project" in child_type:
                sub_dir = dest_dir / name
                count += download_synapse_children_recursive(syn, child_id, sub_dir, depth+1, max_depth)
            elif "File" in child_type:
                if name.endswith(".nii.gz") or name.endswith(".nii") or name.endswith(".csv"):
                    try:
                        out_path = dest_dir / name
                        if not out_path.exists():
                            print(f"{'  '*depth}  Downloading: {name}")
                            syn.get(child_id, downloadLocation=str(dest_dir))
                        else:
                            print(f"{'  '*depth}  Already exists: {name}")
                        count += 1
                    except Exception as e:
                        log_error(f"Failed to download {name}: {e}")
    except Exception as e:
        log_error(f"Failed to list children of {syn_id}: {e}")

    return count


def try_msd_fallback():
    """Try downloading Medical Segmentation Decathlon Task01_BrainTumour."""
    import tarfile

    data_dir = Path("C:/DeCaf/data")
    tar_path = data_dir / "Task01_BrainTumour.tar"

    # Check if already extracted
    if (data_dir / "Task01_BrainTumour").exists():
        print("MSD Task01 already extracted")
        return True

    url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
    print(f"Downloading MSD from {url} ...")
    print("  (This is ~1.8GB, may take a while)")

    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0 and count % 1000 == 0:
                pct = count * block_size / total_size * 100
                print(f"  {pct:.1f}% downloaded...", end="\r")

        urllib.request.urlretrieve(url, str(tar_path), reporthook=progress_hook)
        print("\nDownload complete. Extracting...")

        with tarfile.open(str(tar_path), "r") as tar:
            tar.extractall(str(data_dir))

        print("Extraction complete")
        return True
    except Exception as e:
        log_error(f"MSD download failed: {e}")
        return False


def download_partition_csv(syn=None):
    """Try to get the institutional partition CSV."""
    partition_urls = [
        "https://raw.githubusercontent.com/FeTS-AI/Challenge/main/Task_1/partitioning_1.csv",
        "https://raw.githubusercontent.com/FeTS-AI/Challenge/master/Task_1/partitioning_1.csv",
        "https://raw.githubusercontent.com/FETS-AI/Challenge/main/Task_1/FeTS2022_partitioning_1.csv",
        "https://raw.githubusercontent.com/FeTS-AI/Challenge/main/Task_1/FeTS_2022_partitioning_1.csv",
    ]

    out_path = Path("C:/DeCaf/data/partitioning_1.csv")

    for url in partition_urls:
        try:
            print(f"Trying partition CSV: {url}")
            urllib.request.urlretrieve(url, str(out_path))
            # Verify it's a valid CSV
            import pandas as pd
            df = pd.read_csv(out_path)
            if len(df) > 10:
                print(f"Partition CSV downloaded: {len(df)} rows, columns: {df.columns.tolist()}")
                return str(out_path)
        except Exception as e:
            print(f"  Failed: {e}")

    # Try Synapse
    if syn is not None:
        try:
            for syn_id in ["syn28546456", "syn51514105"]:
                try:
                    children = list(syn.getChildren(syn_id))
                    for child in children:
                        if "partition" in child.get("name","").lower():
                            syn.get(child["id"], downloadLocation="C:/DeCaf/data/")
                            return f"C:/DeCaf/data/{child['name']}"
                except:
                    pass
        except Exception as e:
            log_error(f"Synapse partition search failed: {e}")

    return None


def main():
    print("=" * 60)
    print("STEP 1: DATA DOWNLOAD")
    print("=" * 60)

    status = {
        "data_source": None,
        "partition_source": None,
        "data_dir": None,
        "partition_csv": None,
        "fallbacks": [],
    }

    # Try Synapse first
    synapse_ok, syn, working_syn_id = try_synapse_download()

    data_dir = None

    if synapse_ok and syn is not None:
        print(f"\nSynapse accessible. Recursively downloading from {working_syn_id}...")
        n = download_synapse_children_recursive(syn, working_syn_id, "C:/DeCaf/data/fets_raw")

        # Check if we got NIfTI files
        from glob import glob
        niis = glob("C:/DeCaf/data/fets_raw/**/*.nii.gz", recursive=True)
        print(f"Downloaded {n} items, found {len(niis)} NIfTI files")

        if len(niis) > 0:
            status["data_source"] = f"Synapse {working_syn_id}"
            status["data_dir"] = "C:/DeCaf/data/fets_raw"
            data_dir = "C:/DeCaf/data/fets_raw"
        else:
            print("No NIfTI files found from Synapse, trying MSD fallback...")
            status["fallbacks"].append("Synapse returned no NIfTI files")

    if data_dir is None:
        print("\nFalling back to MSD Task01_BrainTumour...")
        status["fallbacks"].append("Synapse failed — using MSD fallback")
        msd_ok = try_msd_fallback()
        if msd_ok:
            status["data_source"] = "MSD_Task01_BrainTumour"
            status["data_dir"] = "C:/DeCaf/data/Task01_BrainTumour"
            data_dir = "C:/DeCaf/data/Task01_BrainTumour"
        else:
            log_error("ALL data download methods failed!")
            sys.exit(1)

    # Get partition CSV
    print("\nDownloading partition CSV...")
    partition_csv = download_partition_csv(syn)

    if partition_csv:
        status["partition_source"] = "GitHub/Synapse"
        status["partition_csv"] = partition_csv
    else:
        status["fallbacks"].append("No real partition CSV — will use synthetic partitions")
        status["partition_source"] = "synthetic"
        status["partition_csv"] = None

    status["errors"] = ERRORS

    # Save status
    with open("C:/DeCaf/fed_crc_results/step1_status.json", "w") as f:
        json.dump(status, f, indent=2)

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Data source: {status['data_source']}")
    print(f"  Data dir: {status['data_dir']}")
    print(f"  Partition: {status['partition_source']}")
    print(f"  Partition CSV: {status['partition_csv']}")
    if status["fallbacks"]:
        print(f"  Fallbacks: {status['fallbacks']}")
    print("=" * 60)

    return status


if __name__ == "__main__":
    main()
