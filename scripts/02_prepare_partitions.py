"""Step 2: Parse partition CSV and map subjects to sites. Handle MSD fallback with synthetic partitions."""
import os
import sys
import json
import glob
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

ERRORS = []

def log_error(msg):
    ERRORS.append(msg)
    print(f"[ERROR] {msg}", file=sys.stderr)


def find_subjects(data_dir, data_source):
    """Find all subjects and their file paths."""
    data_dir = Path(data_dir)
    subjects = []

    if "MSD" in data_source or "Task01" in str(data_dir):
        # MSD Task01_BrainTumour: single 4D NIfTI files (H, W, D, 4)
        # Channels: FLAIR=0, T1w=1, T1gd=2, T2w=3
        # Labels: 1=edema, 2=non-enhancing tumor, 3=enhancing; WT=(label>0)
        img_dir = data_dir / "imagesTr"
        lbl_dir = data_dir / "labelsTr"

        if not img_dir.exists():
            log_error(f"MSD imagesTr not found at {img_dir}")
            return []

        # Try reading from dataset.json first (most reliable)
        dataset_json = data_dir / "dataset.json"
        if dataset_json.exists():
            with open(dataset_json) as f:
                ds = json.load(f)
            for item in ds.get("training", []):
                img_path = str(data_dir / item["image"].lstrip("./"))
                lbl_path = str(data_dir / item["label"].lstrip("./"))
                subj_name = Path(img_path).stem.replace(".nii", "")
                # Skip macOS resource fork files
                if subj_name.startswith("._"):
                    continue
                img_real = Path(img_path)
                lbl_real = Path(lbl_path)
                if img_real.exists() and lbl_real.exists():
                    subjects.append({
                        "subject_id": subj_name,
                        "image": str(img_real),   # single 4D file
                        "label": str(lbl_real),
                        "data_format": "MSD_4D",
                    })
            print(f"Loaded {len(subjects)} subjects from dataset.json")
            return subjects

        # Fallback: glob for non-resource-fork NIfTI files
        all_imgs = sorted(f for f in img_dir.glob("*.nii.gz") if not f.name.startswith("._"))
        for img_file in all_imgs:
            subj_name = img_file.stem.replace(".nii", "")
            lbl_file = lbl_dir / img_file.name
            if lbl_file.exists():
                subjects.append({
                    "subject_id": subj_name,
                    "image": str(img_file),
                    "label": str(lbl_file),
                    "data_format": "MSD_4D",
                })

    else:
        # FeTS/BraTS format
        # Search for subject directories
        for subj_dir in sorted(data_dir.rglob("*")):
            if not subj_dir.is_dir():
                continue

            subj_name = subj_dir.name
            if not (re.match(r".*\d{3,}", subj_name)):
                continue

            # Try FeTS2022 naming
            t1 = subj_dir / f"{subj_name}_t1.nii.gz"
            t1ce = subj_dir / f"{subj_name}_t1ce.nii.gz"
            t2 = subj_dir / f"{subj_name}_t2.nii.gz"
            flair = subj_dir / f"{subj_name}_flair.nii.gz"
            seg = subj_dir / f"{subj_name}_seg.nii.gz"

            if not all(f.exists() for f in [t1, t1ce, t2, flair, seg]):
                # Try BraTS naming
                t1 = subj_dir / f"{subj_name}_t1.nii.gz"
                t1ce = subj_dir / f"{subj_name}_t1ce.nii.gz"
                # Try different patterns
                alt_flair = subj_dir / f"{subj_name}_T2flair.nii.gz"
                if alt_flair.exists():
                    flair = alt_flair
                alt_t2 = subj_dir / f"{subj_name}_T2.nii.gz"
                if alt_t2.exists():
                    t2 = alt_t2
                alt_seg = subj_dir / f"{subj_name}_truth.nii.gz"
                if alt_seg.exists():
                    seg = alt_seg

            # Check all 4 modalities + seg exist
            if all(f.exists() for f in [t1, t1ce, t2, flair, seg]):
                entry = {
                    "subject_id": subj_name,
                    "image": [str(t1), str(t1ce), str(t2), str(flair)],
                    "label": str(seg),
                    "data_format": "FeTS_BraTS",
                }
                subjects.append(entry)

    print(f"Found {len(subjects)} complete subjects in {data_dir}")
    return subjects


def load_partition_csv(csv_path, subjects):
    """Parse partition CSV and return subject_to_site mapping."""
    df = pd.read_csv(csv_path)
    print(f"Partition CSV columns: {df.columns.tolist()}")
    print(df.head(3).to_string())

    # Find site/partition column
    site_col = None
    for c in df.columns:
        if any(kw in c.lower() for kw in ["partition", "institution", "site", "center"]):
            site_col = c
            break

    if site_col is None:
        site_col = df.columns[-1]
        print(f"Guessing site column: {site_col}")

    # Find subject ID column
    subj_col = None
    for c in df.columns:
        if any(kw in c.lower() for kw in ["subject", "id", "train", "case"]):
            subj_col = c
            break

    if subj_col is None:
        subj_col = df.columns[0]
        print(f"Guessing subject column: {subj_col}")

    print(f"Using site_col='{site_col}', subj_col='{subj_col}'")

    # Map sites to integer IDs
    unique_sites = sorted(df[site_col].unique())
    site_to_id = {s: i for i, s in enumerate(unique_sites)}
    print(f"Found {len(unique_sites)} sites: {unique_sites[:5]}...")

    # Build subject -> site dict
    subject_to_site = {}
    for _, row in df.iterrows():
        subj = str(row[subj_col]).strip()
        subject_to_site[subj] = site_to_id[row[site_col]]

    # Match subjects to partition
    matched = 0
    for s in subjects:
        sid = s["subject_id"]
        if sid in subject_to_site:
            s["site_id"] = subject_to_site[sid]
            matched += 1
        else:
            # Try numeric suffix matching
            sid_nums = re.findall(r"\d+", sid)
            found = False
            for key in subject_to_site:
                key_nums = re.findall(r"\d+", key)
                if sid_nums and key_nums and sid_nums[-1] == key_nums[-1]:
                    s["site_id"] = subject_to_site[key]
                    matched += 1
                    found = True
                    break
            if not found:
                s["site_id"] = None

    print(f"Matched {matched}/{len(subjects)} subjects to partition")
    unmatched = sum(1 for s in subjects if s.get("site_id") is None)
    if unmatched > 0:
        log_error(f"{unmatched} subjects unmatched in partition CSV")

    return len(unique_sites), site_to_id


def create_synthetic_partitions(subjects, k=8):
    """Create synthetic site partitions from intensity stats + tumor size proxies."""
    import nibabel as nib

    print(f"Creating {k} synthetic partitions from intensity statistics...")
    features = []
    n_sample = min(len(subjects), 484)

    for i, s in enumerate(subjects[:n_sample]):
        try:
            img_source = s["image"]
            data_format = s.get("data_format", "")

            if data_format == "MSD_4D" or isinstance(img_source, str):
                # Single 4D file — load and take first channel (FLAIR)
                img_nib = nib.load(img_source)
                raw = img_nib.get_fdata(dtype=np.float32)
                if raw.ndim == 4:
                    data = raw[..., 0]  # FLAIR channel
                else:
                    data = raw
            else:
                # Multi-file: load first file (T1)
                img_nib = nib.load(img_source[0])
                data = img_nib.get_fdata(dtype=np.float32)

            brain_mask = data > np.percentile(data[data > 0], 10) if (data > 0).any() else data > 0
            if brain_mask.sum() > 0:
                brain_vals = data[brain_mask]
                feat = [
                    float(brain_vals.mean()),
                    float(brain_vals.std()),
                    float(np.percentile(brain_vals, 10)),
                    float(np.percentile(brain_vals, 90)),
                    float(brain_vals.max()),
                    float((data > brain_vals.mean() + 2 * brain_vals.std()).sum()),  # hyperintense voxels
                ]
            else:
                feat = [0.0] * 6
        except Exception as e:
            feat = [0.0] * 6

        features.append(feat)

        if (i + 1) % 100 == 0:
            print(f"  Extracted features for {i+1}/{n_sample} subjects")

    features = np.array(features, dtype=np.float32)

    # Check variance — if all features are zero, use positional assignment
    if features.std(0).max() < 1e-6:
        print("  All features identical — using uniform spacing assignment")
        labels = np.arange(len(subjects)) % k
    else:
        # Normalize and cluster
        features = (features - features.mean(0)) / (features.std(0) + 1e-8)
        from sklearn.cluster import KMeans
        import os
        os.environ["OMP_NUM_THREADS"] = "2"
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(features)

        # Check if we got at least k/2 distinct clusters
        n_distinct = len(set(labels))
        if n_distinct < k // 2:
            print(f"  Only {n_distinct} distinct clusters — falling back to sorted assignment")
            # Sort by first feature and divide evenly
            sorted_idx = np.argsort(features[:, 0])
            labels = np.zeros(len(subjects), dtype=int)
            for rank, idx in enumerate(sorted_idx):
                labels[idx] = int(rank * k / len(sorted_idx))

    for i, s in enumerate(subjects):
        s["site_id"] = int(labels[i])

    site_counts = Counter(labels.tolist())
    print(f"Synthetic partition counts: {dict(sorted(site_counts.items()))}")
    return k


def main():
    print("=" * 60)
    print("STEP 2: PARSE PARTITION & FIND SUBJECTS")
    print("=" * 60)

    # Load step 1 status
    with open("C:/DeCaf/fed_crc_results/step1_status.json") as f:
        step1 = json.load(f)

    data_dir = step1["data_dir"]
    partition_csv = step1["partition_csv"]
    data_source = step1["data_source"]

    fallbacks_used = list(step1.get("fallbacks", []))

    print(f"Data dir: {data_dir}")
    print(f"Partition CSV: {partition_csv}")

    # Find all subjects
    subjects = find_subjects(data_dir, data_source)

    if len(subjects) == 0:
        log_error("No complete subjects found!")
        sys.exit(1)

    # Get partitions
    n_sites = 0
    partition_source = "none"

    if partition_csv and Path(partition_csv).exists():
        try:
            n_sites, site_to_id = load_partition_csv(partition_csv, subjects)
            partition_source = "real_csv"
        except Exception as e:
            log_error(f"Partition CSV parsing failed: {e}")
            fallbacks_used.append(f"Partition CSV parse error: {e}")

    if partition_source == "none" or n_sites == 0:
        print("Using synthetic partitions...")
        fallbacks_used.append("Using synthetic K=8 partitions (no real CSV)")
        n_sites = create_synthetic_partitions(subjects, k=8)
        partition_source = "synthetic"

    # Handle None site_ids
    for s in subjects:
        if s.get("site_id") is None:
            s["site_id"] = -1  # unknown

    # Print stats
    site_counts = Counter(s["site_id"] for s in subjects)
    print(f"\nFinal partition: {n_sites} sites, {len(subjects)} subjects")
    for sid in sorted(site_counts.keys()):
        print(f"  Site {sid}: {site_counts[sid]} subjects")

    # Filter out unknown sites
    valid_subjects = [s for s in subjects if s["site_id"] >= 0]
    print(f"\nValid subjects (with site assignment): {len(valid_subjects)}")

    # Save
    status = {
        "subjects": subjects,
        "valid_subjects": valid_subjects,
        "n_sites": n_sites,
        "partition_source": partition_source,
        "data_source": data_source,
        "fallbacks": fallbacks_used,
        "errors": ERRORS,
    }

    with open("C:/DeCaf/fed_crc_results/step2_subjects.json", "w") as f:
        json.dump(status, f, indent=2)

    print("\nSTEP 2 COMPLETE")
    print(f"  Subjects: {len(valid_subjects)}")
    print(f"  Sites: {n_sites}")
    print(f"  Partition source: {partition_source}")

    return status


if __name__ == "__main__":
    main()
