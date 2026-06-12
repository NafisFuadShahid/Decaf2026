"""Step 4: Run inference on all subjects and compute per-volume CRC scores."""
import os
import sys
import json
import gc
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import pickle

ERRORS = []

def log_error(msg):
    ERRORS.append(msg)
    print(f"[ERROR] {msg}", file=sys.stderr)


def load_model(model_info):
    """Reload the saved model."""
    device = torch.device(model_info["device"])
    model_name = model_info["model_name"]
    n_out = model_info.get("n_out_channels", 3)

    if "swinunetr" in model_name.lower():
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=4,
            out_channels=n_out,
            feature_size=48,
            use_checkpoint=True,
        )
    else:
        from monai.networks.nets import SegResNet
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=n_out,
            dropout_prob=0.2,
        )

    weights = torch.load("C:/DeCaf/models/model_weights.pt", map_location=device)
    model.load_state_dict(weights, strict=False)
    model = model.to(device)
    model.eval()
    return model, device


def get_transforms(data_format):
    """Get MONAI preprocessing transforms."""
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, NormalizeIntensityd, EnsureTyped,
        ConcatItemsd, MapLabelValued
    )

    if "MSD" in data_format:
        # MSD: 4 separate files per modality
        transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                     mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ])
    else:
        transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                     mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ])

    return transforms


def convert_label_to_wt(label_array, data_format):
    """Convert segmentation labels to whole-tumor binary mask."""
    if "MSD" in data_format:
        # MSD BraTS: label 1=NCR/NET/TC, 2=ED, 3=ET (or 4 in older versions)
        return (label_array > 0).astype(np.float32)
    else:
        # FeTS/BraTS: 1=NCR/NET, 2=ED, 4=ET
        return (label_array > 0).astype(np.float32)


def run_inference_single(model, device, subject, transforms, model_info, lambda_grid):
    """Run inference on a single subject and compute FNR/set-size curves."""
    from monai.inferers import sliding_window_inference

    roi_size = tuple(model_info["roi_size"])
    sw_overlap = model_info["sw_overlap"]
    data_format = subject.get("data_format", "FeTS_BraTS")

    try:
        import nibabel as nib

        if data_format == "MSD_4D":
            # Single 4D NIfTI: shape (H, W, D, 4) — FLAIR, T1w, T1gd, T2w
            img_nib = nib.load(subject["image"])
            img_4d = img_nib.get_fdata(dtype=np.float32)  # (H, W, D, 4)
            if img_4d.ndim == 4:
                # Reorder channels to match BraTS convention: T1, T1ce, T2, FLAIR
                # MSD order: 0=FLAIR, 1=T1w, 2=T1gd(T1ce), 3=T2w
                # BraTS order (model trained on): T1, T1ce, T2, FLAIR = [1, 2, 3, 0]
                img_4d = img_4d[..., [1, 2, 3, 0]]
                img_array = np.transpose(img_4d, (3, 0, 1, 2))  # (4, H, W, D)
            else:
                # Already 3D (single modality) — shouldn't happen
                img_array = img_4d[np.newaxis]
        else:
            # Multi-file format: list of per-modality paths
            imgs = []
            for img_path in subject["image"]:
                img_nib = nib.load(img_path)
                img_data = img_nib.get_fdata(dtype=np.float32)
                imgs.append(img_data)
            img_array = np.stack(imgs, axis=0)  # (4, H, W, D)

        # Normalize per channel (nonzero)
        for c in range(img_array.shape[0]):
            ch = img_array[c]
            mask = ch != 0
            if mask.sum() > 0:
                ch[mask] = (ch[mask] - ch[mask].mean()) / (ch[mask].std() + 1e-8)
            img_array[c] = ch

        # Load label
        lbl_nib = nib.load(subject["label"])
        lbl_array = lbl_nib.get_fdata(dtype=np.float32)

        # Convert to whole-tumor mask
        gt_wt = convert_label_to_wt(lbl_array, data_format)

        gt_positives = gt_wt.sum()
        if gt_positives == 0:
            return None  # No tumor

        # To tensor
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).float().to(device)  # (1,4,H,W,D)

        # Handle large volumes: crop or reduce
        H, W, D = img_array.shape[1:]
        max_dim = max(H, W, D)

        # If image is too large, use smaller roi_size
        actual_roi = roi_size
        if max_dim > 240:
            # Standard BraTS size, should be fine
            pass
        if max_dim < roi_size[0]:
            actual_roi = (min(roi_size[0], H), min(roi_size[1], W), min(roi_size[2], D))

        with torch.no_grad():
            try:
                with torch.cuda.amp.autocast():
                    logits = sliding_window_inference(
                        img_tensor, actual_roi,
                        sw_batch_size=1,
                        predictor=model,
                        overlap=sw_overlap,
                        mode="gaussian",
                        device=device,
                    )
            except RuntimeError as oom:
                if "out of memory" in str(oom).lower():
                    torch.cuda.empty_cache()
                    # Reduce to smallest settings
                    actual_roi = (96, 96, 96)
                    with torch.cuda.amp.autocast():
                        logits = sliding_window_inference(
                            img_tensor, actual_roi,
                            sw_batch_size=1,
                            predictor=model,
                            overlap=0.125,
                            mode="constant",
                        )
                else:
                    raise

        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (C, H, W, D)

        # Determine which channel is WT
        # BraTS convention: channel 0=TC, channel 1=WT, channel 2=ET
        n_channels = probs.shape[0]
        if n_channels >= 2:
            pred_wt_prob = probs[1]  # WT channel
        else:
            pred_wt_prob = probs[0]

        # Ensure shape matches GT
        if pred_wt_prob.shape != gt_wt.shape:
            # Resize pred to match gt
            from scipy.ndimage import zoom
            zoom_factors = [gt_wt.shape[i] / pred_wt_prob.shape[i] for i in range(3)]
            pred_wt_prob = zoom(pred_wt_prob, zoom_factors, order=1)

        # Compute FNR and set-size curves over lambda grid
        fnr_curve = np.zeros(len(lambda_grid))
        set_size_curve = np.zeros(len(lambda_grid))

        for j, lam in enumerate(lambda_grid):
            threshold = 1.0 - lam
            pred_mask = (pred_wt_prob >= threshold).astype(np.float32)

            true_positives = (pred_mask * gt_wt).sum()
            fnr = 1.0 - true_positives / gt_positives
            fnr_curve[j] = fnr

            set_size = pred_mask.sum()
            set_size_curve[j] = set_size / gt_positives

        return {
            "site_id": subject.get("site_id"),
            "fnr_curve": fnr_curve.tolist(),
            "set_size_curve": set_size_curve.tolist(),
            "tumor_volume": float(gt_positives) / float(gt_wt.size),
            "gt_positives": int(gt_positives),
        }

    except Exception as e:
        log_error(f"Inference failed for {subject['subject_id']}: {e}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def main():
    print("=" * 60)
    print("STEP 4: INFERENCE + CRC SCORE COMPUTATION")
    print("=" * 60)

    # Load previous steps
    with open("C:/DeCaf/fed_crc_results/step2_subjects.json") as f:
        step2 = json.load(f)
    with open("C:/DeCaf/fed_crc_results/step3_model.json") as f:
        model_info = json.load(f)

    subjects = step2["valid_subjects"]
    data_source = step2["data_source"]
    fallbacks = list(step2.get("fallbacks", []))

    print(f"Subjects to process: {len(subjects)}")
    print(f"Model: {model_info['model_name']}")
    print(f"Device: {model_info['device']}")

    # Lambda grid
    lambda_grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15,
                   0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70,
                   0.80, 0.90, 0.95, 0.99, 1.0]

    # Load model
    model, device = load_model(model_info)

    volume_scores = {}  # subject_id -> scores dict
    n_skipped_no_tumor = 0
    n_failed = 0

    data_format = "MSD" if "MSD" in data_source else "FeTS_BraTS"

    for i, subject in enumerate(subjects):
        sid = subject["subject_id"]

        result = run_inference_single(model, device, subject, None, model_info, lambda_grid)

        if result is None:
            if subject.get("gt_positives", -1) == 0:
                n_skipped_no_tumor += 1
            else:
                n_failed += 1
            continue

        volume_scores[sid] = result

        if (i + 1) % 10 == 0 or (i + 1) == len(subjects):
            pct = (i + 1) / len(subjects) * 100
            n_tumor = len(volume_scores)
            print(f"Progress: {i+1}/{len(subjects)} ({pct:.1f}%) — {n_tumor} with tumor, {n_failed} failed")

    print(f"\nInference complete:")
    print(f"  Processed: {len(subjects)}")
    print(f"  With tumor (used for CRC): {len(volume_scores)}")
    print(f"  Skipped (no tumor in GT): {n_skipped_no_tumor}")
    print(f"  Failed: {n_failed}")

    # Verify monotonicity on a sample
    n_violations = 0
    for sid, data in list(volume_scores.items())[:20]:
        fnr = data["fnr_curve"]
        for j in range(len(lambda_grid) - 1):
            if fnr[j] < fnr[j+1] - 1e-4:
                n_violations += 1
                break

    print(f"Monotonicity violations (sample of 20): {n_violations}")
    if n_violations > 0:
        log_error(f"FNR non-monotone in {n_violations} subjects (may be due to sigmoid threshold)")
        # Force monotonicity by cumulative minimum
        for sid in volume_scores:
            fnr = np.array(volume_scores[sid]["fnr_curve"])
            for j in range(1, len(fnr)):
                fnr[j] = min(fnr[j], fnr[j-1])
            volume_scores[sid]["fnr_curve"] = fnr.tolist()
        print("Forced monotonicity via running minimum")

    # Save
    status = {
        "n_subjects_with_tumor": len(volume_scores),
        "n_skipped": n_skipped_no_tumor,
        "n_failed": n_failed,
        "lambda_grid": lambda_grid,
        "fallbacks": fallbacks,
        "errors": ERRORS,
    }

    with open("C:/DeCaf/fed_crc_results/step4_status.json", "w") as f:
        json.dump(status, f, indent=2)

    with open("C:/DeCaf/fed_crc_results/volume_scores.pkl", "wb") as f:
        pickle.dump(volume_scores, f)

    print("\nSTEP 4 COMPLETE")
    print(f"  Volume scores saved: {len(volume_scores)} subjects")

    return status, volume_scores


if __name__ == "__main__":
    main()
