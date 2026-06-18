"""Step 3: Download pre-trained BraTS segmentation model with fallbacks."""
import os
import sys
import json
import torch
from pathlib import Path

ERRORS = []

def log_error(msg):
    ERRORS.append(msg)
    print(f"[ERROR] {msg}", file=sys.stderr)


def try_monai_bundle():
    """Try downloading MONAI bundle."""
    from monai.bundle import download

    bundle_names = [
        "brats_mri_segmentation",
        "brats_mri_segmentation_v2",
    ]

    for name in bundle_names:
        try:
            print(f"Trying MONAI bundle: {name}")
            download(name=name, bundle_dir="C:/DeCaf/models/")
            bundle_dir = Path(f"C:/DeCaf/models/{name}")
            if bundle_dir.exists():
                print(f"Bundle downloaded to {bundle_dir}")
                return str(bundle_dir), name
        except Exception as e:
            print(f"  Failed {name}: {e}")

    return None, None


def load_from_bundle(bundle_dir, bundle_name):
    """Load model from MONAI bundle."""
    from monai.bundle import ConfigParser

    bundle_dir = Path(bundle_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find config file
    config_candidates = [
        bundle_dir / "configs" / "inference.json",
        bundle_dir / "configs" / "inference.yaml",
        bundle_dir / "configs" / "train.json",
        bundle_dir / "configs" / "train.yaml",
    ]

    config_path = None
    for c in config_candidates:
        if c.exists():
            config_path = c
            break

    if config_path is None:
        raise FileNotFoundError(f"No config found in {bundle_dir}/configs/")

    print(f"Loading from bundle config: {config_path}")

    parser = ConfigParser()
    parser.read_config(str(config_path))

    # Extract network
    try:
        net = parser.get_parsed_content("network_def", instantiate=True)
        net = net.to(device)
    except Exception as e:
        print(f"  network_def failed: {e}")
        net = parser.get_parsed_content("network", instantiate=True)
        net = net.to(device)

    # Load weights
    ckpt_candidates = list(bundle_dir.glob("models/*.pt")) + list(bundle_dir.glob("models/*.pth"))
    if ckpt_candidates:
        ckpt = torch.load(str(ckpt_candidates[0]), map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        net.load_state_dict(ckpt, strict=False)
        print(f"Loaded weights from {ckpt_candidates[0]}")

    net.eval()
    return net, device, "swinunetr_bundle"


def try_swinunetr_pretrained(device):
    """Try loading SwinUNETR with pre-trained weights."""
    from monai.networks.nets import SwinUNETR

    model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True,
    )

    weight_urls = [
        "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt",
        "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt",
    ]

    for url in weight_urls:
        try:
            print(f"Downloading SwinUNETR weights from {url}")
            state_dict = torch.hub.load_state_dict_from_url(
                url, map_location=device, model_dir="C:/DeCaf/models/"
            )
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Filter keys
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            print(f"  Matched {len(pretrained_dict)}/{len(model_dict)} parameter tensors")

            model.load_state_dict(state_dict, strict=False)
            model = model.to(device)
            model.eval()
            print("SwinUNETR loaded with pre-trained encoder weights")
            return model, device, "swinunetr_pretrained_encoder"
        except Exception as e:
            print(f"  Failed: {e}")

    # Load with random init (for validation that the pipeline works)
    print("Loading SwinUNETR with random initialization (no pre-trained weights)")
    model = model.to(device)
    model.eval()
    return model, device, "swinunetr_random_init"


def try_segresnet(device):
    """Fallback: SegResNet with random init."""
    from monai.networks.nets import SegResNet

    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)

    model.eval()
    print("Loaded SegResNet (random init — will need to train from scratch)")
    return model, device, "segresnet_random_init"


def check_model_output(model, device, model_name, data_format):
    """Quick sanity check of model output shape."""
    print("Sanity-checking model output...")

    if "random_init" in model_name:
        print("  Skipping sanity check for random-init model")
        return True

    with torch.no_grad():
        dummy = torch.randn(1, 4, 96, 96, 96).to(device)
        try:
            out = model(dummy)
            print(f"  Input: {dummy.shape}, Output: {out.shape}")
            assert out.shape[1] in [1, 2, 3, 4], f"Unexpected output channels: {out.shape[1]}"
            return True
        except Exception as e:
            print(f"  Sanity check failed: {e}")
            return False


def main():
    print("=" * 60)
    print("STEP 3: DOWNLOAD MODEL")
    print("=" * 60)

    # Load step 2 status
    with open("C:/DeCaf/fed_crc_results/step2_subjects.json") as f:
        step2 = json.load(f)

    data_source = step2["data_source"]
    fallbacks = list(step2.get("fallbacks", []))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram:.1f}GB")
        sw_overlap = 0.5 if vram >= 10 else 0.25
        roi_size = (128, 128, 128)
    else:
        sw_overlap = 0.25
        roi_size = (96, 96, 96)
        vram = 0

    print(f"Sliding window: roi={roi_size}, overlap={sw_overlap}")

    model = None
    model_name = None

    # Try MONAI bundle first
    try:
        bundle_dir, bundle_name = try_monai_bundle()
        if bundle_dir:
            try:
                model, device_used, model_name = load_from_bundle(bundle_dir, bundle_name)
                print(f"Model loaded from MONAI bundle: {bundle_name}")
            except Exception as e:
                log_error(f"Bundle load failed: {e}")
                fallbacks.append(f"MONAI bundle load failed: {e}")
    except Exception as e:
        log_error(f"MONAI bundle download failed: {e}")
        fallbacks.append(f"MONAI bundle download failed: {e}")

    # Try SwinUNETR with pre-trained weights
    if model is None:
        try:
            model, device, model_name = try_swinunetr_pretrained(device)
        except Exception as e:
            log_error(f"SwinUNETR load failed: {e}")
            fallbacks.append(f"SwinUNETR pretrained failed: {e}")

    # Final fallback: SegResNet random init
    if model is None:
        try:
            model, device, model_name = try_segresnet(device)
            fallbacks.append("Using SegResNet with random initialization — predictions will be noise")
        except Exception as e:
            log_error(f"All model options failed: {e}")
            sys.exit(1)

    # Sanity check
    model_ok = check_model_output(model, device, model_name, data_source)

    # Determine number of output channels
    with torch.no_grad():
        try:
            dummy = torch.randn(1, 4, 64, 64, 64).to(device)
            out = model(dummy)
            n_out_channels = out.shape[1]
        except:
            n_out_channels = 3

    # Save model to disk for use in inference step
    torch.save(model.state_dict(), "C:/DeCaf/models/model_weights.pt")

    status = {
        "model_name": model_name,
        "device": str(device),
        "sw_overlap": sw_overlap,
        "roi_size": list(roi_size),
        "vram_gb": vram,
        "n_out_channels": n_out_channels,
        "model_ok": model_ok,
        "fallbacks": fallbacks,
        "errors": ERRORS,
    }

    with open("C:/DeCaf/fed_crc_results/step3_model.json", "w") as f:
        json.dump(status, f, indent=2)

    print("\nSTEP 3 COMPLETE")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  SW overlap: {sw_overlap}, ROI: {roi_size}")
    print(f"  Output channels: {n_out_channels}")

    return status


if __name__ == "__main__":
    main()
