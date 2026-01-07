import os
import yaml
import gc
import h5py
import torch
import numpy as np

# Import modules needed for YAML !!python/name
import model.loss
import model.metric

from configs.config_inference import INFERENCE_CONFIG
from model import models as SuperResolver


# -------------------------------------------------------
# Load CONFIG.yaml from logs directory
# -------------------------------------------------------
def load_training_config(logs_dir):
    config_path = os.path.join(logs_dir, "CONFIG.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"CONFIG.yaml not found in: {logs_dir}")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(f"Loaded training config: {config_path}")
    return config


# -------------------------------------------------------
# Load model with full params (in_ch, out_ch, **params)
# -------------------------------------------------------
def load_model(checkpoint_path, training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = training_config["model"]["model_name"]
    in_ch = training_config["model"]["in_channels"]
    out_ch = training_config["model"]["out_channels"]
    model_params = training_config["params"]  # full dictionary of SRNET args

    ModelClass = getattr(SuperResolver, model_name)
    model = ModelClass(in_ch=in_ch, out_ch=out_ch, **model_params).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded model `{model_name}` with checkpoint: {checkpoint_path}")
    return model, device


# -------------------------------------------------------
# Core SR function using patches (memory safe)
# -------------------------------------------------------
def super_resolve_volume(
        volume,
        model,
        device,
        output_file,
        patch_size,
        stride,
        upscale_factor
    ):
    # MATLAB is (X,Y,Z). PyTorch uses (H,W,Z).
    # Our training uses (H,W), so reorder to (Y,X,Z).
    volume = volume.transpose(2, 1, 0)

    H, W, Z = volume.shape
    SR_H, SR_W = H * upscale_factor, W * upscale_factor

    print(f"Volume shape: {volume.shape} → SR: {(SR_H, SR_W, Z)}")

    with h5py.File(output_file, "w") as f:
        dset = f.create_dataset(
            "super_resolved",
            (SR_H, SR_W, Z),
            dtype="uint8",
            compression="gzip",
            chunks=True
        )

        for z in range(Z):
            print(f"➡ Processing slice {z+1}/{Z}...")
            sl = volume[:, :, z].astype(np.float32) / 255.0

            sr_slice = np.zeros((SR_H, SR_W), dtype=np.float32)
            count = np.zeros_like(sr_slice)

            # Compute sliding grid positions
            xs = list(range(0, H - patch_size + 1, stride))
            ys = list(range(0, W - patch_size + 1, stride))

            if (H - patch_size) % stride != 0:
                xs.append(H - patch_size)
            if (W - patch_size) % stride != 0:
                ys.append(W - patch_size)

            for i in xs:
                for j in ys:
                    patch = sl[i:i + patch_size, j:j + patch_size]
                    patch_t = torch.from_numpy(patch)[None, None].float().to(device)

                    # Forward pass
                    with torch.no_grad():
                        sr_patch = model(patch_t).cpu().numpy().squeeze()

                    # Paste location in SR image
                    x0 = i * upscale_factor
                    y0 = j * upscale_factor
                    x1 = x0 + sr_patch.shape[0]
                    y1 = y0 + sr_patch.shape[1]

                    sr_slice[x0:x1, y0:y1] += sr_patch
                    count[x0:x1, y0:y1] += 1

            # Normalize overlapping regions
            count[count == 0] = 1
            sr_slice /= count

            # Convert to uint8 and write directly to file
            dset[:, :, z] = (sr_slice * 255.0).astype(np.uint8)

            torch.cuda.empty_cache()
            gc.collect()

    print(f" Saved SR volume → {output_file}")


# -------------------------------------------------------
# Process all MAT files in directory
# -------------------------------------------------------
def process_mat_directory(mat_dir, logs_dir, output_dir, ckpt_path=None):
    os.makedirs(output_dir, exist_ok=True)

    # Load training config
    train_cfg = load_training_config(logs_dir)

    # Load model checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(logs_dir, "checkpoints", "last_checkpoint.pt")

    model, device = load_model(ckpt_path, train_cfg)

    # List mat files
    mat_files = [f for f in os.listdir(mat_dir) if f.endswith(".mat")]
    if not mat_files:
        print("No .mat files found.")
        return

    for fname in mat_files:
        path = os.path.join(mat_dir, fname)
        print(f" Reading {path}")

        with h5py.File(path, "r") as f:
            key = INFERENCE_CONFIG["mat_dataset_key"] or list(f.keys())[0]
            volume = np.array(f[key], dtype=np.float32)

        out_file = os.path.join(output_dir, fname.replace(".mat", "_SR.h5"))

        super_resolve_volume(
            volume,
            model,
            device,
            out_file,
            patch_size=INFERENCE_CONFIG["patch_size"],
            stride=INFERENCE_CONFIG["stride"],
            upscale_factor=INFERENCE_CONFIG["upscale_factor"]
        )

        del volume
        gc.collect()
        torch.cuda.empty_cache()

    print(" All MAT files processed.")


# -------------------------------------------------------
# Command-line entry
# -------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", required=True)
    parser.add_argument("--mat_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt", default=None)

    args = parser.parse_args()

    process_mat_directory(
        mat_dir=args.mat_dir,
        logs_dir=args.logs_dir,
        output_dir=args.output_dir,
        ckpt_path=args.ckpt
    )




