# ======================================================================
# Training Configuration 
# ======================================================================

from PIL import Image
from model import metric, loss

CONFIG = {

    # -------------------------
    # System / Device
    # -------------------------
    "system": {
        "device": "cuda",      # "cuda", "cpu", or "auto"
    },

    # -------------------------
    # Experiment
    # -------------------------
    "experiment": {
        "name": "logs",
        "epochs": 200,
    },
    
    # -------------------------
    # Data
    # -------------------------
    "data": {
        "train_dir":
            "/research/work/mkhateri/FULL_EM_DATA_TRAIN_TEST/Set3/X3/Train",

        "test_dir":  None,

        "data_mode": {"y": True, "x": True, "mask": False},
        "task_mode": "SR",

        "WHC": (341, 341, 1),
        "noise_level": [0.0, 5.0],

        "img_format": ["png", "jpg", "jpeg"],
        "interpolation": Image.BICUBIC,

        "val_split": 0.01,
        "batch_size": 2,
        "num_workers": 1,
        "drop_last": True,
        "pin_memory": True,
        "shuffle": True,
        "random_seed": 0,

        "crop_emb": False,
    },

    # -------------------------
    # Model / Optimizer
    # -------------------------
    "model": {
        "lr": 1e-4,
        "in_channels": 1,
        "out_channels": 1,
        "model_name": "SRNET",
        "optimizer": "Adam",
        "update_lr": True,
    },

    # -------------------------
    # Model Parameters
    # -------------------------
    "params": {
        "n_feat": 64,
        "n_blocks": 3,
        "levels": 3,
        "upscale_ratio_x": 3,
        "upscale_ratio_y": 3,
        "input_size": (341, 341),
        "num_heads": 16,
        "patch_size": 4,
        "mlp_ratio": 2,
    },

    # -------------------------
    # Metrics & Loss
    # -------------------------
    "metrics": {
        "PSNR": metric.PSNR_fn,
        "SSIM": metric.SSIM_fn,
    },

    "loss": {
        "MAE": loss.Loss_L1,
    },

    "loss_weights": {
        "MAE": 1,
    },
}
