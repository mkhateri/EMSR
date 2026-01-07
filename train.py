import os
import torch
import yaml
import json

from utils.dataloader import DataLoader_cls
from model.EMSR_trainer import EMSR_Trainer 
from model import metric, loss
from utils import util

from configs.config_train import CONFIG



def save_config(config, save_dir):
    config_path = os.path.join(save_dir, "CONFIG.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"[i] Saved CONFIG to: {config_path}")


# =============================================================================
# DATA LOADER BUILDER
# =============================================================================

def build_dataloader(train_dir, test_dir, crop_flag, dcfg):
    """
    Builds one DataLoader dictionary.
    Logic stays identical to your original implementation.
    """
    return DataLoader_cls(
        train_dir,
        test_dir,
        dcfg["data_mode"],
        dcfg["task_mode"],
        dcfg["noise_level"],
        dcfg["WHC"],
        dcfg["img_format"],
        dcfg["interpolation"],
        dcfg["val_split"],
        dcfg["batch_size"],
        dcfg["num_workers"],
        dcfg["drop_last"],
        dcfg["pin_memory"],
        dcfg["random_seed"],
        dcfg["shuffle"],
        crop_flag,
    )._get_DS()


# =============================================================================
# RUN EMSR
# =============================================================================

def run_emsr():

    # -------------------------
    # System / Device
    # -------------------------
    dev = CONFIG["system"]["device"]

    if dev == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev if torch.cuda.is_available() else "cpu")

    print("device:", device)

    # -------------------------
    # Paths
    # -------------------------
    root_dir = os.getcwd()
    exp_name = CONFIG["experiment"]["name"]

    write_path = util.makedirs_fn(root_dir, exp_name)
    checkpoints_path = util.makedirs_fn(root_dir, exp_name, "checkpoints")

    # ---------------- save CONFIG -------------------
    save_config(CONFIG, write_path)
    # -------------------------------------------------

    # -------------------------
    # DataLoaders
    # -------------------------
    dcfg = CONFIG["data"]

    DL_emb = build_dataloader(
        dcfg["train_dir"],
        dcfg["test_dir"],
        dcfg["crop_emb"],
        dcfg,
    )

    ds_train_loader_dict = {
        "ds_train_loader_emb": DL_emb["train_loader"],
    }

    ds_valid_loader_dict = {
        "ds_valid_loader_emb": DL_emb["valid_loader"],
    }

    ds_test_loader_dict = {
        "ds_test_loader_emb": DL_emb["test_loader"],
    }

    # -------------------------
    # Trainer
    # -------------------------
    mcfg = CONFIG["model"]

    emsr_trainer = EMSR_Trainer(
        mcfg["lr"],
        mcfg["in_channels"],
        mcfg["out_channels"],
        device,
        mcfg["optimizer"],
        mcfg["update_lr"],
        CONFIG["metrics"],
        CONFIG["loss"],
        CONFIG["loss_weights"],
        mcfg["model_name"],
        write_path,
        checkpoints_path,
        CONFIG["params"],          
    )

    # -------------------------
    # Training
    # -------------------------
    emsr_trainer(
        ds_train_loader_dict,
        ds_valid_loader_dict,
        ds_test_loader_dict,
        CONFIG["experiment"]["epochs"],
    )

    print("Done!")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_emsr()

