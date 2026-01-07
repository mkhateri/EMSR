import abc
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Callable
import torch.nn.functional as F
import torchvision


class Trainer(abc.ABC):
    def __init__(self, ckpts_path: str, log_dir: str):
        """
        Base Trainer class.

        Args:
            ckpts_path (str): Directory to save checkpoints.
            log_dir     (str): Directory to store TensorBoard logs.
        """
        super().__init__()
        self.ckpts_path = ckpts_path
        self.log_dir = log_dir

    # -------------------------------------------------------------------------
    # Compile model + optimizer + settings from child trainer
    # -------------------------------------------------------------------------
    def compile(self, *args, **kwargs):
        """Configures the trainer for training (delegated to subclass)."""
        self._compile(*args, **kwargs)
        self.compiled = True

    # -------------------------------------------------------------------------
    # MAIN TRAIN LOOP
    # -------------------------------------------------------------------------
    def fit(
        self,
        train_data,
        validation_data,
        test_data,
        epochs: int = 100,
        validation_per_step: int = 1000,
        test_per_step: int = 5000,
        ckpts_per_step: int = 50,
        **kwargs
    ):

        if not getattr(self, "compiled", False):
            raise ValueError("Please call compile() before training!")

        self.writer = SummaryWriter(self.log_dir)

        # Gracefully handle missing test data
        if test_data is None:
            test_data = {}

        train_data_emb = train_data["ds_train_loader_emb"]
        steps = epochs * len(train_data_emb)
        decay = 0

        for epoch in range(epochs):
            for i, batch_emb_i in enumerate(train_data_emb):

                step_i = epoch * len(train_data_emb) + i
                #------------------------- EPOCH/STEP INFO ------------------
                epoch_idx = epoch
                epoch_print = epoch + 1
                steps_per_epoch = len(train_data_emb)
                epoch_step = i + 1
                #------------------------------------------------------------
                # Construct batch (identical embeddings for consistency)
                batch_i = {
                    "batch_emb1_i": batch_emb_i,
                    "batch_emb2_i": batch_emb_i,
                }

                # -------------------------  TRAIN  -------------------------
                loss_tot = self._get_train_one_batch(batch_i, step_i)

                # ------------------------- VALIDATION ----------------------
                if step_i > 0 and step_i % validation_per_step == 0:
                    summary = self._get_val(self.model, validation_data, step_i)

                    print()
                    print(f"{datetime.now()}  |  "
                        f"Epoch {epoch_print}/{epochs}  |  "
                        f"Global Step {step_i}/{steps}  |  "
                        f"Epoch Step {epoch_step}/{steps_per_epoch}")

                    print(f"TRAIN-> total:{loss_tot:2.6f}")
                    self._print_summary("VALIDATION", summary)


                # -------------------------  LR DECAY  ----------------------
                if self.is_update_lr and step_i > 0 and step_i % 50000 == 0:
                    decay += 1
                    self.optimizer.param_groups[0]["lr"] = self.lr * (0.5 ** decay)

                # --------------------------- TEST --------------------------
                test_loader = test_data.get("ds_test_loader_emb", None)

                if (
                    step_i > 0
                    and test_loader is not None
                    and len(test_loader) > 0
                    and step_i % test_per_step == 0
                ):
                    summary = self._get_test(self.model, test_data, step_i)
                    self._print_summary("TEST", summary)

                # ------------------------ CHECKPOINT -----------------------
                if step_i > 0 and step_i % ckpts_per_step == 0:
                    self._save_model(self.model, step_i, self.optimizer, loss_tot)
                    print(f"[i] Checkpoint stored: step {step_i}")

                # Update gradients
                self.optimizer.zero_grad()
                loss_tot.backward()
                self.optimizer.step()

        self.writer.close()

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------
    def _get_val(self, model, validation_data, step_i):
        model.eval()
        validation_data_emb = validation_data["ds_valid_loader_emb"]

        summary = {name: 0.0 for name in self.metrics}

        with torch.no_grad():
            for batch_emb_i in validation_data_emb:

                # Noise simulation
                noise_level = np.random.uniform(9.99, 10.0, size=(1,))
                wgn_y = (
                    batch_emb_i["y"]
                    + np.random.normal(0.0, noise_level / 255.0, size=batch_emb_i["y"].shape)
                )

                # Build GT & input
                data_GT = torch.cat([batch_emb_i["x"], batch_emb_i["x"]], dim=2)
                data_y = torch.cat([batch_emb_i["y"], wgn_y], dim=2)

                GT = data_GT.to(torch.float32).to(self.device)
                y = data_y.to(torch.float32).to(self.device)

                y_pred = model(y)

                # Split embeddings
                split_pos = GT.shape[2] // 2
                GT_emb1, GT_emb2 = GT[:, :, :split_pos, :], GT[:, :, split_pos:, :]
                y_emb1, y_emb2 = y_pred[:, :, :split_pos, :], y_pred[:, :, split_pos:, :]

                # Compute metrics
                for name, fn in self.metrics.items():
                    val1 = fn(GT_emb1, y_emb1)
                    val2 = fn(GT_emb2, y_emb2)
                    val3 = fn(y_emb1, y_emb2)

                    summary[name] += val3

                    self.writer.add_scalar(f"/valid/metric/emb1/{name}", val1, step_i)
                    self.writer.add_scalar(f"/valid/metric/emb2/{name}", val2, step_i)
                    self.writer.add_scalar(f"/valid/metric/emb3/{name}", val3, step_i)

        # Normalize
        for name in summary.keys():
            summary[name] /= len(validation_data_emb)

        return summary

    # -------------------------------------------------------------------------
    # TESTING
    # -------------------------------------------------------------------------
    def _get_test(self, model, test_data, step_i):
        summary = {name: 0.0 for name in self.metrics}
        test_data_emb = test_data["ds_test_loader_emb"]

        model.eval()
        with torch.no_grad():
            for batch_emb_i in test_data_emb:

                GT = batch_emb_i["x"].to(torch.float32).to(self.device)
                y = batch_emb_i["y"].to(torch.float32).to(self.device)

                y_pred = model(y)

                for name, fn in self.metrics.items():
                    val = fn(GT, y_pred)
                    summary[name] += val
                    self.writer.add_scalar(f"/test/metric/{name}", val, step_i)

        for name in summary:
            summary[name] /= len(test_data_emb)

        return summary

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    def _save_model(self, model, step_i, optimizer, loss_tot):
        path = f"{self.ckpts_path}/ckpts_step_{step_i}.pt"
        torch.save(
            {
                "step": step_i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_tot,
            },
            path,
        )

    def _print_summary(self, prefix, summary):
        print(f"{prefix.ljust(11)}-> ", end="")
        keys = list(summary.keys())

        for idx, key in enumerate(keys):
            end_char = " | " if idx < len(keys) - 1 else " "
            print(f"{key.ljust(4)}: {summary[key]:.4f}", end=end_char)

        print()


