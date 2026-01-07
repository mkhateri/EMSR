from typing import Any, Callable, Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np

from model import trainer
from model import models as SuperResolver

# ============================================================================
#  HIGH-LEVEL TRAINER WRAPPER
# ============================================================================

class EMSR_Trainer(object):
    def __init__(self,
                 lr,
                 in_channels,
                 out_channels,
                 device,
                 optimizer_name,
                 is_update_lr,
                 metrics,
                 loss,
                 loss_weights,
                 model_name,
                 write_path,
                 checkpoints_path,
                 model_params):
        """
        Wrapper class around the Trainer module.
        Configures model, optimizer, metrics, losses, and IO paths.
        """
        super(EMSR_Trainer, self).__init__()

        self.device = device
        self.lr = lr
        self.is_update_lr = is_update_lr
        self.metrics = metrics
        self.optimizer_name = optimizer_name
        self.loss = loss
        self.loss_weights = loss_weights
        self.write_path = write_path
        self.checkpoints_path = checkpoints_path
        self.model_params = model_params

        # build model
        self.model = self._get_model(model_name, in_channels, out_channels, self.model_params)

    # ----------------------------------------------------------------------

    def __call__(self, ds_train_loader, ds_valid_loader, ds_test_loader, epochs):
        """Entry point for running the entire training."""
        
        print("[i] Configuring super-resolver...")

        trainer_mod = SuperResolver_Trainer_module(
            model=self.model,
            device=self.device,
            lr=self.lr,
            is_update_lr=self.is_update_lr,
            write_path=self.write_path,
            checkpoints_path=self.checkpoints_path
        )

        print("[i] Compiling super-resolver...")
        optimizer = self._get_optimizer(self.optimizer_name)

        trainer_mod.compile(
            optimizer=optimizer,
            loss=self.loss,
            loss_weights=self.loss_weights,
            metrics=self.metrics
        )

        print("[i] Fitting super-resolver...")
        trainer_mod.fit(
            train_data=ds_train_loader,
            validation_data=ds_valid_loader,
            test_data=ds_test_loader,
            epochs=epochs
        )

        return True

    # ----------------------------------------------------------------------
    def _get_model(self, model_name, in_channels, out_channels, model_params):
        """Fetch model class dynamically and instantiate it."""
        mod = getattr(SuperResolver, model_name)
        return mod(in_ch=in_channels, out_ch=out_channels, **model_params)

    # ----------------------------------------------------------------------

    def _get_optimizer(self, optimizer_name):
        """Build PyTorch optimizer."""
        if optimizer_name == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("Optimizer not implemented")


# ============================================================================
#  CORE TRAINER MODULE EXTENDING Base Trainer
# ============================================================================

class SuperResolver_Trainer_module(trainer.Trainer):
    def __init__(self, model, device, lr, is_update_lr, write_path, checkpoints_path):
        super(SuperResolver_Trainer_module, self).__init__(
            ckpts_path=checkpoints_path,
            log_dir=write_path
        )

        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.is_update_lr = is_update_lr

    # ----------------------------------------------------------------------

    def _compile(self, optimizer, loss, loss_weights, metrics=None):
        """Configure the model, optimizer, loss, and metrics."""
        self.optimizer = optimizer
        self.loss = loss
        self.loss_weights = _loss_weights(loss_weights, loss)
        self.metrics = metrics if metrics is not None else {}

    # ----------------------------------------------------------------------

    def _get_train_one_batch(self, data, step_i):
        """Compute forward → losses → TensorBoard logging for ONE batch."""

        self.model.train()

        # noise generation
        noise_level = np.random.uniform(0., 5., size=(1,))
        wgn_y = data['batch_emb2_i']['y'] + np.random.normal(
            0., noise_level / 255., size=data['batch_emb2_i']['y'].shape
        )

        # concat embeddings
        data_GT = torch.cat(
            [data['batch_emb1_i']['x'], data['batch_emb2_i']['x']], dim=2
        )
        data_y = torch.cat(
            [data['batch_emb1_i']['y'], wgn_y], dim=2
        )

        GT = data_GT.to(torch.float32).to(self.device)
        y = data_y.to(torch.float32).to(self.device)

        # forward
        y_pred = self.model(y)

        # split embeddings
        split = GT.shape[2] // 2
        y_emb1 = y_pred[:, :, :split, :]
        y_emb2 = y_pred[:, :, split:, :]

        GT_emb1 = GT[:, :, :split, :]
        GT_emb2 = GT[:, :, split:, :]

        # --------------------------------
        # LOSS COMPUTATION
        # --------------------------------
        loss_tot_emb1 = 0.
        loss_tot_emb2 = 0.
        loss_tot_emb3 = 0.

        for name, fn in self.loss.items():

            loss1 = fn(GT_emb1, y_emb1)
            loss2 = fn(GT_emb2, y_emb2)
            loss3 = fn(y_emb1, y_emb2)

            loss_tot_emb1 += loss1 * self.loss_weights[name]
            loss_tot_emb2 += loss2 * self.loss_weights[name]
            loss_tot_emb3 += loss3 * self.loss_weights[name]

            self.writer.add_scalar(f'/loss/emb1/{name}', loss1, step_i)
            self.writer.add_scalar(f'/loss/emb2/{name}', loss2, step_i)
            self.writer.add_scalar(f'/loss/emb3/{name}', loss3, step_i)

        # total loss
        loss_tot = loss_tot_emb1 + loss_tot_emb2 + loss_tot_emb3
        self.writer.add_scalar('/Loss_tot/train', loss_tot, step_i)

        return loss_tot

    # ----------------------------------------------------------------------

    def _rescale_gt_2d(self, _GT1):
        """Downsample GT for multi-scale supervision (2D)."""
        _GT2 = F.interpolate(
            _GT1, size=(_GT1.shape[-2] // 2, _GT1.shape[-1] // 2),
            mode='bilinear', align_corners=False
        ).clamp(0, 1)
        _GT4 = F.interpolate(
            _GT1, size=(_GT1.shape[-2] // 4, _GT1.shape[-1] // 4),
            mode='bilinear', align_corners=False
        ).clamp(0, 1)
        return _GT1, _GT2, _GT4

    def _rescale_gt_3d(self, _GT1):
        """Downsample GT for multi-scale supervision (3D)."""
        _GT2 = F.interpolate(
            _GT1,
            size=(_GT1.shape[-3] // 2, _GT1.shape[-2] // 2, _GT1.shape[-1] // 2),
            mode='area'
        ).clamp(0, 1)
        _GT4 = F.interpolate(
            _GT1,
            size=(_GT1.shape[-3] // 4, _GT1.shape[-2] // 4, _GT1.shape[-1] // 4),
            mode='area'
        ).clamp(0, 1)
        return _GT1, _GT2, _GT4


# ============================================================================
#  Loss Weight Helper
# ============================================================================

def _loss_weights(loss_weights: Union[None, Dict[str, float]],
                  loss: Dict[str, Any]) -> Dict[str, float]:
    """Ensures every loss has an assigned weight."""
    weights = {name: 1.0 for name in loss}
    if loss_weights is not None:
        weights.update(loss_weights)
    return weights

