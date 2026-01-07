import torch
import torch.nn as nn

################################################ 
# L1 loss
def Loss_L1(x_gt, x_pred):
  return nn.L1Loss().__call__(x_gt, x_pred)

################################################ 
# L2 loss
def Loss_L2(x_gt, x_pred):
  return nn.MSELoss().__call__(x_gt, x_pred)

################################################ 
# Edge loss
def EdgeLoss(x_gt, yhat):
  """Laplacian edge detector and L1Loss"""
  batch_sz = x_gt.shape[0]
  num_ch = x_gt.shape[1]
  laplace_kernel = torch.tensor([[0.25,  0.5, 0.25],
                                 [0.50, -3.0, 0.50],
                                 [0.25,  0.5, 0.25]])
  laplace_kernel = laplace_kernel.repeat(batch_sz, num_ch, 1, 1)
  x_gt_lap = torch.nn.functional.conv2d(x_gt.to(torch.float32), laplace_kernel, padding=(1,1))
  yhat_lap = torch.nn.functional.conv2d(yhat.to(torch.float32), laplace_kernel, padding=(1,1))

  return torch.nn.L1Loss(reduction='mean')(x_gt_lap,yhat_lap)

################################################ 
# Log MSE
def Loss_MSLE(x_gt, x_pred):
    log_gt = torch.log(x_gt + 1e-20)
    log_pred = torch.log(x_pred + 1e-20)
    return nn.MSELoss().__call__(log_gt,log_pred)

################################################ 
# Charbonnier loss
def CharbonnierLoss(x_gt, yhat):
    """Charbonnier Loss (L1)"""
    eps=1e-9
    diff = x_gt - yhat
    return torch.sqrt((diff * diff) + eps).mean()

################################################ 
# TV loss 2D
def total_variation_loss_aniso(x_gt, x_pred):
    """TV Loss"""
    def high_pass_x_y(image):
        der_v = image[:, :, :, 1:] - image[:, :, :, :-1]
        der_h = image[:, :, 1:, :] - image[:, :, :-1, :]
        return der_v, der_h

    image = x_gt - x_pred
    delta_x, delta_y = high_pass_x_y(image)

    return torch.abs(delta_x).mean() + torch.abs(delta_y).mean()

################################################ 
# TV loss 3D
def total_variation_loss_aniso_3d(x_gt, x_pred):
    """TV Loss"""
    def high_pass_x_y(image):
        der_v = image[:, :, :, 1:] - image[:, :, :, :-1]
        der_h = image[:, :, 1:, :] - image[:, :, :-1, :]
        der_z = image[:, 1:, :, :] - image[:, :-1, :, :]
        return der_v, der_h, der_z

    image = x_gt - x_pred
    delta_x, delta_y, delta_z = high_pass_x_y(image)

    return torch.abs(delta_x).mean() + torch.abs(delta_y).mean() + 0.33*torch.abs(delta_z).mean()

##############################################
# Histogram loss 2D

def Hist_loss(x_gt, y_pred):

    B = x_gt.shape[0]
    n_channels = x_gt.shape[1]
    
    x_gt = torch.clamp(x_gt,0,1)
    y_pred = torch.clamp(y_pred,0,1)
    
    loss_ = 0
    for i in range(n_channels):
        x_hist = torch.histc(x_gt[:,i,:,:], bins=256, min=0, max=1)
        y_hist = torch.histc(y_pred[:,i,:,:], bins=256, min=0, max=1)
        
        loss_ += torch.mean(torch.log(torch.cosh(x_hist - y_hist + 1e-20)))
        
    return loss_

##############################################
# Histogram loss 3D

def Hist_loss(x_gt, y_pred):

    B = x_gt.shape[0]
    n_channels = x_gt.shape[1]
    n_slices = x_gt.shape[2]
    
    x_gt = torch.clamp(x_gt,0,1)
    y_pred = torch.clamp(y_pred,0,1)
    
    loss_ = 0
    for i in range(n_slices):
        x_hist = torch.histc(x_gt[:,:,i,:,:], bins=256, min=0, max=1)
        y_hist = torch.histc(y_pred[:,:,i,:,:], bins=256, min=0, max=1)
        
        loss_ += torch.mean(torch.log(torch.cosh(x_hist - y_hist + 1e-20)))
        
    return loss_
##############################################
