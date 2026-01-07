# from torchmetrics.functional import ssim, psnr
from torchmetrics.functional.image import (
    structural_similarity_index_measure as ssim,
    peak_signal_noise_ratio as psnr,
)


def PSNR_fn(gt, pred):
  return psnr(gt,pred)

def SSIM_fn(gt, pred):
  return ssim(gt,pred)

