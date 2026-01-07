from torchvision import transforms
import numpy as np
import torch
from typing import Any, Callable, Dict, Tuple, Union


# -------------------------------------------------------------------------
# Basic Normalization
# -------------------------------------------------------------------------
class Normalization(object):
    def __call__(self, sample):
        for key, val in sample.items():
            if val is not False:
                sample[key] = val / 255.0
        return sample


# -------------------------------------------------------------------------
# Noise for Denoising
# -------------------------------------------------------------------------
class AddGaussianNoise(object):
    def __init__(self, noise_level: Union[list, float], Training: bool):
        for n in noise_level:
            assert 0.0 <= n <= 255.0, "Enter valid noise level"

        self.noise_level = (
            np.random.uniform(noise_level[0], noise_level[1], size=(1,))
            if Training
            else noise_level[-1]
        )

    def __call__(self, sample):
        wgn = np.random.normal(0.0, self.noise_level / 255.0, size=sample["x"].shape)
        sample["y"] = sample["x"] + wgn
        return sample


# -------------------------------------------------------------------------
# Noise for Super-Resolution
# -------------------------------------------------------------------------
class AddGaussianNoise_SR(object):
    def __init__(self, noise_level: Union[list, float], Training: bool):
        for n in noise_level:
            assert 0.0 <= n <= 255.0, "Enter valid noise level"

        self.noise_level = (
            np.random.uniform(noise_level[0], noise_level[1], size=(1,))
            if Training
            else noise_level[-1]
        )

    def __call__(self, sample):
        wgn = np.random.normal(0.0, self.noise_level / 255.0, size=sample["y"].shape)
        sample["y"] = sample["y"] + wgn
        return sample


# -------------------------------------------------------------------------
# Tensor Conversion
# -------------------------------------------------------------------------
class ToTensor(object):
    def __call__(self, sample):
        for key, val in sample.items():
            if val is not False:
                sample[key] = torch.from_numpy(val)
        return sample


# -------------------------------------------------------------------------
# Random 90° Rotation
# -------------------------------------------------------------------------
class ImRotate90(object):
    """
    Randomly rotate images by 90°, 180°, or 270° with probability p.
    """

    def __init__(self, p: float):
        assert 0.0 <= p <= 1.0, "p must be between 0 and 1"
        self.p = p

    def __call__(self, sample):
        if np.random.rand() <= self.p:
            k = np.random.choice([1, 2, 3])  # no 0 -> rotation must happen
            for key, val in sample.items():
                if val is not False:
                    sample[key] = np.rot90(val, k).copy()
        return sample


# -------------------------------------------------------------------------
# Left-Right Flip
# -------------------------------------------------------------------------
class ImFlip_lr(object):
    def __init__(self, p: float):
        assert 0.0 <= p <= 1.0, "p must be between 0 and 1"
        self.p = p

    def __call__(self, sample):
        if np.random.rand() <= self.p:
            for key, val in sample.items():
                if val is not False:
                    sample[key] = np.fliplr(val).copy()
        return sample


# -------------------------------------------------------------------------
# Up-Down Flip
# -------------------------------------------------------------------------
class ImFlip_ud(object):
    def __init__(self, p: float):
        assert 0.0 <= p <= 1.0, "p must be between 0 and 1"
        self.p = p

    def __call__(self, sample):
        if np.random.rand() <= self.p:
            for key, val in sample.items():
                if val is not False:
                    sample[key] = np.flipud(val).copy()
        return sample


# -------------------------------------------------------------------------
# Channel Transpose (H,W,C → C,H,W)
# -------------------------------------------------------------------------
class Channel_transpose(object):
    def __init__(self, transpose_tuple):
        assert isinstance(transpose_tuple, tuple) and len(transpose_tuple) == 3, \
            "Invalid transpose tuple"
        self.transpose_tuple = transpose_tuple

    def __call__(self, sample):
        for key, val in sample.items():
            if val is not False:
                sample[key] = np.transpose(val, self.transpose_tuple)
        return sample


# -------------------------------------------------------------------------
# Final SR Transform Pipeline
# -------------------------------------------------------------------------
def Transform_sr(
    noise_level: Union[list, float],
    Training: bool,
    transpose_tuple=(2, 0, 1),
    p=0.5
):
    return transforms.Compose([
        Normalization(),
        AddGaussianNoise_SR(noise_level, Training),
        ImRotate90(p),
        ImFlip_lr(p),
        ImFlip_ud(p),
        Channel_transpose(transpose_tuple),
        ToTensor(),
    ])



# from torchvision import transforms
# import numpy as np
# import torch
# from typing import Any, Callable, Dict, Tuple, Union 



# class Normalization(object):
       
#     def __call__(self, sample):
              
#         for key in sample.keys():
#             if sample[key] is not False:
#                 sample[key] = sample[key]/255.
        
#         return sample

# class AddGaussianNoise(object):
#     def __init__(self, noise_level:Union[list,float], Training:bool):
#         for noise_ in noise_level:
#             assert noise_ >= 0. and noise_ <= 255., 'Enter valid noise level!'

#         if Training:
#             self.noise_level = np.random.uniform(low=noise_level[0], high=noise_level[1], size=(1,))   
#         else:
#             self.noise_level = noise_level[-1] # get the maximum noise value for test dataset

#     def __call__(self, sample):
#         wgn = np.random.normal(0., self.noise_level/255., size=(sample['x'].shape))
#         sample['y'] = sample['x'] +  wgn
#         return sample

# class AddGaussianNoise_SR(object):
#     def __init__(self, noise_level:Union[list,float], Training:bool):
#         for noise_ in noise_level:
#             assert noise_ >= 0. and noise_ <= 255., 'Enter valid noise level!'

#         if Training:
#             self.noise_level = np.random.uniform(low=noise_level[0], high=noise_level[1], size=(1,))   
#         else:
#             self.noise_level = noise_level[-1] # get the maximum noise value for test dataset

#     def __call__(self, sample):
#         wgn = np.random.normal(0., self.noise_level/255., size=(sample['y'].shape))
#         sample['y'] = sample['y'] +  wgn
#         return sample
    
# class ToTensor(object):
    
#     def __call__(self, sample):
        
#         for key in sample.keys():
#             if sample[key] is not False:
#                 sample[key] = torch.from_numpy(sample[key])
                
#         return sample


# class ImRotate90(object):
#     """
#     Randomly rotate image by 0, 90, 180, or 270 degrees with probability p.
#     """

#     def __init__(self, p: float):
#         assert 0.0 <= p <= 1.0, "p must be between 0 and 1"
#         self.p = p

#     def __call__(self, sample):
#         if np.random.rand() <= self.p:

#             # choose rotation: 0, 1, 2, or 3 times 90 degrees
#             k = np.random.choice([1, 2, 3])  # do NOT include 0, since rotating by 0 is useless

#             for key in sample.keys():
#                 if sample[key] is not False:
#                     sample[key] = np.rot90(sample[key], k).copy()

#         return sample




# class ImFlip_lr(object):

#     def __init__(self,p):
#         self.p = p
        
#     def __call__(self, sample):
        
#         assert 1>=self.p>=0, 'p is limited in [0 1]'
#         p = self.p*100
#         if np.random.randint(100, size=1).item()<= p:
#             for key in sample.keys():
#                 if sample[key] is not False:
#                     sample[key] = np.fliplr(sample[key]).copy() 
#         return sample
          


# class ImFlip_ud(object):

#     def __init__(self,p):
#         self.p = p
        
#     def __call__(self, sample):
        
#         assert 1>=self.p>=0, 'p is limited in [0 1]'
#         p = self.p*100
        
#         if np.random.randint(100, size=1).item()<= p:
#             for key in sample.keys():
#                 if sample[key] is not False:
#                     sample[key] = np.flipud(sample[key]).copy() 
#         return sample


# class Channel_transpose(object):
    
#     def __init__(self, transpose_tuple):
               
#         assert isinstance(transpose_tuple, tuple) and len(transpose_tuple)==3, "Invalid transposed tuple" 
#         self.transpose_tuple = transpose_tuple

#     def __call__(self, sample):
#         for key in sample.keys():
#             if sample[key] is not False:
#                 sample[key] = np.transpose(sample[key], self.transpose_tuple) 
                
#         return sample
  

# def Transform_sr(
#     noise_level: Union[list, float],
#     Training: bool,
#     transpose_tuple=(2, 0, 1),
#     p=0.5
# ):
#     return transforms.Compose([
#         Normalization(),
#         AddGaussianNoise_SR(noise_level, Training),
#         ImRotate90(p),
#         ImFlip_lr(p),
#         ImFlip_ud(p),
#         Channel_transpose(transpose_tuple),
#         ToTensor(),
#     ])
