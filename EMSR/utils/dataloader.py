"""
This script reads data balah balh
@author: mohammad
"""

import os
import torch
import torch.nn as nn
from torch.utils.data  import Dataset, DataLoader
import numpy as np
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Any, Callable, Dict, Tuple, Union 
import re

from utils.transform_collections import Transform_sr



def _get_dir(_dir:Callable[[Union[list,str]],str])->list:
    if _dir is None:
        return []

    if isinstance(_dir, str):
        _dir = [_dir]

    dirs = []
    for item in _dir:
        dirs.append("".join(c for c in item if c not in r'[,*?!:"<>|] \\'))
    return dirs



def _initilize_data_mode(data_mode:Dict[str,bool])->Dict[str,Any]:
    """ Ininitlize data mode by the input data mode"""

    data_mode_ = {key: False for key in ['y', 'x', 'mask']} 
    if data_mode is not None:
        data_mode_.update(data_mode)
    
    return data_mode_

def _get_files(data_dirs, img_format, data_mode):
    """ Description... 
        in-args:
        out-args: ...
    """
    data = _initilize_data_mode(data_mode)

    #
    data_dirs = [os.path.join(dir_,'HR') for dir_ in data_dirs] 

    data['x'] = [os.path.join(path, name)
                    for data_dir in data_dirs 
                        for path, subdirs, files_ in os.walk(data_dir)  
                            for name in files_
                                if name.lower().endswith(tuple(img_format))]
    if data_mode['y']:
        data['y'] =  [files.replace('/HR/', '/LR/') for files in data['x']]
    if data_mode['mask']:
        data['mask'] =  [files.replace('/HR/', '/mask/') for files in data['x']]

    return data


def _im_read_resize(PATH, WHC, interpolation, crop=False):
    #imread and imresize
    assert isinstance(WHC, tuple) and len(WHC)==3, "Invalid tuple for width, height, channel"
    Width, Height, Channel = WHC[0], WHC[1], WHC[2]

    if Channel ==1:
        if not crop:
            x =  np.array(Image.open(PATH).convert('L').resize(size=(Width, Height), resample=interpolation))
            return x[:,:,np.newaxis]
        else:
            x =  np.array(Image.open(PATH).convert('L'))
            return x[0:Width,0:Height,np.newaxis]
        
    elif Channel==3:
        if not crop:
            x =  np.array(Image.open(PATH).resize(size=(Width, Height), resample=interpolation))
            return x
        else:
            x = np.array(Image.open(PATH))
        return x[0:Width,0:Height,:]
    else:
        raise NameError('Invalid channel number')



def _im_read_resize2(PATH, WHC, interpolation, crop=False, mode='H'):
    #imread and imresize
    assert isinstance(WHC, tuple) and len(WHC)==3, "Invalid tuple for width, height, channel"
    Width, Height, Channel = WHC[0], WHC[1], WHC[2]

    #if Channel==1 and crop==False:
    #    x =  np.array(Image.open(PATH).convert('L').resize(size=(1024, 1024), resample=interpolation))
    #    return x[:,:,np.newaxis]
    if mode=='H':
        if Channel==1:
            x =  np.array(Image.open(PATH).convert('L').resize(size=(1023, 1023), resample=interpolation))
            return x[:,:,np.newaxis]  #x[0:Width,0:Height,np.newaxis]
        else:
            raise NameError('Invalid channel number')
            
    elif mode=='L':
        if Channel==1:
            x =  np.array(Image.open(PATH).convert('L').resize(size=(Width, Height), resample=interpolation))
            return x[:,:,np.newaxis]  #x[0:Width,0:Height,np.newaxis]
        else:
            raise NameError('Invalid channel number')
    else:
        raise NameError('Invalid imresize mode')


class Dataset_cls(Dataset):
    def __init__(self,
                data_dirs: str,
                Training: bool,
                data_mode:Dict[str, bool],
                task_mode:str,
                noise_level: Union[list,float],
                WHC:list,
                img_format:list,
                interpolation: str,
                crop: bool
                ):
        super(Dataset, self).__init__()

        self.data_mode = _initilize_data_mode(data_mode)
        self.task_mode = task_mode
        self.noise_level = noise_level
        self.WHC = WHC
        self.interpolation = interpolation or Image.BICUBIC
        self.Training = Training
        self.files_ = _get_files(data_dirs, img_format, self.data_mode)
        self.crop = crop

    def __len__(self):
        return len(self.files_['x'])


    def __getitem__(self, index):
        # dict_files_dirs = self._get_files() mor: we dont have enough time to re-read data
        sample_index = _initilize_data_mode(self.data_mode)
        
        if self.data_mode['x']:
            sample_index['x'] = _im_read_resize2(self.files_['x'][index], self.WHC, self.interpolation, self.crop, mode='H')
        else:
            raise NameError('There is no HR directory') 
        
        if self.data_mode['y']:
            sample_index['y'] = _im_read_resize2(self.files_['y'][index], self.WHC, self.interpolation, self.crop, mode='L')
        
        if self.data_mode['mask']:
            sample_index['mask'] = _im_read_resize2(self.files_['mask'][index], self.WHC, self.interpolation, self.crop, mode='L')
      
        if self.task_mode =='SR':
            T = Transform_sr(self.noise_level,self.Training)
            return T(sample_index) 
            
 
 
class DataLoader_cls(Dataset): 
    def __init__(self,
                train_dir: Union[str,list],
                test_dir: Union[str,list],
                data_mode,
                task_mode,
                noise_level,
                WHC,
                img_format,
                interpolation,
                val_split,
                batch_size,
                num_workers,
                drop_last,
                pin_memory,
                random_seed,
                shuffle, 
                crop
                ): 
        
        self.train_dir= _get_dir(train_dir) 
        self.test_dir = _get_dir(test_dir)
        self.val_split  = val_split
        self.random_seed= random_seed
        self.shuffle = shuffle 
        self.batch_size = batch_size
        
        self.DL_params = {'num_workers':num_workers,
                        'pin_memory': pin_memory,
                        'drop_last':  drop_last
                    }
        
        self.DS_params = {'data_mode': data_mode, 
                        'task_mode': task_mode, 
                        'noise_level': noise_level, 
                        'WHC': WHC, 
                        'img_format': img_format, 
                        'interpolation': interpolation,
                        'crop': crop
                    }



    def _get_DS(self):
        # get train (& validation) and test datasets
        Dataset_Train = Dataset_cls(data_dirs=self.train_dir, Training=True, **self.DS_params)
        Dataset_Test  = Dataset_cls(data_dirs=self.test_dir,  Training=False, **self.DS_params)

        # some description here
        len_train_valid = Dataset_Train.__len__()
        train_sampler, valid_sampler = self._train_valid_sampler(len_train_valid)

        # comment here?
        train_loader = torch.utils.data.DataLoader(dataset= Dataset_Train, sampler= train_sampler, batch_size = self.batch_size, **self.DL_params )
        valid_loader = torch.utils.data.DataLoader(dataset= Dataset_Train, sampler= valid_sampler, batch_size = self.batch_size, **self.DL_params )
        test_loader  = torch.utils.data.DataLoader(dataset= Dataset_Test,  shuffle= False, batch_size=1, **self.DL_params)

        return {'train_loader': train_loader, 
                'valid_loader': valid_loader, 
                'test_loader': test_loader,
            }



    def _train_valid_sampler(self, len_train_valid):
        indices = list(range(len_train_valid))
        split = int(np.floor(self.val_split * len_train_valid))

        if self.shuffle:
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        return train_sampler, valid_sampler


