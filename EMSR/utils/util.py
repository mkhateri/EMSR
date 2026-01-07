from typing import Callable, Tuple, Dict, Union
import os
import json
import string
import shutil
from PIL import Image
import numpy as np 
import re



def makedirs_fn(*argv):
    path_ = [arg for arg in argv if arg]
    path_ = os.path.join(*path_)
    if not os.path.exists(path_):
        os.makedirs(path_)
        #tf.gfile.MakeDirs(path_)
    return path_