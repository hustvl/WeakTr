import os
import torch
import random
import numpy as np
from torch import distributed as dist

"""
GPU wrappers
"""

use_gpu = False
gpu_id = 0
device = None

distributed = False
dist_rank = 0
world_size = 1

def set_gpu_dist_mode(mode):
    global use_gpu
    global device
    global gpu_id
    global distributed
    global dist_rank
    global world_size

    if dist.is_available() and dist.is_initialized():
        dist_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        dist_rank = 0
        world_size = 1

    distributed = world_size > 1
    use_gpu = mode

    device = dist_rank % torch.cuda.device_count()
    torch.backends.cudnn.benchmark = True
