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


def set_gpu_mode(mode):
    global use_gpu
    global device
    global gpu_id
    global distributed
    global dist_rank
    global world_size
    gpu_id = int(os.environ.get("SLURM_LOCALID", 0))
    dist_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    distributed = world_size > 1
    use_gpu = mode
    device = torch.device(f"cuda:{gpu_id}" if use_gpu else "cpu")
    torch.backends.cudnn.benchmark = True

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
    device = torch.device(f"cuda" if use_gpu else "cpu")
    torch.backends.cudnn.benchmark = True

