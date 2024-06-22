import os
import torch
import random
import time
import yaml
import numpy as np


project_name = os.path.basename(os.getcwd())


def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus]

    cuda = all(gpu >= 0 for gpu in gpus)

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(gpu) for gpu in gpus])
        # torch.backends.cudnn.benchmark = True # speed-up cudnn
        # torch.backends.cudnn.fastest = True # even more speed-up?
        hint('Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'])

    else:
        hint('Launching on CPU')

    return cuda


def make_reproducible(iscuda, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if iscuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # set True will make data load faster
        #   but, it will influence reproducible
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


def hint(msg):
    timestamp = f'{time.strftime("%m/%d %H:%M:%S", time.localtime(time.time()))}'
    print('\033[1m' + project_name + ' >> ' +
          timestamp + ' >> ' + '\033[0m' + msg)


def clean_summary(filesuammry):
    """
    remove keys from wandb.log()
    Args:
        filesuammry:

    Returns:

    """
    keys = [k for k in filesuammry.keys() if not k.startswith('_')]
    for k in keys:
        filesuammry.__delitem__(k)
    return filesuammry


def login():
    local_server = {
        'WANDB_BASE_URL': '',
        'WANDB_ENTITY': '',
        'WANDB_API_KEY': '',
    }
    for k, v in local_server.items():
        os.environ[k] = v

