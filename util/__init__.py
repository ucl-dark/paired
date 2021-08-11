import glob
import os
import shutil
import collections
import timeit
import random
import shutil

import numpy as np
import torch

from envs.registration import make as gym_make
from .make_agent import make_agent
from .filewriter import FileWriter
from envs.wrappers import ParallelAdversarialVecEnv, VecMonitor, \
    VecNormalize, VecPreprocessImageWrapper, TimeLimit


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self

def cprint(condition, *args, **kwargs):
    if condition:
        print(*args, **kwargs)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def safe_checkpoint(state_dict, path, index=None, archive_interval=None):
    filename, ext = os.path.splitext(path)
    path_tmp = f'{filename}_tmp{ext}'
    torch.save(state_dict, path_tmp)

    os.replace(path_tmp, path)

    if index is not None and archive_interval is not None and archive_interval > 0:
        if index % archive_interval == 0:
            archive_path = f'{filename}_{index}{ext}'
            shutil.copy(path, archive_path)

def cleanup_log_dir(log_dir, pattern='*'):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, pattern))
        for f in files:
            os.remove(f)

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_obs_at_index(obs, i):
    if isinstance(obs, dict):
        return {k: obs[k][i] for k in obs.keys()}
    else:
        return obs[i]

def set_obs_at_index(obs, obs_, i):
    if isinstance(obs, dict):
        for k in obs.keys():
            obs[k][i] = obs_[k].squeeze(0)
    else:
        obs[i] = obs_[0].squeeze(0)

def is_discrete_actions(env, adversary=False):
    if adversary:
        return env.adversary_action_space.__class__.__name__ == 'Discrete'
    else:
        return env.action_space.__class__.__name__ == 'Discrete'

def _make_env(args):
    env_kwargs = {'seed': args.seed}
    if args.singleton_env:
        env_kwargs.update({
            'fixed_environment': True})

    env = gym_make(args.env_name, **env_kwargs)
    return env

def create_parallel_env(args, adversary=True):
    is_multigrid = args.env_name.startswith('MultiGrid')
    is_minihack = args.env_name.startswith('MiniHack')

    make_fn = lambda: _make_env(args)

    venv = ParallelAdversarialVecEnv([make_fn]*args.num_processes, adversary=adversary)
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False, ret=args.normalize_returns)

    obs_key = None
    scale = None
    transpose_order = None
    if is_multigrid:
        obs_key = 'image'
        scale = 10.0
        transpose_order = [2,0,1] # Channels first

    elif is_minihack:
        ued_venv = VecPreprocessImageWrapper(venv=venv)

    venv = VecPreprocessImageWrapper(venv=venv, obs_key=obs_key, 
        transpose_order=transpose_order, scale=scale)

    if is_multigrid:
        ued_venv = venv

    if args.singleton_env:
        seeds = [args.seed]*args.num_processes
    else:
        seeds = [i for i in range(args.num_processes)]
    venv.set_seed(seeds)

    return venv, ued_venv
