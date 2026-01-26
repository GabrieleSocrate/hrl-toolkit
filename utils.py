import gymnasium as gym
import numpy as np
import torch
import random

def set_seed(seed):
    """In Python there isn't a single "global" random  generator.
    random, NumPy, PyTorch and the enviroment all use their own randomness.
    We set the same seed everywhere so the code behaves the same every time we run it"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env(env_name, seed = 0):
    env = gym.make(env_name)
    env.reset(seed = seed)
    env.action_space.seed(seed)
    return env

def to_tensor(obs, device = None):
    obs = np.asarray(obs, dtype = np.float32)
    t = torch.as_tensor(obs, dtype = torch.float32)
    if device is not None:
        t = t.to(device)
    return t

def one_hot_option(option, num_options):
    v = np.zeros(num_options, dtype = np.float32)
    v[int(option)] = 1.0
    return v