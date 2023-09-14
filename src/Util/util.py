import torch.optim
import torch.utils.data
import yaml
import math
import numpy as np

"""
Some useful utility functions that don"t fit in anywhere else
"""


def load_params(path):
    """
    Method to load a parameter dict from a yaml file
    :param path: path to a *.yaml parameter file
    :return: the parameters as a dict
    """
    with open(path) as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
        return param


def save_params(params, name="paramfile.yaml"):
    """
    Method to save a parameter dict to a yaml file
    :param params: the parameter dict
    :param name: the name of the yaml file
    """
    with open(name, 'w') as f:
        yaml.dump(params, f)


def get(dict, key, default):
    """
    Method to extract a key from a dict.
    If the key is not contained in the dict, the default value is returned and written into the dict.
    :param dict: the dictionary
    :param key: the key
    :param default: the default value of the key
    :return: the value of the key in the dict if it exists, the default value otherwise
    """

    if key in dict:
        return dict[key]
    else:
        dict[key] = default
        return default


def get_device():
    """Check whether cuda can be used"""
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    return device


def linear_beta_schedule(timesteps):
    """
    linear beta schedule for DDPM diffusion models
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine beta schedule for DDPM diffusion models
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
