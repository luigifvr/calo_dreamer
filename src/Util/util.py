import torch.optim
import torch.utils.data
import yaml
import math
import numpy as np
import h5py

from challenge_files.XMLHandler import XMLHandler
import challenge_files.HighLevelFeatures as HLF
import transforms

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


def load_data(filename, particle_type,  xml_filename, threshold=1e-5, single_energy=None):
    """Loads the data for a dataset 1 from the calo challenge"""
    
    # Create a XML_handler to extract the layer boundaries. (Geometric setup is stored in the XML file)
    xml_handler = XMLHandler(particle_name=particle_type, 
    filename=xml_filename)
    
    layer_boundaries = np.unique(xml_handler.GetBinEdges())

    # Prepare a container for the loaded data
    data = {}

    # Load and store the data. Make sure to slice according to the layers.
    # Also normalize to 100 GeV (The scale of the original data is MeV)
    data_file = h5py.File(filename, 'r')
    #data["energy"] = data_file["incident_energies"][:]
    if single_energy is not None:
        energy_mask = data_file["incident_energies"][:] == single_energy
    else:
        energy_mask = np.full(len(data_file["incident_energies"]), True)

    data["energy"] = data_file["incident_energies"][:][energy_mask].reshape(-1, 1)
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        data[f"layer_{layer_index}"] = data_file["showers"][..., layer_start:layer_end][energy_mask.flatten()]
    data_file.close()
    
    return data, layer_boundaries

def get_energy_and_sorted_layers(data):
    """returns the energy and the sorted layers from the data dict"""
    
    # Get the incident energies
    energy = data["energy"]

    # Get the number of layers layers from the keys of the data array
    number_of_layers = len(data)-1
    
    # Create a container for the layers
    layers = []

    # Append the layers such that they are sorted.
    for layer_index in range(number_of_layers):
        layer = f"layer_{layer_index}"
        
        layers.append(data[layer])
       
    layers = np.concatenate(layers, axis=1)
            
    return energy, layers

def get_transformations(transforms_list):
    func = []
    for key, params in transforms_list.items():
        func.append(getattr(transforms, key)(**params))
    return func
