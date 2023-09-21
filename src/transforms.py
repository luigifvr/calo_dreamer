import torch
import numpy as np

from challenge_files import *

class Standardize(object):
    """
    Standardize features 
        mean: vector of means 
        std: vector of stds
    """
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = shower*stds + means
        else:
            transformed = (shower - means)/stds
        return transformed, energy

class LogTransform(object):
    """
    Take log of input data
        alpha: regularization
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, shower, energy, rev=False):
        if rev:
            shower = torch.clone(shower)
            transformed = torch.exp(shower) - self.alpha
        else:
            shower = torch.clone(shower)
            transformed = torch.log(shower + self.alpha)
        return transformed, energy

class AddNoise(object):
    """
    Add noise to input data
        func: torch distribution used to sample from
        width_noise: noise rescaling
    """
    def __init__(self, noise_width):
        #self.func = func
        self.func = torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(1.0))
        self.noise_width = noise_width
    
    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower < self.noise_width)
            transformed = shower
            transformed[mask] = 0.0
        else:
            noise = self.func.sample(shower.shape)*self.noise_width
            transformed = shower + noise.reshape(shower.shape)
        return transformed, energy

class NormalizeByEinc(object):
    """
    Normalize each shower by the incident energy

    """
    def __call__(self, shower, energy, rev=False):
        if rev:
            shower *= energy
        else:
            shower /= energy
        return shower, energy

class NormalizeByElayer(object):
    """
    Normalize each shower by the layer energy
    This will change the shower shape to N_voxels+N_layers
       layer_boundaries: ''
       eps: numerical epsilon
    """
    def __init__(self, ptype, xml_file, eps=1.e-10):
        self.eps = eps
        self.xml = XMLHandler.XMLHandler(xml_file, ptype)
        self.layer_boundaries = np.unique(self.xml.GetBinEdges())
        self.number_of_layers = len(self.layer_boundaries) - 1

    def __call__(self, shower, energy, rev=False):
        if rev:
            shower = shower.to(torch.float64)

            extra_dims = shower[..., -self.number_of_layers:]
            extra_dims[:, (-self.number_of_layers+1):] = torch.clip(extra_dims[:, (-self.number_of_layers+1):], min=torch.tensor(0.), max=torch.tensor(1.))   #clipping 
            shower = shower[:, :-self.number_of_layers]
            transformed = torch.zeros_like(shower, dtype=torch.float64)

            layer_energies = []
            en_tot = torch.multiply(energy.flatten(), extra_dims[:,0])
            cum_sum = torch.zeros_like(en_tot, dtype=torch.float64)
            for i in range(extra_dims.shape[-1]-1):
                ens = (en_tot - cum_sum)*extra_dims[:,i+1]
                layer_energies.append(ens)
                cum_sum += ens

            layer_energies.append((en_tot - cum_sum))
            layer_energies = torch.vstack(layer_energies).T
            # Normalize each layer and multiply it with its original energy
            for layer_index, (layer_start, layer_end) in enumerate(zip(self.layer_boundaries[:-1], self.layer_boundaries[1:])):
                transformed[..., layer_start:layer_end] = shower[..., layer_start:layer_end] * layer_energies[..., [layer_index]]  / \
                                             (torch.sum(shower[..., layer_start:layer_end], axis=1, keepdims=True) + self.eps)

        else:
            shower = torch.clone(shower.to(torch.float64))
            energy = torch.clone(energy.to(torch.float64))

            #calculate extra dimensions
            layer_energies = []
             
            for layer_start, layer_end in zip(self.layer_boundaries[:-1], self.layer_boundaries[1:]):
                layer_energy = torch.sum(shower[layer_start:layer_end], dim=0, keepdim=True)
                shower[layer_start:layer_end] = shower[layer_start:layer_end] / (layer_energy + self.eps)
                layer_energies.append(layer_energy)
        
            layer_energies_torch = torch.tensor(layer_energies)
        
            # Compute the generalized extra dimensions
            extra_dims = [torch.sum(layer_energies_torch, dim=0, keepdim=True) / energy]

            for layer_index in range(len(self.layer_boundaries)-2):
                extra_dim = layer_energies_torch[..., [layer_index]] / (torch.sum(layer_energies_torch[layer_index:], dim=0, keepdim=True) + self.eps)
                extra_dims.append(extra_dim)
        
            extra_dims = torch.cat(extra_dims, dim=0)
            # normalize by E_layer
            for layer_index, (layer_start, layer_end) in enumerate(zip(self.layer_boundaries[:-1], self.layer_boundaries[1:])):
                shower[layer_start:layer_end] = shower[layer_start:layer_end] / ( torch.sum(shower[layer_start:layer_end], dim=0, keepdim=True) + self.eps)

            transformed = torch.cat((shower, extra_dims), dim=0)

        return transformed, energy
        
