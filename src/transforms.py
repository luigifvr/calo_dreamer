import torch
import numpy as np

from challenge_files import *
from challenge_files import XMLHandler

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
            transformed = shower*self.stds + self.means
        else:
            transformed = (shower - self.means)/self.stds
        return transformed, energy

class SelfStandardize(object):
    """
    Standardize features 
        mean: vector of means 
        std: vector of stds
    """

    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = shower*self.stds.to(shower.device) + self.means.to(shower.device)
        else:
            if hasattr(self, 'means'):
                print('WARNING: Standardize transformation is recalculating modes!')
            self.means = shower.mean(axis=0).to(shower.device)
            self.stds = shower.std(axis=0).to(shower.device)
            transformed = (shower - self.means)/self.stds
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

class SelectiveLogTransform(object):
    """
    Take log of input data
        alpha: regularization
        exclusions: list of indices for features that should not be transformed
    """
    def __init__(self, alpha, exclusions=None):
        self.alpha = alpha
        self.exclusions = exclusions

    def __call__(self, shower, energy, rev=False):
        shower = torch.clone(shower)
        if rev:
            transformed = torch.exp(shower) - self.alpha
        else:
            transformed = torch.log(shower + self.alpha)
        if self.exclusions is not None:
            transformed[..., self.exclusions] = shower[..., self.exclusions]
        return transformed, energy

class AddNoise(object):
    """
    Add noise to input data
        func: torch distribution used to sample from
        width_noise: noise rescaling
    """
    def __init__(self, noise_width, cut=False):
        #self.func = func
        self.func = torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(1.0))
        self.noise_width = noise_width
        self.cut = cut # apply cut if True

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower < self.noise_width)
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0 
        else:
            noise = self.func.sample(shower.shape)*self.noise_width
            transformed = shower + noise.reshape(shower.shape).to(shower.device)
        return transformed, energy

class AddPowerlawNoise(object):
    """
    Add noise to input data following a power law distribution:
        eps ~ k x^(k-1)
        k   -- The power parameter of the distribution
        cut -- The value below which voxels will be masked to zero in the reverse transformation
    """
    def __init__(self, k, cut=None):
        self.k = k
        self.cut = cut

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower < self.cut)
            transformed = shower
            if self.cut is not None:
                transformed[mask] = 0.0
        else:
            noise = torch.from_numpy(np.random.power(self.k, shower.shape)).to(shower.dtype)
            transformed = shower + noise.reshape(shower.shape).to(shower.device)
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
            # Testing no casting to float64
            #shower = shower.to(torch.float64)

            extra_dims = shower[..., -self.number_of_layers:]
            extra_dims[:, (-self.number_of_layers+1):] = torch.clip(extra_dims[:, (-self.number_of_layers+1):], min=torch.tensor(0.), max=torch.tensor(1.))   #clipping 
            shower = shower[:, :-self.number_of_layers]
            transformed = torch.zeros_like(shower)

            layer_energies = []
            en_tot = torch.multiply(energy.flatten(), extra_dims[:,0])
            cum_sum = torch.zeros_like(en_tot)
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
            #calculate extra dimensions
            layer_energies = []
             
            for layer_start, layer_end in zip(self.layer_boundaries[:-1], self.layer_boundaries[1:]):
                layer_energy = torch.sum(shower[:, layer_start:layer_end], dim=1, keepdim=True)

                shower[:, layer_start:layer_end] = shower[:, layer_start:layer_end] / (layer_energy + self.eps)
                layer_energies.append(layer_energy)
        
            layer_energies_torch = torch.cat(layer_energies, dim=1).to(shower.device)

            # Compute the generalized extra dimensions
            extra_dims = [torch.sum(layer_energies_torch, dim=1, keepdim=True) / energy]

            for layer_index in range(len(self.layer_boundaries)-2):
                extra_dim = layer_energies_torch[..., [layer_index]] / (torch.sum(layer_energies_torch[:, layer_index:], dim=1, keepdim=True) + self.eps)
                extra_dims.append(extra_dim)
        
            extra_dims = torch.cat(extra_dims, dim=1)
            # normalize by E_layer
            for layer_index, (layer_start, layer_end) in enumerate(zip(self.layer_boundaries[:-1], self.layer_boundaries[1:])):
                shower[:, layer_start:layer_end] = shower[:, layer_start:layer_end] / ( torch.sum(shower[:, layer_start:layer_end], dim=1, keepdim=True) + self.eps)

            transformed = torch.cat((shower, extra_dims), dim=1)

        return transformed, energy
