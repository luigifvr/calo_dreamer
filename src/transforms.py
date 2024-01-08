import torch
import numpy as np
import os

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

class StandardizeFromFile(object):
    """
    Standardize features 
        mean_path: path to `.npy` file containing means of the features 
        std_path: path to `.npy` file containing standard deviations of the features
        create: whether or not to calculate and save mean/std based on first call
    """

    def __init__(self, model_dir):

        self.model_dir = model_dir
        self.mean_path = os.path.join(model_dir, 'means.npy')
        self.std_path = os.path.join(model_dir, 'stds.npy')
        self.dtype = torch.get_default_dtype()
        try:
            # load from file
            self.mean = torch.from_numpy(np.load(self.mean_path)).to(self.dtype)
            self.std = torch.from_numpy(np.load(self.std_path)).to(self.dtype)
            self.written = True
        except FileNotFoundError:
            self.written = False

    def write(self, shower, energy):
        self.mean = shower.mean(axis=0)
        self.std = shower.std(axis=0)
        np.save(self.mean_path, self.mean.detach().cpu().numpy())
        np.save(self.std_path, self.std.detach().cpu().numpy())
        self.written = True

    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = shower*self.std.to(shower.device) + self.mean.to(shower.device)
        else:
            if not self.written:
                self.write(shower, energy)
            transformed = (shower - self.mean.to(shower.device))/self.std.to(shower.device)
        return transformed, energy

class SelectDims(object):
    """
    Selects a subset of the features 
        start: start of range of indices to keep
        end:   end of range of indices to keep (exclusive)
    """

    def __init__(self, start, end):
        self.indices = torch.arange(start, end)
    def __call__(self, shower, energy, rev=False):
        if rev:
           return shower, energy
        transformed = shower[..., self.indices]
        return transformed, energy

class AddFeaturesToCond(object):
    """
    Transfers a subset of the input features to the condition
        split_index: Index at which to split input. Features past the index will be moved
    """

    def __init__(self, split_index):
        self.split_index = split_index
    
    def __call__(self, x, c, rev=False):
        
        if rev:
            c_, split = c[:, :1], c[:, 1:]
            x_ = torch.cat([x, split], dim=1)
        else:
            x_, split = x[:, :self.split_index], x[:, self.split_index:]
            c_ = torch.cat([c, split], dim=1)
        return x_, c_
    
class LogEnergy(object):
    """
    Log transform incident energies
        alpha: Optional regularization for the log
    """            
    def __init__(self, alpha=0.):
        self.alpha = alpha
        self.cond_transform = True
        
    def __call__(self, shower, energy, rev=False):
            if rev:
                transformed = torch.exp(energy) - self.alpha
            else:
                transformed = torch.log(energy + self.alpha)
            return shower, transformed              

class ScaleEnergy(object):
    """
    Scale incident energies to lie in the range [0, 1]
        e_min: Expected minimum value of the energy
        e_max: Expected maximum value of the energy
    """
    def __init__(self, e_min, e_max):
        self.e_min = e_min
        self.e_max = e_max
        self.cond_transform = True

    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = energy * (self.e_max - self.e_min)
            transformed += self.e_min
        else:
            transformed = energy - self.e_min
            transformed /= (self.e_max - self.e_min)
        return shower, transformed

class LogTransform(object):
    """
    Take log of input data
        alpha: regularization
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = torch.exp(shower) - self.alpha
        else:
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
        if rev:
            transformed = torch.exp(shower) - self.alpha
        else:
            transformed = torch.log(shower + self.alpha)
        if self.exclusions is not None:
            transformed[..., self.exclusions] = shower[..., self.exclusions]
        return transformed, energy


class ExclusiveLogitTransform(object):
    """
    Take log of input data
        delta: regularization
        exclusions: list of indices for features that should not be transformed
    """

    def __init__(self, delta, exclusions=None):
        self.delta = delta
        self.exclusions = exclusions

    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = torch.special.expit(shower)
        else:
            transformed = torch.special.logit(shower, eps=self.delta)
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


class SmoothUPeaks(object):
    """
    Smooth voxels equal to 0 or 1 using uniform noise
        w0: noise width for zeros
        w1: noise width for ones
        eps: threshold below which values are considered zero
    """

    def __init__(self, w0, w1, eps=1.e-10):
        self.func = torch.distributions.Uniform(
            torch.tensor(0.0), torch.tensor(1.0))
        self.w0 = w0
        self.w1 = w1
        self.scale = 1 + w0 + w1
        self.eps = eps

    def __call__(self, u, energy, rev=False):
        if rev:
            # undo scaling
            transformed = u*self.scale - self.w0
            # clip to [0, 1]
            transformed = torch.clip(transformed, min=0., max=1.)
            # restore u0
            transformed[:, 0] = u[:, 0]
        else:
            # sample noise values
            n0 = self.w0*self.func.sample(u.shape).to(u.device)
            n1 = self.w1*self.func.sample(u.shape).to(u.device)
            # add noise to us
            transformed = u - n0*(u<=self.eps) + n1*(u>=1-self.eps)
            # scale to [0,1] in preparation for logit
            transformed = (transformed + self.w0)/self.scale
            # restore u0
            transformed[:, 0] = u[:, 0]
            
        return transformed, energy

class SelectiveUniformNoise(object):
    """
    Add noise to input data with the option to exlude some features
        func: torch distribution used to sample from
        width_noise: noise rescaling
        exclusions: list of indices for features that should not be transformed
    """
    def __init__(self, noise_width, exclusions = None, cut=False):
        #self.func = func
        self.func = torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(1.0))
        self.noise_width = noise_width
        self.exclusions = exclusions
        self.cut = cut # apply cut if True

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower < self.noise_width)
            if self.exclusions:
                mask[:, self.exclusions] = False
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0 
        else:
            noise = self.func.sample(shower.shape)*self.noise_width
            if self.exclusions:
                noise[:, self.exclusions] = 0.0
            transformed = shower + noise.reshape(shower.shape).to(shower.device)
        return transformed, energy        

class ZeroMask(object):
    """
    Masks voxels to zero in the reverse transformation
        cut: threshold value for the mask
    """
    def __init__(self, cut=0.):
        self.cut = cut

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower < self.cut)
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0 
        else:
            transformed = shower
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
            transformed = shower*energy
        else:
            transformed = shower/energy
        return transformed, energy

class Reshape(object):
    """
    Reshape the shower as specified. Flattens batch in the reverse transformation.
        shape -- Tuple representing the desired shape of a single example
    """

    def __init__(self, shape):
        self.shape = torch.Size(shape)

    def __call__(self, shower, energy, rev=False):
        if rev:
            shower = shower.reshape(-1, self.shape.numel())
        else:
            shower = shower.reshape(-1, *self.shape)
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
            extra_dims[:, (-self.number_of_layers+1):] = torch.clip(extra_dims[:, (-self.number_of_layers+1):], min=torch.tensor(0., device=shower.device), max=torch.tensor(1., device=shower.device))   #clipping 
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

class AddCoordChannels(object):
    """
    Add channel to image containing the coordinate value along particular
    dimension. This breaks the translation symmetry of the convoluitons,
    as discussed in arXiv:2308.03876

        dims -- List of dimensions for which should have a coordinate channel
                should be created.
    """

    def __init__(self, dims):
        self.dims = dims

    def __call__(self, shower, energy, rev=False):

        if rev:
            transformed = shower # generated shower already only has 1 channel
        else:
            coords = []
            for d in self.dims:
                bcst_shp = [1] * shower.ndim
                bcst_shp[d] = -1
                size = shower.size(d)
                coords.append(torch.ones_like(shower) / size *
                              torch.arange(size).view(bcst_shp))
            transformed = torch.cat([shower] + coords, dim=1)
        return transformed, energy