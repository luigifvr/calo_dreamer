import torch


def get_transforms(list_tr):
    transforms = []
    for i in list_tr:
        if i == 'std':
            pass
    return transforms

class Standardize(object):
    """
    Standardize features 
        mean: vector of means 
        std: vector of stds
    """
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def __call__(self, shower, energy):
        transformed = (shower - means)/stds
        return transformed, energy

class LogTransform(object):
    """
    Take log of input data
        alpha: regularization
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, shower, energy):
        transformed = torch.log(shower + self.alpha)
        return transformed, energy

class AddNoise(object):
    """
    Add noise to input data
        func: torch distribution used to sample from
        width_noise: noise rescaling
    """
    def __init__(self, func, noise_width):
        self.func = func
        self.noise_width = noise_width
    
    def __call__(self, shower, energy):
        noise = self.func.sample(shower.shape)*self.noise_width
        transformed = shower + noise.reshape(shower.shape)
        return transformed, energy

class NormalizeByEinc(object):
    """
    Normalize each shower by the incident energy

    """
    def __call__(self, shower, energy):
        shower /= energy
        return shower, energy

class NormalizeByElayer(object):
    """
    Normalize each shower by the layer energy
    This will change the shower shape to N_voxels+N_layers
       layer_boundaries: ''
       eps: numerical epsilon
    """
    def __init__(self, layer_boundaries, eps=1.e-10):
        self.eps = eps
        self.layer_boundaries = layer_boundaries
        self.number_of_layers = len(self.layer_boundaries) - 1

    def __call__(self, shower, energy):
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

        shower = torch.cat((shower, extra_dims), dim=0)
        return shower, energy
        
