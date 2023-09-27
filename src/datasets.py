import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from Util.util import *
from transforms import *

class CaloChallengeDataset(Dataset):
    """ Dataset for CaloChallenge showers """
    def __init__(self, hdf5_file, particle_type, xml_filename, val_frac=0.3, 
            transform=None, split='training', device='cpu'):
        """
        Arguments:
            hdf5_file: path to hdf5 file
            particle_type: photon, pion or electron
            xml_filename: path to XML filename
            transform: list of transformations
        """
        self.data, self.layer_boundaries = load_data(hdf5_file, particle_type, xml_filename)
        self.energy, self.layers = get_energy_and_sorted_layers(self.data)
        
        del self.data
        print("Dataset loaded, shape: ", self.layers.shape, self.energy.shape)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.energy)

    def __getitem__(self, idx):
        showers = torch.clone(torch.tensor(self.layers[idx], dtype=torch.get_default_dtype()).to(self.device))
        energies = torch.clone(torch.tensor(self.energy[idx], dtype=torch.get_default_dtype()).to(self.device))

        if self.transform:
            for fn in self.transform:
                showers, energies = fn(showers, energies)

        # Apply log-condition
        energies = np.log10(energies/1e3)
        return showers, energies
