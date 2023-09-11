import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from data_util import *
from transforms import *

class CaloChallengeDataset(Dataset):
    """ Dataset for CaloChallenge showers """
    def __init__(self, hdf5_file, particle_type, xml_filename, transform=None):
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

    def __len__(self):
        return len(self.energy)

    def __getitem__(self, idx):
        showers = torch.tensor(self.layers[idx])
        energies = torch.tensor(self.energy[idx])

        if self.transform:
            for fn in self.transform:
                showers, energies = fn(showers, energies)
        return showers, energies
