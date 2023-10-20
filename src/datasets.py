import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from Util.util import *
from transforms import *

class CaloChallengeDataset(Dataset):
    """ Dataset for CaloChallenge showers """
    def __init__(self, hdf5_file, particle_type, xml_filename, val_frac=0.3, 
            transform=None, split='training', device='cpu', single_energy=None):
        """
        Arguments:
            hdf5_file: path to hdf5 file
            particle_type: photon, pion or electron
            xml_filename: path to XML filename
            transform: list of transformations
        """
        
        # TODO: Use `val_frac` argument to select subset of data according to `split`
        self.voxels, self.layer_boundaries = load_data(hdf5_file, particle_type, xml_filename, single_energy=single_energy)
        self.energy, self.layers = get_energy_and_sorted_layers(self.voxels)
        del self.voxels
                
        self.transform = transform
        self.device = device
        self.dtype = torch.get_default_dtype()
        self.energy = torch.tensor(self.energy, dtype=self.dtype).to(device=self.device)
        self.layers = torch.tensor(self.layers, dtype=self.dtype).to(device=self.device)

        # apply preprocessing
        if self.transform:
            for fn in self.transform:
                self.layers, self.energy = fn(self.layers, self.energy)
        self.energy = torch.log(self.energy/1e3)

        print("Dataset loaded, shape: ", self.layers.shape, self.energy.shape)

        # make train/val split
        val_size = int(len(self.energy)*val_frac)
        trn_size = len(self.energy) - val_size
        self.data = torch.utils.data.random_split(
            torch.utils.data.TensorDataset(self.layers, self.energy),
            [trn_size, val_size]
        )[0 if split=='training' else 1 if split=='validation' else None]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # Testing loading everything on GPU
        #showers = torch.tensor(self.layers[idx]).to(device=self.device)
        #energies = torch.tensor(self.energy[idx]).to(device=self.device)
        # showers = self.layers[idx]
        # energies = self.energy[idx]
        return self.data[idx]
