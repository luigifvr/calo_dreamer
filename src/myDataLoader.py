import math
from typing import Tuple, Iterator

import torch

class MyDataLoader:
    """ A data loader that is way faster than the PyTorch data loader for data with a dimension of less than tausend. """

    data: torch.Tensor
    batch_size: int
    drop_last: bool
    shuffle: bool

    def __init__(self, data: torch.Tensor, cond: torch.Tensor, batch_size: int,
                drop_last:bool=False, shuffle:bool=True, width_noise:float=1e-7) -> None:
        """
            Initializes MyDataLoader class.

            Parameters:
            data (tensor): Data
            cond (tensor): Condition
            batch_size (int): Batch size
            drop_last (bool): If true the last batch is dropped if it is smaller as the given batch size
            shuffle (bool): Weather or not to shuffle the data
            width_noise (float): Width of the noise to add to the data
        """
        self.data = data
        self.cond = cond
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.width_noise  = width_noise
        self.noise_distribution = torch.distributions.Uniform(torch.tensor(0., device=data.device), torch.tensor(1., device=data.device))
        self.noise_distribution_2 = torch.distributions.Uniform(torch.tensor(1.0e-6, device=data.device), torch.tensor(1.0e-5, device=data.device))
        self.noise_distribution_32 = torch.distributions.log_normal.LogNormal(torch.tensor([-10.0], device=data.device), torch.tensor([0.3], device=data.device))
        self.noise_distribution_3 = torch.distributions.log_normal.LogNormal(torch.tensor([-12.0], device=data.device), torch.tensor([0.3], device=data.device))

       # self.noise_distribution = torch.distributions.Beta(torch.tensor(3., device=data.device), torch.tensor(3., device=data.device))

        if self.drop_last:
            self.max_batch = len(self.data) // self.batch_size
        else:
            self.max_batch = math.ceil(len(self.data) / self.batch_size)
	
    def u_noise(self, us):
        uc = torch.clone(us)
        #noise = (torch.rand(uc.shape, device=uc.device)*(1.0e-3 - 1.0e-6) + 1.0e-6)*(-1)*torch.sign(uc-0.5)
        noise = self.noise_distribution_2.sample(uc.shape)*(-1.*torch.sign(uc-0.5))
        return noise

    def add_noise(self, input: torch.Tensor) -> torch.Tensor:
        noise = self.noise_distribution.sample(input.shape)*self.width_noise
        #noise[:, -5] = 0.0
        #noise[:, -4:] = self.u_noise(input[:, -4:])
        return input + noise.reshape(input.shape)

    def add_noise_v2(self, input: torch.Tensor) -> torch.Tensor:
        noise = self.noise_distribution.sample(input.shape)*self.q
        
        #mask = input==0.
        #input[mask] += noise[mask]
        return input

    def add_noise_v3(self, input: torch.Tensor) -> torch.Tensor:
        noise = self.noise_distribution_3.sample(input.shape)
        noise[:, -2] = self.noise_distribution_32.sample(noise[:,-2].shape).reshape(-1, 1)
        noise[:, -17:-12] = self.noise_distribution_32.sample(noise[:, -17:-12].shape).reshape(-1, 5, 1)
        #noise[:, -4:] = self.u_noise(input[:, -4:])
        return input + noise.reshape(input.shape)


    def set_quantiles(self, q: torch.Tensor) -> torch.Tensor:
        self.q = q

    def __len__(self) -> int:
        return self.max_batch

    def __iter__(self) -> Iterator[torch.Tensor]:
        # reset
        if self.shuffle:
            self.index = torch.randperm(len(self.data), device=self.data.device)
        else:
            self.index = torch.arange(len(self.data), device=self.data.device)
        self.batch = 0

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch >= self.max_batch:
            raise StopIteration
        first = self.batch*self.batch_size
        last = min(first+self.batch_size, len(self.data))
        idx = self.index[first:last]
        self.batch += 1
        return torch.clone(self.add_noise(self.data[idx])), torch.clone(self.cond[idx])
        #data = torch.clone(self.add_noise(self.data[idx]))
        #data[:,:-3].clamp_(max=1.0)
        #cond = torch.clone(self.cond[idx])
        #return data, cond
