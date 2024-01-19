import numpy as np
import torch

import Networks
import Models
from Models.ModelBase import GenerativeModel
from Util.util import get

from challenge_files import *
from challenge_files import evaluate

class AE(GenerativeModel):
    """
    Class for the AutoEncoder
    """

    def __init__(self, params, device, doc):
        super().__init__(params, device, doc)
        # parameters for autoencoder

    def build_net(self):
        """
        Build the network
        """
        network = get(self.params, "ae_network", "AutoEncoder")
        try:
            return getattr(Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")
    
    def get_conditions_and_input(self, input):
        """
        :param input: model input + conditional input
        :return: model input, conditional input
        """
        # x = input[0].clone()
        condition = input[1]
        weights = None
        # return x, condition, weights
        return input[0], condition, weights

    def batch_loss(self, x):
        #calculate loss for 1 batch
        x, condition, weights = self.get_conditions_and_input(x)

        rec = self.net(x, condition)
        loss_fn = self.params.get('ae_loss', 'mse')
        if loss_fn == 'mse':
            loss = torch.mean((x - rec) ** 2)
        elif loss_fn == 'bce':
            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(rec, x)
        elif loss_fn == 'mod-bce':
            loss_fn = torch.nn.CrossEntropyLoss()
            rec = rec.squeeze().reshape(-1, 16*9)
            x = x.squeeze().reshape(-1, 16*9)
            loss = loss_fn(rec, x)
        elif loss_fn == 'bce_mse':
            loss_fn = torch.nn.BCELoss()
            loss_bce = loss_fn(rec, x)
            loss_mse = torch.mean((torch.special.logit(x, eps=1.e-6) - torch.special.logit(rec, eps=1.e-6))**2)
            loss = loss_bce+ 0.001*loss_mse
        else:
            raise Exception("Unknown loss function")

        return loss
    
    def sample_batch(self, x):
        with torch.no_grad():
            x, condition, weights = self.get_conditions_and_input(x)
            rec = self.net(x, condition)
        return rec.detach().cpu(), condition.detach().cpu()
 
    def plot_samples(self, samples, conditions, name="", energy=None):
        transforms = self.transforms
        print("Plotting reconstructions of input showers")

        for fn in transforms[::-1]:
            samples, conditions = fn(samples, conditions, rev=True)

        samples = samples.detach().cpu().numpy()
        conditions = conditions.detach().cpu().numpy()

        self.save_sample(samples, conditions, name="_ae_reco")
        evaluate.run_from_py(samples, conditions, self.doc, self.params)
