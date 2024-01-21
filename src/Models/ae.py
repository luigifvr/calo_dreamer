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
        loss = torch.mean((x - rec) ** 2)

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

    def plot_saved_samples(self, name="", energy=None):
        script_args = (
            f"-i {self.doc.basedir}/samples_ae_reco{name}.hdf5 "
            f"-r {self.params['eval_hdf5_file']} -m all --cut {self.params['eval_cut']} "
            f"-d {self.params['eval_dataset']} --output_dir {self.doc.basedir}/final/"
        ) + (f" --energy {energy}" if energy is not None else '')
        evaluate.main(script_args.split())
