import numpy as np
import torch

import Networks
import Models
from Models.ModelBase import GenerativeModel
from Util.util import get
from Util.util import loss_cbvae

from challenge_files import *
from challenge_files import evaluate

class AE(GenerativeModel):
    """
    Class for the AutoEncoder
    """

    def __init__(self, params, device, doc):
        super().__init__(params, device, doc)
        self.lat_mean = None
        self.lat_std = None
        self.shape = self.params.get('ae_latent_shape')
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

    def forward(self, x):
        # Get reconstructions
        x, condition, weights = self.get_conditions_and_input(x)
        c = self.net.c_encoding(condition)
        z = self.net.encode(x, c)
        if self.params.get('ae_kl', False):
            mu, logvar = z[0], z[1]
            z = self.net.reparameterize(mu, logvar)
            #z = z.reshape(-1,*self.shape)
            rec = self.net.decode(z, c)
            return rec, mu, logvar
        return rec

    def batch_loss(self, x):
        #calculate loss for 1 batch
        if self.params.get('ae_kl', False):
            rec, mu, logvar = self.forward(x)
        else:
            rec = self.forward(x)
        x, condition, weights = self.get_conditions_and_input(x)
        loss_fn = self.params.get('ae_loss', 'mse')
        if loss_fn == 'mse':
            loss = torch.mean((x - rec) ** 2)
        elif loss_fn == 'bce':
            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(rec, x)
        elif loss_fn == 'mod-bce':
            loss = -torch.mean(x*rec)
        elif loss_fn == 'bce_mse':
            loss_fn = torch.nn.BCELoss()
            loss_bce = loss_fn(rec, x)
            loss_mse = torch.mean((torch.special.logit(x, eps=1.e-6) - torch.special.logit(rec, eps=1.e-6))**2)
            loss = loss_bce+ 0.0001*loss_mse
        elif loss_fn == 'cbce':
            loss = loss_cbvae(rec, x)
        elif loss_fn == 'bce_reg':
            alpha = self.params.get('bce_reg_alpha', 1.0)
            loss_fn = torch.nn.BCELoss()
            
            loss_reg = torch.mean(enc**2/2)
            loss = loss_fn(rec, x)
            loss += alpha*loss_reg
        elif loss_fn == 'bce_kl':
            loss_fn = torch.nn.BCELoss()
            beta = self.params.get('ae_kl_beta', 1.e-5)
            KLD = -0.5 * torch.mean(1 + logvar - mu**2 -  logvar.exp())
            loss = loss_fn(rec, x) + beta*KLD
        else:
            raise Exception("Unknown loss function")

        return loss
    
    def sample_batch(self, x):
        with torch.no_grad():
            x_inp, condition, weights = self.get_conditions_and_input(x)
            if self.params.get('ae_kl', False):
                rec, mu, logvar = self.forward(x)
            else:
                rec = self.net(x, condition)
        return rec.detach().cpu(), condition.detach().cpu()
 
    def plot_samples(self, samples, conditions, name="", energy=None, mode='all'): #TODO
        transforms = self.transforms
        print("Plotting reconstructions of input showers")

        for fn in transforms[::-1]:
            samples, conditions = fn(samples, conditions, rev=True)

        samples = samples.detach().cpu().numpy()
        conditions = conditions.detach().cpu().numpy()

        self.save_sample(samples, conditions, name="_ae_reco")
        evaluate.run_from_py(samples, conditions, self.doc, self.params)

    @torch.no_grad()
    def encode(self, x, c):
        c = self.net.c_encoding(c)
        enc = self.net.encode(x,c)
        return enc

    @torch.no_grad()
    def decode(self, x, c):
        c = self.net.c_encoding(c)
        if self.params.get('ae_kl', False):
            z = self.net.parameterize(x[0], x[1])
            return self.net.decode(z, c)
        return self.net.decode(x, c)

    @torch.no_grad()
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(mu.device)
        z = mu + std*esp
        return z
