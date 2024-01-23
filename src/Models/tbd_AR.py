import numpy as np
import torch
from scipy.integrate import solve_ivp
import Networks
from Util.util import get
from Models.ModelBase import GenerativeModel
import Networks
import Models

import math
from typing import Type, Callable, Union, Optional
import torch
import torch.nn as nn
from torchdiffeq import odeint

class TransfusionAR(GenerativeModel):

    def __init__(self, params: dict, device, doc):
        super().__init__(params, device, doc)
        self.params = params
        trajectory = get(self.params, "trajectory", "linear_trajectory")
        try:
            self.trajectory = getattr(Models.tbd, trajectory)
        except AttributeError:
            raise NotImplementedError(f"build_model: Trajectory type {trajectory} not implemented")

        self.dim_embedding = params["dim_embedding"]

        self.t_min = get(self.params, "t_min", 0)
        self.t_max = get(self.params, "t_max", 1)
        self.distribution = torch.distributions.uniform.Uniform(low=self.t_min, high=self.t_max)

    def build_net(self):
        """
        Build the network
        """
        network = get(self.params, "network", "ARtransformer")
        try:
            return getattr(Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def get_condition_and_input(self, input):
        """
        :param input: model input + conditional input
        :return: model input, conditional input
        """
        # x = input[0].clone()
        condition = input[1]
        weights = None
        # return x, condition, weights
        return input[0], condition, weights

    def batch_loss(self, input):
        """

        Args:
            x: input tensor, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
            kl_scale: factor in front of KL loss term, default 0
        Returns:
            loss: batch loss
            loss_terms: dictionary with loss contributions
        """
        x, c, _ = self.get_condition_and_input(input)

        # Sample noise variables
        x_0 = torch.randn_like(x)
        # Sample time steps
        #t = torch.rand((x.size(0), x.size(1)), dtype=x.dtype, device=x.device)
        t = self.distribution.sample([x.shape[0]] + [1] * (x.dim() - 1)).to(x.device)
        # Calculate point and derivative on trajectory
        x_t, x_t_dot = self.trajectory(x_0, x, t)

        v_pred = self.net(c,x_t,t,x)
        # Mask out masses if not needed
        loss = ((v_pred - x_t_dot) ** 2).mean()

        return loss

    def sample_batch(self,c):
        pred = self.net(c, rev=True)
        return pred.detach().cpu().numpy()

def linear_trajectory(x_0, x_1, t):
    x_t = (1 - t) * x_0 + t * x_1
    x_t_dot = x_1 - x_0
    return x_t, x_t_dot