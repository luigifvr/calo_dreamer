import numpy as np
import torch
from scipy.integrate import solve_ivp
import Source.Networks
from Source.Util.util import get
from Source.Models.ModelBase import GenerativeModel


class TBD(GenerativeModel):
    """
     Class for Trajectory Based Diffusion
     Inheriting from the GenerativeModel BaseClass
    """

    def __init__(self, params):
        super().__init__(params)
        trajectory = get(self.params, "trajectory", "sine_cosine_trajectory")
        try:
            self.trajectory = getattr(Source.Models.tbd, trajectory)
        except AttributeError:
            raise NotImplementedError(f"build_model: Trajectory type {trajectory} not implemented")


        self.C = get(self.params, "C", 1)
        if self.C != 1:
            print(f"C is {self.C}")

        self.bayesian = get(self.params, "bayesian", 0)


    def build_net(self):
        """
        Build the network
        """
        network = get(self.params, "network", "Resnet")
        try:
            return getattr(Source.Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def get_condition_and_input(self, input):
        """
        :param input: model input + conditional input
        :return: model input, conditional input
        """
        x = input.clone()
        condition = None
        weights = None
        return x, condition, weights

    def batch_loss(self, x):
        """
        Calculate batch loss as described by Peter
        """
        # get input and conditions
        x, condition, weights = self.get_condition_and_input(x)

        t = self.dist.sample((x.size(0),1)).to(x.device)
        x_0 = torch.randn_like(x)
        x_t, x_t_dot = self.trajectory(x_0, x, t)

        self.net.kl = 0
        drift = self.net(x_t, t, condition)


        loss = torch.mean((drift - x_t_dot) ** 2 * torch.exp(self.t_factor * t))
        self.regular_loss.append(loss.detach().cpu().numpy())
        if self.C != 0:
            kl_loss = self.C*self.net.kl / self.n_traindata
            self.kl_loss.append(kl_loss.detach().cpu().numpy())
            loss = loss + kl_loss

        return loss

    def get_condition_for_sample(self, n_samples, batch_size):
        """
        :param n_samples: number of samples
        :param batch_size: batch size of sampling process
        :return: conditions as additional input for model during sampling
        """
        condition = None
        return condition

    def sample_n(self, n_samples, prior_samples=None, con_depth=0):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        if self.net.bayesian:
            self.net.map = get(self.params,"fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
        self.eval()
        batch_size = get(self.params, "batch_size", 8192)
        x_T = np.random.randn(n_samples + batch_size, self.dim)

        condition = self.get_condition_for_sample(n_samples=n_samples, batch_size=batch_size)


        def f(t, x_t, c=None):
            x_t_torch = torch.Tensor(x_t).reshape((batch_size, self.dim)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])

            with torch.no_grad():
                if c is not None:
                    c_torch = torch.Tensor(c).reshape((batch_size, self.n_con)).to(self.device)
                    f_t = self.net(x_t_torch, t_torch, c_torch).detach().cpu().numpy().flatten()
                else:
                    f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        events = []
        with torch.no_grad():
            for i in range(int(n_samples / batch_size) + 1):
                if self.conditional:
                    c = condition[batch_size * i: batch_size * (i + 1)].flatten()
                else:
                    c = None
                sol = solve_ivp(f, (0, 1), x_T[batch_size * i: batch_size * (i + 1)].flatten(), args=[c])

                if self.conditional:
                    c = condition[batch_size * i: batch_size * (i + 1)]
                    if self.n_jets == 1:
                        s = np.concatenate([c, sol.y[:, -1].reshape(batch_size, self.dim)], axis=1)
                    elif self.n_jets == 2:
                        s = np.concatenate([c[:,-2:], sol.y[:, -1].reshape(batch_size, self.dim)], axis=1)
                    elif self.n_jets == 3:
                        s = sol.y[:, -1].reshape(batch_size, self.dim)
                else:
                    s = sol.y[:, -1].reshape(batch_size, self.dim)

                events.append(s)

        return np.concatenate(events, axis=0)[:n_samples]



    def invert_n(self, samples):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        if self.net.bayesian:
            self.net.map = get(self.params,"fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
        self.eval()
        batch_size = get(self.params, "batch_size", 8192)
        n_samples = samples.shape[0]

        def f(t, x_t):
            x_t_torch = torch.Tensor(x_t).reshape((-1, self.dim)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])
            with torch.no_grad():
                f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        events = []
        with torch.no_grad():
            for i in range(int(n_samples / batch_size)):
                sol = solve_ivp(f, (1, 0), samples[batch_size * i: batch_size * (i + 1)].flatten())
                s = sol.y[:, -1].reshape(batch_size, self.dim)
                events.append(s)
            sol = solve_ivp(f, (1, 0), samples[batch_size * (i+1):].flatten())
            s = sol.y[:, -1].reshape(-1, self.dim)
            events.append(s)
        return np.concatenate(events, axis=0)[:n_samples]

    def sample_n_evolution(self, n_samples):

        n_frames = get(self.params, "n_frames", 1000)
        t_frames = np.linspace(0, 1, n_frames)

        batch_size = get(self.params, "batch_size", 8192)
        x_T = np.random.randn(n_samples + batch_size, self.dim)

        def f(t, x_t):
            x_t_torch = torch.Tensor(x_t).reshape((batch_size, self.dim)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])
            with torch.no_grad():
                f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        events = []
        with torch.no_grad():
            for i in range(int(n_samples / batch_size) + 1):
                sol = solve_ivp(f, (0, 1), x_T[batch_size * i: batch_size * (i + 1)].flatten(), t_eval=t_frames)
                s = sol.y.reshape(batch_size, self.dim, -1)
                events.append(s)
        return np.concatenate(events, axis=0)[:n_samples]



def sine_cosine_trajectory(x_0, x_1, t):
    c = torch.cos(t * np.pi / 2)
    s = torch.sin(t * np.pi / 2)
    x_t = c * x_0 + s * x_1

    c_dot = -np.pi / 2 * s
    s_dot = np.pi / 2 * c
    x_t_dot = c_dot * x_0 + s_dot * x_1
    return x_t, x_t_dot

def sine2_cosine2_trajectory(x_0, x_1, t):
    c = torch.cos(t * np.pi / 2)
    s = torch.sin(t * np.pi / 2)
    x_t = c**2 * x_0 + s**2 * x_1

    c_dot = -np.pi / 2 * s
    s_dot = np.pi / 2 * c
    x_t_dot = 2 * c_dot * c * x_0 + 2 * s_dot * s * x_1
    return x_t, x_t_dot

def linear_trajectory(x_0, x_1, t):
    x_t = (1 - t) * x_0 + t * x_1
    x_t_dot = x_1 - x_0
    return x_t, x_t_dot

def vp_trajectory(x_0, x_1, t, a=19.9, b=0.1):

    e = -1./4. * a * (1-t)**2 - 1./2. * b * (1-t)
    alpha_t = torch.exp(e)
    beta_t = torch.sqrt(1-alpha_t**2)
    x_t = x_0 * alpha_t + x_1 * beta_t

    e_dot = 2 * a * (1-t) + 1./2. * b
    alpha_t_dot = e_dot * alpha_t
    beta_t_dot = -2 * alpha_t * alpha_t_dot / beta_t
    x_t_dot = x_0 * alpha_t_dot + x_1 * beta_t_dot
    return x_t, x_t_dot