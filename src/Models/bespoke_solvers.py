import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from abc import abstractmethod
from documenter import Documenter
from Models import TBD
from Util.util import load_params, set_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint

    
class BespokeSolver(nn.Module):

    # TODO: Implement `run_training`, `sample_n`, `plot_samples`

    """
    A base class for implementing bespoke solvers from arxiv:2310.19075. The
    model parameterises a transformation of the integration path of an ode and
    optimizes it such that solutions match that of an expensive solver.

    __init__(params, device, doc):
        
        params: A dictionary specifying the parameters of the solver:
            flow       -- Path to a flow model representing the vector field to
                       be integrated. It should have signature (x,t,c) --> x,
                       where c is a possible condition.
            num_steps  -- The number of integration steps to take.
            shape      -- The shape of the state x
            L_tau      -- The Lipschitz contant hyperparameter # TODO: explain better
            truth_tols -- Settings for the atol and rtol parameters of the truth
                       solver, passed as a dictionary.
        device: The device on which to store model parameters and flow network.
        doc: A Documenter object used for logging and saving outputs
    """

    def __init__(self, params, device, doc):
        
        super().__init__()
        
        self.params = params
        self.device = device
        self.doc = doc
        self.flow_dir = params['shape_model']
        self.num_steps = params['num_steps']
        self.L_tau = params.get('L_tau', 1.)
        self.truth_tols = params.get('truth_tols', None)

        self.loss = params.get('loss', 'gte_bound')
        self.init_params()
        self.load_flow_models()
        self.cast_shape = [-1] + [1]*(1+len(self.shape))
        
    @abstractmethod
    def init_params():
        pass

    @abstractmethod
    def step():
        pass
        
    @abstractmethod
    def lipschitz():
        pass

    @abstractmethod
    def t_sol():
        pass

    @property
    def t(self):
        t = torch.linspace(0,1,len(self.theta_t)+2, device=self.device)
        t[1:-1] = self.theta_t.abs().cumsum(0)/len(self.theta_t_dot)
        return t

    @property
    def t_dot(self):
        return F.softplus(self.theta_t_dot)

    @property
    def s(self):
        s = torch.ones(len(self.theta_s)+1, device=self.device)
        s[1:] = F.softplus(self.theta_s) 
        return s
        
    @property
    def s_dot(self):
        return self.theta_s_dot

    @property
    def h(self):
        return 1/self.num_steps

    @staticmethod
    def lipschitz_u(s, s_dot, t_dot, L_tau):
        return abs(s_dot)/s + t_dot*L_tau

    def flow_fn(self, x, t, c):
        """A wrapper for self.flow.net that correctly shapes things."""

        orig_shape = x.shape
        batch_size = x.shape[-len(self.shape)-1]
        if len(x.shape) > len(self.shape)+1:
            # running in parallel (training)
            c = c.repeat(len(t), 1)
            x = x.view(-1, *self.shape)
        t = t.repeat_interleave(batch_size).view(-1,1)
        return self.flow.net(x, t, c).view(*orig_shape)

    def gte_bound_loss(self, x, cond=None):

        # print(f'{x.shape=}')
        # Eq. 24
        d = torch.sqrt(torch.mean(
            (x[1:] - self.step(x[:-1], cond))**2, dim=list(range(2, x.ndim))
        ))
        # print(f'{d.shape=}')
        # print(f'{self.lipschitz.shape=}')
        # Eq. 25
        m = self.lipschitz[1:].flip([0]).cumprod(0).flip([0])
        # print(f'{m.shape=}')
        
        return m @ d[:-1] + d[-1] # L_n set to 1

    def lte_loss(self, x, cond=None):

        # Eq. 24
        d = torch.sqrt(torch.mean(
            (x[1:] - self.step(x[:-1], cond))**2, dim=list(range(2, x.ndim))
        ))
        
        return d.sum(0)
    
    def gte_loss(self, x_true, x0, cond=None):
        x_sol = self.solve(x0=x0, cond=cond)
        rmse = torch.sqrt(torch.mean(
            (x_true - x_sol)**2, dim=list(range(1, x_sol.ndim))
        ))
        return rmse

    def forward(self, cond=None, batch_size=None):
        """
        Solve a truth trajectory and return a bound on the truncation error of
        the bespoke solver.
        """
        if cond is not None:
            batch_size = cond.shape[0]

        f = lambda t, x: self.flow_fn(x, t, cond)
        x0 = torch.randn((batch_size, *self.shape), device=self.device)

        if self.loss == 'gte':
            t_sol = torch.tensor(
                [0, 1], dtype=torch.float32, device=self.device
            )
            with torch.inference_mode():
                x_true = odeint(f, x0, t_sol, **self.truth_tols)[-1]
            return self.gte_loss(x_true, x0, cond)
        
        with torch.inference_mode():
            t_stop = self.t_sol.detach()
            x_true = odeint(f, x0, t_stop, **self.truth_tols)
            vel =  f(t_stop, x_true)
        x_aux = x_true + vel*(self.t_sol - t_stop).view(*self.cast_shape)
        
        if self.loss == 'gte_bound':
            return self.gte_bound_loss(x_aux, cond)
        
        if self.loss == 'lte':
            return self.lte_loss(x_aux, cond)

    # @torch.inference_mode()
    def solve(self, cond=None, x0=None):
        """Alg. 2 and Eq. 17"""

        if x0 is None:
            # assume initial state x0 follows standard normal
            x0 = torch.randn((cond.shape[0], *self.shape), device=self.device)
        
        x_next = x0
        for i in range(self.num_steps):
            x_next = self.step(x_next, cond, r=i)
        
        return x_next
    
    def condition_generator(self, iterations, batch_size):
        """A generator for sampling conditions during training and inference."""
        for _ in range(iterations):
            Eincs = torch.rand([batch_size, 1], device=self.device) # Assumes u_model expects Einc uniform in [0,1]
            with torch.inference_mode():
                u_samples = self.u_model.sample_batch(Eincs)
            yield torch.cat([Eincs, u_samples], dim=1) 

    def load_flow_models(self):
        """Load the flow model defining the vector field to be integrated."""
        
        # read flow parameters
        flow_params = load_params(os.path.join(self.flow_dir, 'params.yaml'))
        flow_params['eval_mode'] = self.params.get('eval_mode', 'all')
        # initialioze flow
        flow_doc = Documenter(None, existing_run=self.flow_dir, read_only=True)
        self.flow = TBD(flow_params, self.device, flow_doc)
        self.flow.load()
        self.flow.eval()
        for p in self.flow.parameters():
            p.requires_grad=False
        # set shape based on flow
        self.shape = self.flow.shape
        # initialize energy model for conditions
        self.u_model = self.flow.load_other(flow_params['energy_model'])

    def prepare_training(self):
        
        self.iterations = self.params['iterations']
        print(f"train_model: Beginning training. Number of iterations set to {self.iterations}")

        # initialize optimizer
        self.optimizer = torch.optim.Adam(
            [self.theta_t, self.theta_t_dot, self.theta_s, self.theta_s_dot],
            lr=self.params.get('lr', 2e-3),
            betas=self.params.get('betas', [0.9, 0.999]),
            eps=self.params.get('eps', 1e-6),
        )

        # initialize scheduler
        if self.params.get('use_scheduler', False):
            self.params['n_epochs'] = self.iterations # avoid 'n_epochs' in config file        
            self.scheduler = set_scheduler(self.optimizer, self.params)
          
        # initialize logging
        self.log = self.params.get("log", True)
        if self.log:
            log_dir = self.doc.basedir
            self.logger = SummaryWriter(log_dir)
            print(f"train_model: Logging to log_dir {log_dir}")
        else:
            print('train_model: log set to False. No logs will be written')

    def run_training(self):
        """
        Fit the solver to truth trajectories by optimising the upper bound to
        the global truncation error (Alg. 3).
        """
        
        self.prepare_training()
        cond_generator = self.condition_generator(
            self.iterations, self.params.get('batch_size', 1)
        )

        # loop over conditions
        loss_buffer = []
        loss_window = self.params.get('loss_window', 300)
        prev_window_loss = None
        for i, c in enumerate(cond_generator):
            
            # standard forward/backward passes
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.forward(c).mean()
            loss.backward()
            self.optimizer.step()
            if hasattr(self, 'scheduler'):
                self.scheduler.step()

            # log
            if self.log:
                self.logger.add_scalar('train_losses', loss.item(), i)
                if hasattr(self, 'scheduler'):
                    self.logger.add_scalar(
                        'learning_rate', self.scheduler.get_last_lr()[0], i
                    )

            # moving average loss for early stopping
            loss_buffer.append(loss.item())
            if i and not (i%loss_window):
                average = np.mean(loss_buffer)
                loss_buffer.clear()
                if average < (prev_window_loss or 1e10):
                    # save best model
                    print(
                        f'train_model: Saving best model (loss={average})',
                        flush=True
                    )
                    torch.save(
                        {'solver_params': self.state_dict()},
                        self.doc.get_file('model.pt')
                    )
                    prev_window_loss = average
                else:
                    print(f'Early stopping after {i} iterations.')
                    break

        # generate and plot samples
        if self.params.get('sample', True):
            samples, c = self.sample_n()
            self.plot_samples(samples=samples, conditions=c)     

    @torch.inference_mode()
    def sample_n(self):
        print("generate_samples: Start generating samples", flush=True)
        t_0 = time.time()

        # initialize condition generator
        n_batches = self.params.get('n_batches_sample', 10)
        batch_size = self.params.get('batch_size_sample', 10000)
        cond_generator = self.condition_generator(n_batches, batch_size)

        # sample model in batches
        conds, samples = [], []
        for c in cond_generator:
            samples.append(self.solve(c).cpu())
            conds.append(c.cpu())
        
        # report timing
        t_1 = time.time()
        sampling_time = t_1 - t_0
        self.params["sampling_time"] = sampling_time
        print(f"generate_samples: Finished generating {n_batches*batch_size} "
              f" samples after {sampling_time} s.", flush=True)

        return torch.vstack(samples), torch.vstack(conds)
    
    def load(self, epoch=None):
        self.load_state_dict(torch.load(
            self.doc.get_file(f"model{epoch or ''}.pt"),
            map_location=self.device
        )['solver_params'])

    def plot_samples(self, samples, conditions, **kwargs):
        self.flow.plot_samples(samples, conditions, doc=self.doc, **kwargs)

class BespokeEuler(BespokeSolver):
    """A concrete bespoke solver for the Euler (RK1) method."""

    def init_params(self):
        self.theta_t = torch.nn.Parameter(torch.ones(
            self.num_steps-1, requires_grad=True, device=self.device
        ))
        self.theta_t_dot = torch.nn.Parameter(torch.full(
            [self.num_steps], np.log(np.expm1(1)), requires_grad=True,
            device=self.device
        ))
        self.theta_s = torch.nn.Parameter(torch.full(
            [self.num_steps], np.log(np.expm1(1)), requires_grad=True,
            device=self.device
        ))
        self.theta_s_dot = torch.nn.Parameter(
            torch.zeros(self.num_steps, requires_grad=True, device=self.device)
        )

    def step(self, x, cond, r=None):
        """Step along the integration path (Eq. 17)"""

        if r is None: # parallel (for training)
            cast_shape = self.cast_shape
            s, s_plus, s_dot = self.s[:-1], self.s[1:], self.s_dot
            t, t_dot = self.t[:-1], self.t_dot
        else: # single (for sampling)
            cast_shape = self.cast_shape[1:]
            s, s_plus, s_dot = self.s[r], self.s[r+1], self.s_dot[r]
            t, t_dot = self.t[r], self.t_dot[r]
        xfac = ((s + self.h * s_dot) / s_plus).view(*cast_shape)
        ffac = (self.h * t_dot * s / s_plus).view(*cast_shape)

        return xfac * x + ffac * self.flow_fn(x, t, cond)

    @property
    def lipschitz(self):
        """Eqs. 48, 49"""
        s, s_plus = self.s[:-1], self.s[1:]
        L_u = self.lipschitz_u(s, self.s_dot, self.t_dot, self.L_tau)
        return s/s_plus * (1 + self.h * L_u)

    @property
    def t_sol(self):
        """Returns integer-index time steps for the truth trajectory."""
        return self.t

class BespokeMidpoint(BespokeSolver):
    """A concrete bespoke solver for the midpoint (RK2) method."""

    def init_params(self):
        self.theta_t = torch.nn.Parameter(torch.ones(
            2*self.num_steps-1, requires_grad=True, device=self.device
        ))
        self.theta_t_dot = torch.nn.Parameter(torch.full(
            [2*self.num_steps], np.log(np.expm1(1)), requires_grad=True,
            device=self.device
        ))
        self.theta_s = torch.nn.Parameter(torch.full(
            [2*self.num_steps], np.log(np.expm1(1)), requires_grad=True,
            device=self.device
        ))
        self.theta_s_dot = torch.nn.Parameter(
            torch.zeros(2*self.num_steps, requires_grad=True,
            device=self.device
        ))
        
    def step(self, x, cond, r=None):
        """Step along the integration path (Eqs. 19, 20)"""

        if r is None: # parallel (for training)
            cast_shape = self.cast_shape
            s, s_half, s_plus = self.s[:-1:2], self.s[1::2], self.s[2::2]
            t, t_half = self.t[:-1:2], self.t[1::2]
            s_dot, s_dot_half = self.s_dot[::2], self.s_dot[1::2]
            t_dot, t_dot_half = self.t_dot[::2], self.t_dot[1::2]
        else: # single (for sampling)
            cast_shape = self.cast_shape[1:]
            s, s_half, s_plus = self.s[2*r], self.s[2*r+1], self.s[2*r+2]
            t, t_half = self.t[2*r], self.t[2*r+1]
            s_dot, s_dot_half = self.s_dot[2*r], self.s_dot[2*r+1]
            t_dot, t_dot_half = self.t_dot[2*r], self.t_dot[2*r+1]

        zx = (s + self.h * s_dot / 2).view(*cast_shape)
        zf = (self.h * s * t_dot / 2).view(*cast_shape)
        z = zx * x + zf * self.flow_fn(x, t, cond)

        h_on_s = self.h / s_plus
        brace_z = (h_on_s * s_dot_half / s_half).view(*cast_shape)
        brace_u = (h_on_s * t_dot_half * s_half).view(*cast_shape)
        u_half = self.flow_fn(z/s_half.view(*cast_shape), t_half, cond)
        brace = brace_z * z + brace_u * u_half

        return brace + x * (s / s_plus).view(*cast_shape)

    @property
    def lipschitz(self):
        """Eqs. 48, 49"""
        s, s_half, s_plus = self.s[:-1:2], self.s[1::2], self.s[2::2]
        s_dot, s_dot_half = self.s_dot[::2], self.s_dot[1::2]
        t_dot, t_dot_half = self.t_dot[::2], self.t_dot[1::2]
        L_u      = self.lipschitz_u(     s,      s_dot,      t_dot, self.L_tau)
        L_u_half = self.lipschitz_u(s_half, s_dot_half, t_dot_half, self.L_tau)
        return s/s_plus * (1 + self.h * L_u_half * (1 + self.h * L_u / 2))

    @property
    def t_sol(self):
        """Returns integer-index time steps for the truth trajectory."""
        return self.t[::2]