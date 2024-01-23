import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from abc import abstractmethod
    
class BespokeSolver(nn.Module):

    """
    A base class for implementing bespoke solvers from arxiv:2310.19075. The
    model parameterises a transformation of the integration path of an ode and
    optimizes it such that solutions match that of an expensive solver.

    __init__ params:
        
        func       -- The vector field to be integrated. It should have
                      signature (x,t,c) --> x, where c is a possible condition.
        num_steps  -- The number of integration steps to take.
        shape      -- The shape of the state x
        L_tau      -- The Lipschitz contant hyperparameter # TODO: explain better
        truth_tols -- Settings for the atol and rtol parameters of the truth
                      solver, passed as a dictionary.
        device     -- The device on which to store model parameters. This
                      should match the device of any neural networks called by
                      `func`.
    """

    def __init__(self, func, num_steps, shape, L_tau=1., truth_tols=None,
                 device=None):
        
        super().__init__()
        
        self.func = func
        self.num_steps = num_steps
        self.shape = shape
        self.L_tau = L_tau
        self.truth_tols = truth_tols or {}
        self.device = device
        
        self.init_params()
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

    def rmse_bound(self, x, cond=None):

        # Eq. 24
        d = torch.sqrt(torch.mean(
            (x[1:] - self.step(x[:-1], cond))**2,
            dim=list(range(2, x.ndim))
        )) 

        # Eq. 25
        m = torch.cumprod(self.lipschitz, dim=0)
        
        return (m*d.T).sum(dim=1)

    def forward(self, cond=None, batch_size=None):
        """
        Solve a truth trajectory and return a bound on the truncation error of
        the bespoke solver.
        """
        if cond is not None:
            batch_size = cond.shape[0]
            
        f = lambda t, x: self.func(x, t, cond)
        x0 = torch.randn(
            (batch_size, *self.shape), device=self.device
        )
        with torch.no_grad():
            t_stop = self.t_sol.detach()
            x_true = odeint(f, x0, self.t_sol, **self.truth_tols)
            vel =  f(self.t_sol, x_true)
        x_aux = x_true + vel*(self.t_sol - t_stop).view(*self.cast_shape)
        
        return self.rmse_bound(x_aux, cond)

    def optimize(self, optimizer, iterations=None, batch_size=None,
                 cond_generator=None, print_every=0):
        """
        Fit the solver to truth trajectories by optimising the upper bound to
        the local truncation error (Alg. 3).

        parameters:
            optimizer  -- A torch optimizer for the model parameters.
            iterations -- The number of optimization steps to take
            batch_size --
        """

        loss_list = []
        if cond_generator is None:
            cond_generator = iter([None] * iterations)
            
        for i in range(iterations):

            # generate condition
            cond = next(cond_generator)

            optimizer.zero_grad()
            loss = self.forward(cond, batch_size).mean()
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.detach().cpu().numpy())
            if print_every and not i%print_every:
                print(f'Step {i+1}: {np.mean(loss_list[-print_every:]):.5f}')
        
        return loss_list

    @torch.no_grad()
    def solve(self, batch_size=None, cond=None, x0=None):
        """Alg. 2 and Eq. 17"""
        
        if x0 is None:
            x0 = torch.randn((batch_size, *self.shape), device=self.device)
        
        with torch.no_grad():
            x_next = x0
            for i in range(self.num_steps):
                x_next = self.step(x_next, cond, r=i)
                
        return x_next

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
            s, s_plus, s_dot = self.s[:-1], self.s[1:], self.s_dot
            t, t_dot = self.t[:-1], self.t_dot
        else: # single (for sampling)
            s, s_plus, s_dot = self.s[r], self.s[r+1], self.s_dot[r]
            t, t_dot = self.t[r], self.t_dot[r]
        xfac = ((s + self.h * s_dot) / s_plus).view(*self.cast_shape)
        ffac = (self.h * t_dot * s / s_plus).view(*self.cast_shape)

        return xfac * x + ffac * self.func(x, t, cond)

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
            s, s_half, s_plus = self.s[:-1:2], self.s[1::2], self.s[2::2]
            t, t_half, t_plus = self.t[:-1:2], self.t[1::2], self.t[2::2]
            s_dot, s_dot_half = self.s_dot[::2], self.s_dot[1::2]
            t_dot, t_dot_half = self.t_dot[::2], self.t_dot[1::2]
        else: # single (for sampling)
            s, s_half, s_plus = self.s[2*r], self.s[2*r+1], self.s[2*r+2]
            t, t_half, t_plus = self.t[2*r], self.t[2*r+1], self.t[2*r+2]
            s_dot, s_dot_half = self.s_dot[2*r], self.s_dot[2*r+1]
            t_dot, t_dot_half = self.t_dot[2*r], self.t_dot[2*r+1]

        zx = (s + self.h * s_dot / 2).view(*self.cast_shape)
        zf = (self.h * s * t_dot / 2).view(*self.cast_shape)
        z = zx * x + zf * self.func(x, t, cond)

        h_on_s = self.h / s_plus
        brace_z = (s_dot_half * h_on_s / s_half).view(*self.cast_shape)
        brace_u = (t_dot_half * s_half * h_on_s).view(*self.cast_shape)
        u_half = self.func(z/s_half.view(*self.cast_shape), t_half, cond)
        brace = brace_z * z + brace_u * u_half

        return brace + x * (s / s_plus).view(*self.cast_shape)

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