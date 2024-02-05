#!/usr/bin/env python

"""A script to benchmark the performance/efficiency tradeoff of different
   ODE samplers for generating calorimeter showers."""

import Models
import os
# import sys
import torch
from argparse import ArgumentParser
from challenge_files import evaluate
from documenter import Documenter
from torchdiffeq import odeint
from Util.util import load_params

FIXED_SOLVERS = ['euler', 'midpoint']
ADAPTIVE_SOLVERS = ['dopri5']
BESPOKE_SOLVERS = ['BespokeEuler', 'BespokeMidpoint']
solver_choices = FIXED_SOLVERS + ADAPTIVE_SOLVERS + BESPOKE_SOLVERS

parser = ArgumentParser()
parser.add_argument('--shape_model', required=True)
parser.add_argument('--energy_model', required=True)
parser.add_argument('--solver', choices=solver_choices, required=True)
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--bespoke_dir')
parser.add_argument('--steps', type=int)
parser.add_argument('--tols', type=float)
parser.add_argument('--n_samples', type=int, default=5000)
parser.add_argument('--n_runs', type=int, default=20)
parser.add_argument('--eval_mode', default='cls-high')
# parser.add_argument('-q', '--queue', default='gpu-a100-short')
# parser.add_argument('-t', '--time', default=120)
# parser.add_argument('-m', '--memory', default='32G')
args = parser.parse_args()

class SolveFunc:

    def __init__(self, net, cond, device):
        self.net = net
        self.cond = cond
        self.device = device
        self.nfe = 0

    def __call__(self, t, x):
        self.nfe += 1
        t = t.repeat((x.shape[0],1)).to(self.device)
        return self.net(x, t, self.cond)

def benchmark(args):

    assert (args.steps is None) ^ (args.tols is None), \
        "Exactly one of `steps` or `tols` must be set!"
    
    device = 'cuda' if torch.cuda.is_available() and ~args.no_cuda else 'cpu'
    precision = args.tols if args.solver in ADAPTIVE_SOLVERS else args.steps
    doc = Documenter(f'benchmark_{args.solver}_{precision}')

    # load shape and energy models
    models = {}
    for model_type in 'energy', 'shape':
        model_dir = getattr(args, model_type+'_model')
        params = load_params(os.path.join(model_dir, 'params.yaml'))
        params['eval_mode'] = args.eval_mode
        model = Models.TBD(params, device=device,
            doc=Documenter(None, existing_run=model_dir, read_only=True))
        model.load()
        # set to eval mode and freeze weights
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        models[model_type] = model

    for _ in range(args.n_runs):
        
        # generate condition
        Einc = torch.rand([args.n_samples, 1], device=device) # Assuming u_model expects Einc uniform in [0,1]
        with torch.no_grad():
            u_samples = models['energy'].sample_batch(Einc)
        cond = torch.cat([Einc, u_samples], dim=1)

        # dispatch to chosen solver and generate sample
        if args.solver in BESPOKE_SOLVERS:
            solver = getattr(Models, args.solver)(
                params=load_params(os.path.join(args.bespoke_dir, 'params.yaml')),
                doc=Documenter(None, existing_run=args.bespoke_dir, read_only=True),
                device=device
            )
            solver.load()
            sample = solver.solve(cond)
        else:
            solve_fn = SolveFunc(models['shape'].net, cond, device)            
            y0 = torch.randn((args.n_samples, *models['shape'].shape),
                    device=device)            
            times = torch.tensor([0, 1], dtype=torch.float32, device=device)
            solver_kwargs = (
                {'options': {'step_size': 1/args.steps}}
                if args.solver in FIXED_SOLVERS else
                {'atol': args.tols, 'rtol': args.tols}
            )
            with torch.no_grad():
                sample = odeint(solve_fn, y0, times, method=args.solver,
                    **solver_kwargs)[-1]
            # model.params['solver_kwargs'] = (
            #     {'method': args.solver, 'options': {'step_size': 1/args.steps}} 
            #     if args.solver in FIXED_SOLVERS else
            #     {'method': args.solver, 'atol': args.tol, 'rtol': args.tol}
            # )
            # sample = models['shape'].sample_batch(cond)
        
        # post-process
        for fn in models['shape'].transforms[::-1]:
            sample, cond = fn(sample, cond, rev=True)
        sample = sample.detach().cpu().numpy()
        cond = cond.detach().cpu().numpy()

        # classify
        evaluate.run_from_py(sample, cond, doc, models['shape'].params)

        # TODO: Put this _before_ evaluate so that NFE is prepended...
        if args.solver in ADAPTIVE_SOLVERS:
            with open(doc.get_file('eval/classifier_cls-high_2.txt'), 'a') as f:
                f.write(f"NFE: {solve_fn.nfe}\n")
        
if __name__ == '__main__':
    benchmark(args)

        