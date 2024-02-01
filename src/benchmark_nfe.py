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

def benchmark(args):

    assert (args.steps is None) ^ (args.tols is None), \
        "Exactly one of `steps` or `tols` must be set!"
    
    device = 'cuda' if torch.cuda.is_available() and ~args.no_cuda else 'cpu'
    
    precision = args.tol if args.solver in ADAPTIVE_SOLVERS else args.steps
    doc = Documenter(f'benchmark_{args.solver}_{precision}')

    # load shape and energy models
    models = {}
    for model_type in 'energy', 'shape':
        model_dir = getattr(args, model_type+'_model')
        params = load_params(os.path.join(model_dir, 'params.yaml'))
        params['eval_mode'] = args.eval_mode
        model = Models.TBD(params, device=device,
            doc=Documenter(None, existing_run=model_dir, read_only=True))
        # set to eval mode and freeze weights
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        models[model_type] = model

    for _ in range(args.n_runs):
        
        # generate condition
        Einc = torch.rand([args.n_samples, 1], device=device) # Assuming u_model expects Einc uniform in [0,1]
        with torch.no_grad():
            u_samples = models['energy'].sample_batch(Einc).to(device) # TODO: Update sample batch so that it doesn't move to cpu!
        cond = torch.cat([Einc, u_samples], dim=1)

        # dispatch to chosen solver and generate sample
        if args.solver in BESPOKE_SOLVERS:
            solver = getattr(Models, args.solver)(
                params=load_params(os.path.join(args.bespoke_dir, 'params.yaml')),
                doc=Documenter(None, existing_run=args.bespoke_dir, read_only=True),
                device=device
            )
            sample = solver.solve(cond).cpu()
        else:
            # define wrapper function to pass to solvers
            def flow_fn(t, x):
                t = t.repeat((x.shape[0],1)).to(device)
                return models['shape'].net(x, t, cond)
    
            y0 = torch.randn((args.n_samples, *models['shape'].shape),
                    device=device)
            sol_times = torch.tensor([0, 1], dtype=torch.float32).to(device)
            solver_kwargs = (
                {'options': {'step_size': 1/args.steps}} 
                if args.solver in FIXED_SOLVERS else
                {'atol': args.tol, 'rtol': args.tol}
            )
            with torch.no_grad():
                sample = odeint(flow_fn, y0, sol_times, method=args.solver,
                    **solver_kwargs)[-1]
        
        # post-process
        for fn in models['shape'].transforms[::-1]:
            sample, cond = fn(sample, cond, rev=True)
        sample = sample.detach().cpu().numpy()
        cond = cond.detach().cpu().numpy()

        # classify
        evaluate.run_from_py(sample, cond, doc, models['shape'].params)

if __name__ == '__main__':
    benchmark(args)
    # if len(args.steps) > 1: # submit to queue
    #     use_gpu = int('gpu' in args.queue)
    #     setup_cmd = 'ml purge; source ~/setup_ml.sh; source ~/venvs/ml/bin/activate'
    #     script_cmd = f'python ' + ' '.join(sys.argv)
    #     steps_str = ' '.join(map(str, args.steps))
    #     for step in args.steps:
    #         cmd = (
    #             f"sbatch -p {args.queue} --mem {args.memory} -N 1 -c 4 --gpus {int(use_gpu)}"
    #             f" -t {args.time} -J [}] --wrap \"{setup_cmd}; {script_cmd}\""
    #             f" -o {log_dir}/%x.out "
    #         )
    #     # script_cmd.replace()

    #     print(args._get_args())
    # else:
          

        