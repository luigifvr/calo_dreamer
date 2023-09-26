import argparse
import os
import shutil
import yaml
import torch

from documenter import Documenter
from Models import *
from Models.tbd import TBD
from challenge_files import evaluate

def main():
    parser = argparse.ArgumentParser(description='Fast Calorimeter Simulation with CaloDreamer')
    parser.add_argument('param_file', help='yaml parameters file')
    parser.add_argument('-c', '--use_cuda', action='store_true', default=False,)
    parser.add_argument('-p', '--plot', action='store_true', default=False,)
    parser.add_argument('-d', '--model_dir', default=None,)
    parser.add_argument('-ep', '--epoch')

    args = parser.parse_args()

    with open(args.param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    use_cuda = torch.cuda.is_available() and args.use_cuda

    device = 'cuda:0' if use_cuda else 'cpu'
    print('device: ', device)

    if args.plot:
        doc = Documenter(params['run_name'], existing_run=args.model_dir)
    else:
        doc = Documenter(params['run_name'])

    try:
        shutil.copy(args.param_file, doc.get_file('params.yaml'))
    except shutil.SameFileError:
        pass
 
    dtype = params.get('dtype', '')
    if dtype=='float64':
        torch.set_default_dtype(torch.float64)
    elif dtype=='float16':
        torch.set_default_dtype(torch.float16)
    elif dtype=='float32':
        torch.set_default_dtype(torch.float32)

    model = TBD(params, device, doc)
    if not args.plot:
        model.run_training()
    else:
        model.load(args.epoch)
        x, c = model.sample_n()
        model.plot_samples(x, c, name=f"{args.epoch}")
        #run plotting script
        # Ayo: TODO: Remove since this already runs during `plot_samples` in `model.run_training`
        evaluate.main(f"-i {doc.basedir}/samples{args.epoch}.hdf5 -r {params['hdf5_file']} -m all -d {params['eval_dataset']} --output_dir {doc.basedir}/final/ --cut 0.0".split())

if __name__=='__main__':
    main()

   
