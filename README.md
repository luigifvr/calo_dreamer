# CaloDREAM

A repository for fast detector simulation using Conditional Flow Matching
on the CaloChallenge datasets

## Usage

Example 1 (minimal):

Run the dataset 2 model with default settings
```
python3 sample.py models/d2
```

Options:
The following arguments can optionally be specified
Flag		| Usage
------------------------| ----------------------------------------
energy_model  | Directory containing config and checkpoint for the energy model. `models/energy` is used by default.
sample_size   | The number of samples to generate \[default 100,000\]
batch_size    | The batch size used for sampling \[default 5,000\]
use_cpu (flag)| Whether to run on cpu
which_cuda    | Index of the cuda device to use \[default 0\]

Example 2:

Run the dataset 3 model on cpu for specific sample and batch sizes
```
python3 sample.py models/d3 --sample_size 10000 --batch_size 100 --use_cpu
```
