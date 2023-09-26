# CaloDream

A repository for fast detector simulation using Stable Diffusion
on the CaloChallenge datasets.

Two recent papers using diffusion models:

**CaloScore** (Ben Nachman, Vinicius Mikuni)
[https://arxiv.org/abs/2206.11898](https://arxiv.org/abs/2206.11898)

**CaloDiffusion** (Oz Amram, Kevin Pedro)
[https://arxiv.org/abs/2308.03876](https://arxiv.org/abs/2308.03876)

## Usage

Training:
```
python3 src/main.py --use_cuda path/to/yaml
```

The documenter will create a folder in `results` with the date as
prefix and the specified `run_name`.

### Parameters

Parameter		| Usage
------------------------| ----------------------------------------
run\_name		| Name of the output folder
hdf5\_file		| Path to the .hdf5 file used for training
xml\_filename		| Path to the .xml file used to extract the binning information
p\_type 		| "photon", "pion", or "electron"
dtype			| specify default dtype
eval\_dataset		| "1-photons", "1-pions", "2", or "3" used in the CaloChallenge evaluation

### Training parameters

Parameter 		| Usage
------------------------| ----------------------------------------
dim			| Dimensionality of the input
n\_con			| Number of conditions
width\_noise		| Noise width used for the noise injection
val\_frac		| Fraction of events used for validation
transforms		| Pre-processing steps defined as an ordered dictionary
lr			| learning rate
max\_lr			| Maximum learning rate for OneCycleLR scheduling
batch\_size		| batch size
validate\_every		| Interval between validations in epochs
use\_scheduler 		| True or False
lr\_scheduler		| string that defines the learning rate scheduler
cycle\_epochs		| defines the length of the cycle for the OneCycleLR, default to # of epochs
save\_interval		| Interval between each model saving in epochs
n\_epochs		| Number of epochs
alpha			| Regularizer for the log transformation

### ResNet parameters

Parameter		| Usage
------------------------|----------------------------------------
intermediate\_dim	| Dimension of the intermediate layer
layers\_per\_block	| Number of layers per block
n\_blocks		| Number of blocks
conditional		| True/False, it should be always True

An example yaml file is provided in `./configs/cfm_base.yaml`.

Plotting:

To run the sampling and the evaluation of a trained model.
```
python3 src/main.py --use_cuda --plot --model_dir path/to/model --epoch model_name
```

