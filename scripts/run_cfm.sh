#!/bin/bash
#PBS -N det_inn_8
#PBS -l walltime=40:00:00
#PBS -l nodes=1:ppn=1:gpus=1:a30
#PBS -q a30

module load cuda/11.4
source activate /remote/gpu06/favaro/.conda/pytorch
mydev=`cat $PBS_GPUFILE | sed s/.*-gpu// `
export CUDA_VISIBLE_DEVICES=$mydev
cd /remote/gpu06/favaro/calo_dreamer/
python3 src/main.py /remote/gpu06/favaro/calo_dreamer//configs/cfm_base.yaml -c
