#!/bin/bash

#SBATCH -p ckpt-all
#SBATCH --job-name=hmc_socialization_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=0:30:00
#SBATCH --output=viz.out
# setup script for checkpointing

/gscratch/comdata/users/vkoshy/numpyro-cuda/bin/python3 visualize_single_horseshoe.py /gscratch/comdata/users/vkoshy/socialization_data/inputs/hp_weird/ /gscratch/comdata/users/vkoshy/socialization_data/results/hp_weird/ outputs/hp 

