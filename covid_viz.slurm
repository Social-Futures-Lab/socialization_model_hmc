#!/bin/bash

#SBATCH -p ckpt-all
#SBATCH --job-name=hmc_socialization_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=0:30:00
#SBATCH --output=covid_viz.out
# setup script for checkpointing

/gscratch/comdata/users/vkoshy/numpyro-cuda/bin/python3 visualize_time_series.py 2020_1 2021_12 /gscratch/comdata/users/vkoshy/socialization_data/inputs/ /gscratch/comdata/users/vkoshy/socialization_data/results/ outputs/covid_viz

