#!/bin/bash

#SBATCH -p ckpt-all
#SBATCH --job-name=hmc_socialization_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gpus=l40:4
#SBATCH --time=09:00:00

# setup script for checkpointing

/gscratch/comdata/users/vkoshy/numpyro-cuda/bin/python3 hmc_socialization.py --input_dir /gscratch/comdata/users/vkoshy/socialization_data/inputs/hp_weird/ --output_dir /gscratch/comdata/users/vkoshy/socialization_data/results/hp_weird/ --checkpoint_dir /gscratch/comdata/users/vkoshy/hmc_checkpoints/no_topics --samples 100 --warmup 100 --checkpoint_interval 100  --model_name "no_topics_multinomial" --device "cuda" --num_chains 4

