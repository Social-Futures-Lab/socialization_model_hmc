#!/bin/bash

#SBATCH -p ckpt-all
#SBATCH --job-name=hmc_socialization_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gpus=l40:2
#SBATCH --time=09:00:00

# setup script for checkpointing

/gscratch/comdata/users/vkoshy/numpyro-cuda/bin/python3 hmc_socialization.py --input_dir /gscratch/comdata/users/vkoshy/socialization_data/inputs/simulation/ --output_dir /gscratch/comdata/users/vkoshy/socialization_data/results/simulation/ --checkpoint_dir /gscratch/comdata/users/vkoshy/hmc_checkpoints/unpooled --topics 10 --samples 25 --warmup 25 --alpha_sum_topics 0.01 --alpha_sum_vocab 0.01 --alpha_edges 0.01 --model_name "gammas_unpooled" --device "cuda" --num_chains 2

