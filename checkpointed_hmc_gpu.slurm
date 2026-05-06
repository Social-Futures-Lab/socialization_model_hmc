#!/bin/bash

#SBATCH -p ckpt-all
#SBATCH --job-name=hmc_socialization_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gpus=l40:1
#SBATCH --time=12:00:00

# setup script for checkpointing

/gscratch/comdata/users/vkoshy/numpyro-cuda/bin/python3 hmc_socialization.py --input_dir /gscratch/comdata/users/vkoshy/socialization_data/inputs/hp/ --output_dir /gscratch/comdata/users/vkoshy/socialization_data/results/hp/ --checkpoint_dir /gscratch/comdata/users/vkoshy/hmc_checkpoints --topics 100 --samples 100 --warmup 100 --alpha_sum_topics 0.01 --alpha_sum_vocab 0.01 --alpha_edges 0.01 --model_name "gammas_pooled" --device "cuda"

