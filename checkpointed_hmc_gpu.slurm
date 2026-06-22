#!/bin/bash

#SBATCH -p ckpt-all
#SBATCH --job-name=hmc_socialization_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gpus=l40:4
#SBATCH --time=5:00:00

# setup script for checkpointing

/gscratch/comdata/users/vkoshy/numpyro-cuda/bin/python3 hmc_socialization.py --input_dir /gscratch/comdata/users/vkoshy/socialization_data/inputs/hp_smaller_plus/ --output_dir /gscratch/comdata/users/vkoshy/socialization_data/results/hp_smaller_plus/ --checkpoint_dir /gscratch/comdata/users/vkoshy/hmc_checkpoints/no_topics --topics 100 --samples 800 --warmup 500 --alpha_sum_topics 100 --alpha_sum_vocab 2400 --alpha_edges 1  --model_name "no_topics_multinomial" --device "cuda" --num_chains 4

