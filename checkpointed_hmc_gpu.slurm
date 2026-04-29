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
checkpoint_interval=300 # number of seconds between checkpoints
module load testing/dmtcp/3.0.0
export DMTCP_CHECKPOINT_DIR=/gscratch/comdata/users/vkoshy/checkpoints/
export DMTCP_DL_PLUGIN=0 # silences some errors

if [ -n "$(ls $DMTCP_CHECKPOINT_DIR/*.dmtcp 2>/dev/null)" ]; then
    # checkpoints exist, restart
    dmtcp_restart --interval $checkpoint_interval $DMTCP_CHECKPOINT_DIR/*.dmtcp
else
    # no checkpoints, fresh start
    mkdir -p "$DMTCP_CHECKPOINT_DIR"
    # replace with your research program!
    dmtcp_launch --interval $checkpoint_interval /gscratch/comdata/users/vkoshy/numpyro-cuda/bin/python3 hmc_socialization.py --input_dir /gscratch/comdata/users/vkoshy/socialization_data/inputs/hp/ --output_dir /gscratch/comdata/users/vkoshy/socialization_data/results/hp/ --topics 100 --samples 100 --warmup 100 --alpha_sum_topics 0.01 --alpha_sum_vocab 0.01 --alpha_edges 0.01 --model_name "gammas_pooled" --device "cuda"
fi

