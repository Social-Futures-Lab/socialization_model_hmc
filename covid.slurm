#!/bin/bash
#SBATCH -p ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gpus=l40:4
#SBATCH --time=05:00:00
# job-name and output are set by the launcher

set -euo pipefail
PERIOD=$1
BASE=/gscratch/comdata/users/vkoshy

$BASE/numpyro-cuda/bin/python3 hmc_socialization.py \
    --input_dir      $BASE/socialization_data/inputs/$PERIOD/ \
    --output_dir     $BASE/socialization_data/results/$PERIOD \
    --checkpoint_dir $BASE/hmc_checkpoints/$PERIOD \
    --samples 500 --warmup 500 --checkpoint_interval 50 \
    --model_name "no_topics_multinomial" --device "cuda" --num_chains 4
