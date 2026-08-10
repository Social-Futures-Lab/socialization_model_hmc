#!/bin/bash
# usage: ./launch.sh 2020_1 2021_12
set -euo pipefail

start=$1; end=$2
sy=${start%_*}; sm=${start#*_}
ey=${end%_*};   em=${end#*_}

# convert to absolute month index so the loop is trivial
si=$((10#$sy * 12 + 10#$sm - 1))
ei=$((10#$ey * 12 + 10#$em - 1))

mkdir -p logs
for i in $(seq $si $ei); do
    period="$((i / 12))_$((i % 12 + 1))"
    sbatch --job-name="hmc_$period" \
           --output="outputs/${period}.out" \
           covid.slurm "$period"
done
