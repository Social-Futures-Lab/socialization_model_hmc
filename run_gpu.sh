rm /gscratch/comdata/users/vkoshy/hmc_checkpoints/pooled/*
rm /gscratch/comdata/users/vkoshy/hmc_checkpoints/unpooled/*
rm /gscratch/comdata/users/vkoshy/hmc_checkpoints/no_topics/*
sbatch checkpointed_hmc_gpu.slurm
