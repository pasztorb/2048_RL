#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=pre_flat
#SBATCH --gres=gpu:1

# running
module load keras/2.1.4
python3 pre_train.py flat $SLURM_JOB_ID data/random_5000.hdf5
