#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=2048
#SBATCH --gres=gpu:1

# running
module load keras/2.1.4
python3 model.py 500 10000 onehot 10000_onehot.hdf5 &> $SLURM_JOB_ID.out
