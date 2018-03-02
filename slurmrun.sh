#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --job-name=2048
#SBATCH --gres=gpu:1

# running
module load keras/2.1.4
python3 model.py 100000 onehot 100000_onehot.hdf5
