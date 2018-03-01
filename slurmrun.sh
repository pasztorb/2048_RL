#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --job-name=RL2048test
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user:pasztorb@me.com

# running
module load keras/2.1.4
python3 model.py 100000 onehot server_try_onehot.hdf5
