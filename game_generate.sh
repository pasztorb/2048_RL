#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=game_gen
#SBATCH --gres=gpu:1

# running
module load keras/2.1.4
python3 random_pre_games.py 3000 random_3000.hdf5
