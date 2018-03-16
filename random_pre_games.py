from game import *
import numpy as np
import sys
import h5py

pre_game_num = int(sys.argv[1])
output = sys.argv[2]

# List to store the lengths of the games
lengths = []

for i in range(pre_game_num):
    games, scores, actions = test_play(visualize=False)
    with h5py.File("data/"+output,'a') as f:
        lengths += [len(games)-1]
        for j in range(lengths[-1]):
            # Counts the number of games already in the folder
            # Name the training state
            name = "game_"+str(i+1)+"_play_"+str(j+1)
            # Add the games to the hdf5 file
            f[name] = np.stack([games[j],games[j+1]])
            f[name].attrs["score"] = scores[j]
            f[name].attrs["new_score"] = scores[j+1]
            f[name].attrs["action"] = actions[j]
            if j == len(games)-2:
                f[name].attrs["running"] = False
            else:
                f[name].attrs["running"] = True

    if i%100==0:
        print("Done with plays: ",i)

# Add the lengths as an attribe of the file
with h5py.File("data/"+output,'a') as f:
    f["lengths"] = lengths