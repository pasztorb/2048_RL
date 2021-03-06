from shared_functions import *
from game import *
import sys
import h5py
import numpy as np
import time
from collections import deque


"""
This python file initialize a network and pre-trains it for the given data
python3 pre_train.py network_type reshape_type gamma id input_data
"""

network_type = sys.argv[1]
assert network_type in ['flat', 'conv']
reshape_type = sys.argv[2]
assert reshape_type in ['onehot', 'linear', 'trig']
gamma = float(sys.argv[3])
run_id = sys.argv[4]
input_data = sys.argv[5]
print("Reshape_type: ", reshape_type)
print("Network type: ", network_type)
print("Gamma: ", gamma)
print("Run id: ", run_id)
print("Input data: ", input_data)

batch_size = 32
buffer = 1000000
print("Batch size: ",batch_size)
print("Buffer: ", buffer)

"""
Initializing the neural network
"""
# Choose reshape function and game_shape_after_reshaping based on reshape_type
if reshape_type == 'onehot':
    if network_type == 'conv':
        reshape_function = onehot_reshape
    else:
        reshape_function = lambda x: onehot_reshape(x, flat=True)
    game_shape_after_reshaping = (17, 4, 4)
elif reshape_type == 'linear':
    if network_type == 'conv':
        reshape_function = linear_reshape
    else:
        reshape_function = lambda x: linear_reshape(x, flat=True)
    game_shape_after_reshaping = (1, 4, 4)
elif reshape_type == 'trig':
    if network_type == 'conv':
        reshape_function = trig_reshape
    else:
        reshape_function = lambda x: trig_reshape(x, flat=True)
    game_shape_after_reshaping = (3, 4, 4)


# Initialize model, if reshape_style is 'flat' initialize a feed-forward net otherwise a convolutional
if network_type == 'conv':
    model = init_conv_model(game_shape_after_reshaping)
else:
    model = init_flat_model(np.prod(game_shape_after_reshaping))


"""
Pre-training
"""
print("Pre-training...")
# Replay memory
pre_memory = deque(maxlen=buffer) # Replay storage
training_steps = 0 # Counter for the training steps
start_time = time.time()

with h5py.File(input_data, 'a') as f:
    lengths = f['lengths']
    # Iterate over the lengths list
    for i, length in enumerate(lengths):
        # For each game iterate over the states
        for j in range(length):
            # Name of the training state
            name = "game_" + str(i + 1) + "_play_" + str(j + 1)

            # Retrive the data for the game
            state = f[name][0,:,:]
            new_state = f[name][1,:,:]
            score = f[name].attrs["score"]
            new_score = f[name].attrs["new_score"]
            action = f[name].attrs["action"]
            running = f[name].attrs["running"]

            # Caculate the reward
            reward = getReward(state, new_state, score, new_score, running)

            # Add the current state to the experience replay storage
            pre_memory.append((state, action, reward, new_state, running))

            if buffer <= len(pre_memory):
                model = replay_train(reshape_function, model, pre_memory, batch_size, gamma)
                training_steps += 1
        # Print feedback and evaluate model
        if (i+1)%250 == 0:
            print("Pre-trained for: ",i+1," games")
            score_list = avg_test_plays(50, model=model, reshape_function=reshape_function)
            print("Average, min and max of the test scores: ", np.mean(score_list), min(score_list), max(score_list))

# Play one game before starting the training
print("Number of training steps: ", training_steps)
print("Time of the training: ", time.time()-start_time)
print("First game after pre-training...")
test_play(model=model, reshape_function=reshape_function)

"""
Save the pre-trained network
"""
# Save trained model
model.save(network_type+'_'+reshape_type+"_model_"+run_id+".hdf5")