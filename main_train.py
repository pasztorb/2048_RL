import numpy as np
import sys
import h5py
import time
import datetime
from collections import deque

from game import *
from shared_functions import *

from keras.models import load_model

# Variables given in command lines
# Sample call: python3 main_train.py 10000 input_model.hdf5

train_count = int(sys.argv[1])
input_model_path = sys.argv[2]
model_name = input_model_path.split(sep='/')[-1]

network_type = model_name.split(sep='_')[0]
assert network_type in ['conv', 'flat']
reshape_type = model_name.split(sep='_')[1]
assert reshape_type in ['onehot', 'linear', 'trig']
run_id = model_name.split(sep='_')[-1][:-5]

print("Network type: ", network_type)
print("Reshape type: ", reshape_type)
print("Run ID:", run_id)

# Fixed variables
gamma = 0.9
epsilon = 1
batch_size = 32
buffer = 200000
test_freq = 500

print("Train count: ", train_count)
print("Reshape type: ",reshape_type)
print("Batch size and buffer: ", batch_size, buffer)

"""
Loading the model and prepare for training
"""
# Choose reshape function and game_shape_after_reshaping based on reshape_type
if reshape_type == 'onehot':
    if network_type == 'conv':
        reshape_function = onehot_reshape
    else:
        reshape_function = lambda  x: onehot_reshape(x, flat=True)
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

# Load in the model
model = load_model(input_model_path)

"""
Training
"""

# Variables for storage during training
memory = deque(maxlen=buffer) # Replay storage
train_scores = [] # Scores at the end of the training games
test_scores_avg = [] # Average scores of the test episodes
test_scores_min = [] # Minimum scores of the test episodes
test_scores_max = [] # Maximum scores of the test episodes

# Play one game before starting the training
print("First game after pre-training...")
test_play(model=model, reshape_function=reshape_function)

# Set counter and epoch to zero
count = 0
epoch = 0
epsilon = 1

# Looping over the training counts
while count < train_count:
    print("Epoch number: ", str(epoch+1))
    print("Epsilon value: ", str(epsilon))

    # Initialize game
    game, score = initialize_game()

    # Play training game
    running = True
    while running:
        # Choose if the next action is random or not
        if (np.random.random() < epsilon):
            action = np.random.randint(0, 4)
        else:
            # Evaluate on the current state Q(s_i) and take accordingly
            qval = model.predict(reshape_function(game), batch_size=1)
            action = np.argmax(qval)

        # Make a move
        new_game, new_score, running  = action_move(game, score, action)

        # Observe Reward
        reward = getReward(game, new_game, score, new_score, running)

        # Add the current state to the experience replay storage
        memory.append((game, action, reward, new_game, running))

        if buffer <= len(memory):
            model = replay_train(reshape_function, model, memory, batch_size, gamma)
            # Increase count
            count += 1
            # Decrease epsion if it is over the pre-training count
            if epsilon > 0.1:
                epsilon -= 2 / train_count

        # Update game and score variables
        game = new_game
        score = new_score

    # Add one to epoch number
    epoch += 1

    # Store and print score at the end of the training game
    train_scores += [score]
    print("Score at the end of the game %s: " % (epoch), str(score))
    print("Max tile of the training game: ", str(game.max()))

    # If test_num games have passed play test games
    if ((epoch % test_freq) == 0) and (buffer <= len(memory)):
        print("Running test plays after %s games." %(epoch))
        print("Number of training steps done: ", count)
        # Test play
        score_list = avg_test_plays(20, model=model, reshape_function=reshape_function)
        print("Average, min and max of the test scores: ", np.mean(score_list), min(score_list), max(score_list))
        # Store test statistics
        test_scores_avg += [np.mean(score_list)]
        test_scores_min += [min(score_list)]
        test_scores_max += [max(score_list)]
        print("Maximum average score after Game %s: " % (epoch + 1,), str(max(test_scores_avg)))


# Test play after last train
_, score_list, _ = test_play(model=model, reshape_function=reshape_function)
print("Maximum average score after Game %s: " % (epoch + 1,), str(max(test_scores_avg)))


# Write out the generated data and the statistics into the hdf5 file given as the output path
# name is given as the training timestamp
name = datetime.datetime.fromtimestamp(
    int(time.time())
).strftime('%Y-%m-%d_%H:%M:%S')

with h5py.File("train_statistics_"+run_id+".hdf5", 'a') as f:
    f[name] = train_scores
    f[name].attrs['test_scores_avg'] = test_scores_avg
    f[name].attrs['test_score_min'] = test_scores_min
    f[name].attrs['test_score_max'] = test_scores_max
    f[name].attrs['gamma'] = gamma
    f[name].attrs['batch_size'] = batch_size
    f[name].attrs['buffer'] = buffer
    f[name].attrs['train_count'] = train_count
    f[name].attrs['reshape_type'] = reshape_type
    f[name].attrs['network_type'] = network_type
    f[name].attrs['test_freq'] = test_freq


# Save trained model
model.save(reshape_type+"_trainedmodel_"+run_id+".hdf5")



"""
# Plot the train_scores and test_scores and save it in a .png file
plt.plot(range(1, epochs+1), train_scores, label='Train scores')
plt.plot(list(range(1, epochs+1, epochs//test_num)), test_scores, label='Test scores averages')
plt.legend(loc='upper left')
plt.title("Training on "+str(epochs)+" games")

plt.savefig(output_path)
"""