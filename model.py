import numpy as np
import sys
import h5py
import time
import datetime
from collections import deque
import random

from game import *
from reshape_and_NN import *

# Variables given in command lines
# Sample call: python3 model.py 1000 10000 onehot output_file.hdf5
pre_train_count = int(sys.argv[1])
train_count = int(sys.argv[2])
reshape_type = sys.argv[3]
assert reshape_type in ['onehot', 'linear', 'trig', 'flat']
output_path = sys.argv[4]

# Fixed variables
gamma = 0.9
epsilon = 1
batch_size = 16
buffer = 1000
test_num = 50

"""
Reward function, and function that takes out entries from the replay buffer
"""
def replay_train(reshape_function, model, replay, batch_size, gamma):
    """
    Calculates the target output from the replay variables
    :param reshape_function: given reshape function
    :param model: model in training
    :param replay: replay deque object
    :param batch_size: batch_size
    :param gamma: The discount factor
    :return: model
    """
    # input list of tuples: (game ,action, reward, new_game, running)
    # Random sampling
    sample_batch = random.sample(replay, batch_size)

    # training data sets
    X_train = []
    Y_train = []

    for state, action, reward, next_state, running in sample_batch:
        # Calculate reward
        target = reward
        if running:
          target = reward + gamma * np.amax(model.predict(reshape_function(next_state)))
        # Calculate current qvalues and the target
        target_f = model.predict(reshape_function(state))
        target_f[0][action] = target
        # Appending sample to the batch datasets
        Y_train += [target_f]
        X_train += [reshape_function(state)]

    # merging the data
    X_train = np.concatenate(X_train, axis = 0)
    Y_train = np.concatenate(Y_train, axis = 0)

    # Train the network
    model.fit(X_train, Y_train, epochs=1, verbose=0)

    return model


def getReward(state, new_state, score, new_score, running):
    """
    Function that returns the reward given the state and action made.
    :param state: s_i
    :param new_state: s_{i+1}
    :param score: score at time i
    :param new_score: score at time i+1
    :param running: boolean variable that is true if the game is still going
    :return: reward value
    """
    # if it makes a move that does not change the placement of the tiles
    if not running:
        return -1
    # If the game ended
    elif running and ((state==new_state).sum()==16):
        return -1

    # Else if it reached a new highest tile
    elif (state.max() < new_state.max()) & (new_state.max() in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]):
        return 1
    else:
        return 0


"""
Main training function
"""
def training(main_count, pre_count, gamma, model, reshape_function, epsilon=1):
    # Variables for storage during training
    memory = deque(maxlen=buffer) # Replay storage
    train_scores = [] # Scores at the end of the training games
    test_scores_avg = [] # Average scores of the test episodes
    test_scores_min = [] # Minimum scores of the test episodes
    test_scores_max = [] # Maximum scores of the test episodes


    # Play one game with random weights
    print("First game after initialization...")
    test_play(model=model, reshape_function=reshape_function)

    # Set counter and epoch to zero
    count = 0
    epoch = 0

    # Looping over the epochs
    while count < main_count+pre_count:
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
                if (count >= pre_count) & (epsilon > 0.1):
                    epsilon -= 1.5 / main_count

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
        if ((epoch % 500) == 0) and (epoch != 0):
            print("Running test plays after %s games." %(epoch))
            # Test play
            score_list = avg_test_plays(20, model=model, reshape_function=reshape_function)
            print("Average, min and max of the test scores: ", np.mean(score_list), min(score_list), max(score_list))
            # Store test statistics
            test_scores_avg += [np.mean(score_list)]
            test_scores_min += [min(score_list)]
            test_scores_max += [max(score_list)]
            print("Maximum average score after Game %s: " % (epoch + 1,), str(max(test_scores_avg)))

    # Test play after last train
    score_list, _ = test_play(model=model, reshape_function=reshape_function)
    print("Maximum average score after Game %s: " % (epoch + 1,), str(max(test_scores_avg)))


    return train_scores, test_scores_avg, test_scores_min, test_scores_max, model



# Choose reshape function and game_shape_after_reshaping based on reshape_type
if reshape_type == 'onehot':
    reshape_function = onehot_reshape
    game_shape_after_reshaping = (17, 4, 4)
elif reshape_type == 'linear':
    reshape_function = linear_reshape
    game_shape_after_reshaping = (1, 4, 4)
elif reshape_type == 'trig':
    reshape_function = trig_reshape
    game_shape_after_reshaping = (3, 4, 4)
elif reshape_type == 'flat':
    reshape_function = flat_reshape


# Initialize model, if reshape_style is 'flat' initialize a feed-forward net otherwise a convolutional
if reshape_type in ['onehot', 'linear', 'trig']:
    model = init_conv_model(game_shape_after_reshaping)
else:
    model = init_flat_model()


# Run training on model and reshape function
train_scores, test_scores_avg, test_score_min, test_score_max, model = training(train_count, pre_train_count, gamma, model, reshape_function)


# Write out the generated data and the statistics into the hdf5 file given as the output path
# name is given as the training timestamp
name = datetime.datetime.fromtimestamp(
    int(time.time())
).strftime('%Y-%m-%d_%H:%M:%S')

with h5py.File(output_path, 'a') as f:
    f[name] = train_scores
    f[name].attrs['test_scores_avg'] = test_scores_avg
    f[name].attrs['test_score_min'] = test_score_min
    f[name].attrs['test_score_max'] = test_score_max
    f[name].attrs['gamma'] = gamma
    f[name].attrs['batch_size'] = batch_size
    f[name].attrs['buffer'] = buffer
    f[name].attrs['epochs_num'] = train_count
    f[name].attrs['pre_train_games'] = pre_train_count
    f[name].attrs['reshape_type'] = reshape_type
    f[name].attrs['test_num'] = test_num


# Save trained model
model.save(reshape_type+"_model_"+name+".hdf5")



"""
# Plot the train_scores and test_scores and save it in a .png file
plt.plot(range(1, epochs+1), train_scores, label='Train scores')
plt.plot(list(range(1, epochs+1, epochs//test_num)), test_scores, label='Test scores averages')
plt.legend(loc='upper left')
plt.title("Training on "+str(epochs)+" games")

plt.savefig(output_path)
"""