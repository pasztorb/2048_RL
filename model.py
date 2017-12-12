import numpy as np
import sys
import h5py
import time
import datetime

from game import *
from reshape_and_NN import *

# Variables given in command lines
epochs = int(sys.argv[1])
reshape_type = sys.argv[2]
assert reshape_type in ['onehot', 'linear', 'trig', 'flat']
output_path = sys.argv[3]

# Fixed variables
gamma = 0.9
epsilon = 1
batch_size = 300
buffer = 1000
test_num = 50
game_shape = (20, 4, 4)


"""
Reward function, and entropy calculation function
"""

def state_entropy(state):
    entropy = 0
    for i in range(4): # Iteration over the rows (columns)
        for j in range(3): # Iteration over the columns (rows)
            entropy += np.linalg.norm(state[:, i, j]-state[:, i, j + 1],2)
            entropy += np.linalg.norm(state[:, j, i]-state[:, j + 1, i],2)
    return entropy/16

def replay_to_matrix(reshape_function, model, list):
    """
    Calculates the target output from the replay variables
    :param reshape_function: given reshape function
    :param model: model in training
    :param list: list of replay tuples
    :return: X_train, Y_train
    """
    # input list of tuples: (game ,action, reward, new_game, running)
    X_train, Y_train = [], []

    for i in list:
        # Calculate Q(s_i)
        x = reshape_function(i[0])
        qval = model.predict(x, batch_size=1)

        # Calculate Q(s_{i+1})
        newQ = model.predict(reshape_function(i[3]), batch_size=1)
        maxQ = np.max(newQ)

        # Calculate the target output
        y = np.zeros((1, 4))
        y[:] = qval[:]

        # Update target value
        # If the new state is not terminal and it made a valid move
        if i[4] == True and ((i[0]==i[3]).sum() != game_shape[0]*game_shape[1]*game_shape[2]):
            y[0][i[1]] = (i[2] + (gamma * maxQ))
        else:
            y[0][i[1]] = i[2]

        # Append to X_train, Y_train
        X_train.append(x)
        Y_train.append(y)

    # Concatenate the individual training and target variables
    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)

    return X_train, Y_train

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
        return -20
    # If the game ended
    elif running and ((state==new_state).sum()==game_shape[0]*game_shape[1]*game_shape[2]):
        return -3
    # Else if it reached a new highest tile
    elif np.where(state==1)[0].max() < np.where(new_state==1)[0].max():
        return 2
    else:
        return 1


"""
Main training function
"""
def training(epochs, gamma, model, reshape_function, epsilon=1):
    # Variables for storage during training
    replay = [] # Replay storeage
    train_scores = [] # Scores at the end of the training games
    test_scores_avg = [] # Average scores of the test episodes
    test_scores_min = [] # Minimum scores of the test episodes
    test_scores_max = [] # Maximum scores of the test episodes


    # Play one game with random weights
    print("First game after initialization...")
    test_play(model=model, reshape_function=reshape_function)

    # Looping over the epochs
    for epoch in range(epochs):
        print("Epcoh number: ", str(epoch+1))
        print("Epsilon value: ", str(epsilon))

        # Initialize game
        game, score = initialize_game()

        # Play training game
        running = True
        while running:
            # Evaluate on the current state Q(s_i)
            qval = model.predict(reshape_function(game), batch_size=1)

            # Choose if the next action is random or not
            if (np.random.random() < epsilon):
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(qval)

            # Make a move
            new_game, new_score, running  = action_move(game, score, action)

            # Observe Reward
            reward = getReward(game, new_game, score, new_score, running)

            # Experience replay storage
            if (len(replay) < buffer):  # if buffer not filled, add to it
                replay.append((game ,action, reward, new_game, running))
            else: #Train
                # Choose randomly from the replay list
                indicies = np.random.choice(buffer, batch_size)
                replay_list = []

                # Append chosen entries
                for i in indicies:
                    replay_list.append(replay[i])

                # Remove used entries
                replay = [i for j, i in enumerate(replay) if j not in indicies]

                # Transform the replay list into trainable matrices
                X_train, Y_train = replay_to_matrix(reshape_function, model, replay_list)

                # Train model on X_train, Y_train
                model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=1)

            # Update game and score variables
            game = new_game
            score = new_score

        # Store and print score at the end of the training game
        train_scores += [score]
        print("Score at the end of the training game: ", str(score))

        # Reduce epsilon value after game finished
        if epsilon > 0.1:
            epsilon -= (1 / epochs)

        # If test_num games have passed play test games
        if (epoch % (epochs//test_num)) == 0:
            print("Current epsilon value: ", str(epsilon))
            # Test play
            score_list = avg_test_plays(20, model=model, reshape_function=reshape_function)
            print("Average, min and max of the test scores: ", np.mean(score_list), min(score_list), max(score_list))
            # Store test statistics
            test_scores_avg += [np.mean(score_list)]
            test_scores_min += [min(score_list)]
            test_scores_max += [max(score_list)]
            print("Maximum average score after Game %s: " % (epoch + 1,), str(max(test_scores_avg)))

    # Train on the remaining samples in the replay list
    print("Train on the remaining samples")

    # Reshape remaining replay data
    X_train, Y_train = replay_to_matrix(reshape_function, model, replay)

    # Fit the model on data
    model.fit(X_train, Y_train, batch_size=batch_size//5, epochs=1, verbose=1)

    # Test play after last train
    score_list, _ = test_play(model=model, reshape_function=reshape_function)
    print("Maximum average score after Game %s: " % (epoch + 1,), str(max(test_scores_avg)))


    return train_scores, test_scores_avg, test_scores_min, test_scores_max, model



# Choose reshape function and game_shape_after_reshaping based on reshape_type
if reshape_type == 'onehot':
    reshape_function = reshape_state
    game_shape_after_reshaping = (20, 4, 4)
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
    model = init_conv_model_2(game_shape_after_reshaping)
else:
    model = init_flat_model()


# Run training on model and reshape function
train_scores, test_scores_avg, test_score_min, test_score_max, model = training(epochs, gamma, model, reshape_function)


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
    f[name].attrs['epochs_num'] = epochs
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