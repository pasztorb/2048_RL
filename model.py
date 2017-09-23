import numpy as np
import sys
from game import *

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, concatenate
from keras.layers.core import Permute
from keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l2
from keras import backend as K

import matplotlib.pyplot as plt

# Command line given variables
epochs = int(sys.argv[1])
reshape_type = sys.argv[2]
assert reshape_type in ['onehot', 'linear', 'trig', 'flat']
plot_path = sys.argv[3]

# Fixed variables
gamma = 0.99
epsilon = 1
batch_size = 200
buffer = 1000
test_num = 50
game_shape = (20, 4, 4)


"""
Different function for reshaping the 20x4x4 array before feeding into the neural network
"""

def reshape_state(state):
    """
    Function that reshapes the given array for a 3D convolution. (i.e. adds two axes, one for the channels and one for the batch siez)
    :param state: numpy array representing the current state
    :return: reshaped array
    """
    return state[np.newaxis, :, :, :]


def linear_reshape(state):
    """
    This function reshapes the state variable as a 1x4x4 matrix where each tile is represented as number between 0 and 1. The higher value tiles are represented by larger numbers.
    :param state: 20x4x4 numpy array
    :return: 1x4x4 numpy array
    """
    # Linear scale from 0 to 1. 21 samples for the 20 different tiles plus zero
    linear_scale = np.linspace(0,1,20)
    # New 1x4x4 array for the new state
    new_state = np.zeros((1,4,4))

    for j in range(4): # Iteration for the columns
        for i in range(4): # Iteration for the rows
            index = np.where(state[:,i,j] == 1)[0][0]
            new_state[:,i,j] = linear_scale[index]

    return new_state[np.newaxis,:,:,:]


def trig_reshape(state):
    lin_scale = np.linspace(0, 1, 20, endpoint=True)
    sine_scale = np.sin(lin_scale * 2 * np.pi)
    cos_scale = np.cos(lin_scale * 2 * np.pi)
    sigmoid_scale = 1/(1 + np.exp(-0.5*(lin_scale*20-10)))
    three_chanel_scale = np.stack([sine_scale, cos_scale, sigmoid_scale])

    # New 1x4x4 array for the new state
    new_state = np.zeros((3,4,4))

    # Iterate over the tiles to fill in the new_state variable
    state_sum = state.sum(axis=0)
    for j in range(4): # Iteration for the columns
        for i in range(4): # Iteration for the rows
            index = np.where(state[:,i,j] == 1)[0][0]
            new_state[:,i,j] = three_chanel_scale[:,index]

    return new_state[np.newaxis, :, :, :]

def flat_reshape(state):
    """
    Function that reshapes the given array for a 3D convolution. (i.e. adds two axes, one for the channels and one for the batch siez)
    :param state: numpy array representing the current state
    :return: reshaped array
    """
    return state.reshape((1,state.shape[0]*state.shape[1]*state.shape[2]))


"""
Different neural nets to work with
"""


def init_flat_model():
    """
    Simple feed forward network
    :return: compiled model
    """
    model = Sequential()
    model.add(Dense(
        1024,
        activation='relu',
        input_shape=(320,)
    ))
    model.add(Dense(
        1024,
        activation='relu'
    ))
    model.add(Dense(
        4,
        activation='linear'
    ))

    print(model.summary())

    opt = Adam()
    model.compile(loss='mse', optimizer=opt)

    return model

def init_conv_model_1(input_shape):
    filter_size_1 = 32
    state_input = Input((input_shape[0],input_shape[1],input_shape[2]))
    row_conv_2 = Conv2D(filters = filter_size_1,
                      kernel_size=(1,2),
                      strides=(1,1),
                      data_format="channels_first",
                      activation="relu"
                      )(state_input)
    row_conv_3 = Conv2D(filters = filter_size_1,
                      kernel_size=(1,2),
                      strides=(1,1),
                      data_format="channels_first",
                      dilation_rate=(1,2),
                      activation="relu"
                      )(state_input)
    row_conv_4 = Conv2D(filters = filter_size_1,
                      kernel_size=(1,2),
                      strides=(1,1),
                      data_format="channels_first",
                      dilation_rate=(1,3),
                      activation="relu"
                      )(state_input)
    row_flat_2 = Flatten()(row_conv_2)
    row_flat_3 = Flatten()(row_conv_3)
    row_flat_4 = Flatten()(row_conv_4)

    col_conv_2 = Conv2D(filters = filter_size_1,
                      kernel_size=(2,1),
                      strides=(1,1),
                      data_format="channels_first",
                      activation="relu"
                      )(state_input)
    col_conv_3 = Conv2D(filters = filter_size_1,
                      kernel_size=(2,1),
                      strides=(1,1),
                      data_format="channels_first",
                      dilation_rate=(2,1),
                      activation="relu"
                      )(state_input)
    col_conv_4 = Conv2D(filters = filter_size_1,
                      kernel_size=(2,1),
                      strides=(1,1),
                      data_format="channels_first",
                      dilation_rate=(3,1),
                      activation="relu"
                      )(state_input)
    col_flat_2 = Flatten()(col_conv_2)
    col_flat_3 = Flatten()(col_conv_3)
    col_flat_4 = Flatten()(col_conv_4)

    output = concatenate([row_flat_2, row_flat_3, row_flat_4, col_flat_2, col_flat_3, col_flat_4])

    output = Dense(1024,
                   activation='relu',
                   kernel_regularizer=l2(0.002)
                   )(output)
    output = Dense(1024,
                   activation='relu',
                   kernel_regularizer=l2(0.002)
                   )(output)

    output = Dense(4, activation='linear')(output)

    model = Model(inputs=state_input, outputs=output)

    print(model.summary())

    opt = Adam()
    model.compile(loss='mse', optimizer=opt)
    return model


def init_conv_model_2(input_shape):
    filter_size_1 = 128
    filter_size_2 = 128
    state_input = Input((input_shape[0],input_shape[1],input_shape[2]))
    permut_input = Permute((1,3,2),
                           input_shape=(input_shape[0],input_shape[1],input_shape[2]))(state_input)

    conv_2d = Conv2D(filters=filter_size_1,
                     kernel_size=(4,1),
                     strides=(4,1),
                     data_format='channels_first',
                     name='first_conv',
                     input_shape=(input_shape[0],input_shape[1],input_shape[2]))
    conv_1d = Conv2D(filters=filter_size_2,
                     kernel_size=(1,1),
                     strides=(1,1),
                     data_format='channels_first',
                     name='1x1_conv')

    conv_1 = conv_2d(state_input)
    conv_1 = conv_1d(conv_1)
    conv_1 = Flatten()(conv_1)

    conv_2 = conv_2d(permut_input)
    conv_2 = conv_1d(conv_2)
    conv_2 = Flatten()(conv_2)

    output = concatenate([conv_1, conv_2])

    output = Dense(1024,
                   activation='relu'
                   )(output)
    output = Dense(4, activation='linear')(output)

    model = Model(inputs=state_input, outputs=output)

    print(model.summary())

    opt = Adam()
    model.compile(loss='mse', optimizer=opt)
    return model


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

def getReward(state, new_state, score, new_score, running):
    # if it makes a move that does not change the placement of the tiles
    if not running:
        return -20
    # If the game ended
    elif running and ((state==new_state).sum()==game_shape[0]*game_shape[1]*game_shape[2]):
        return -1
    # Else if it made a valid move
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
    test_scores = [] # Scores at the end of the test games

    # Play one game with random weights
    print("First game after initialization...")
    test_play(model=model, reshape_function=reshape_function)

    # Looping over the epochs number
    for epoch in range(epochs):
        print("Epcoh number: ", str(epoch+1))
        print("Epsilon value: ", str(epsilon))

        # Initialize game
        game, score = initialize_game()

        # Running variable for looping
        running = True
        while running:
            # Evaluate on the current state
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

            # Predict for the outcome state
            newQ = model.predict(reshape_function(new_game), batch_size=1)
            maxQ = np.max(newQ)

            # Calculate the target output
            y = np.zeros((1, 4))
            y[:] = qval[:]

            # Calculate update value
            if running == True and ((game==new_game).sum() != game_shape[0]*game_shape[1]*game_shape[2]):  # non-terminal state and not an invalid move
                update = (reward + (gamma * maxQ))
            else:  # terminal state
                update = reward
            # target output
            y[0][action] = update

            # Experience replay storage
            if (len(replay) < buffer):  # if buffer not filled, add to it
                replay.append((game, y))
            else:
                # Choose random from the replay list
                indicies = np.random.choice(buffer, batch_size)
                X_train, Y_train = [], []

                # Append chosen entries
                for i in indicies:
                    X_train.append(reshape_function(replay[i][0]))
                    Y_train.append(replay[i][1])

                # Remove used entries
                replay = [i for j, i in enumerate(replay) if j not in indicies]

                # Concatenate the training data into one matrix
                X_train = np.concatenate(X_train,axis=0)
                Y_train = np.concatenate(Y_train,axis=0)

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

        # If one test_num games have passed play a test game
        if (epoch % (epochs//test_num)) == 0:
            print("Current epsilon value: ", str(epsilon))
            # Test play
            score_list = avg_test_plays(20, model=model, reshape_function=reshape_function)
            print("Average, min and max of the test scores: ", np.mean(score_list), min(score_list), max(score_list))
            test_scores += [np.mean(score_list)] # Store test score
            print("Maximum average score after Game %s: " % (epoch + 1,), str(max(test_scores)))

    # Train on the remaining samples in the replay list
    print("Train on the remaining samples")
    X_train, Y_train = [], []

    # Append chosen entries
    for i in range(len(replay)):
        X_train.append(reshape_function(replay[i][0]))
        Y_train.append(replay[i][1])
    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)

    # Fit the model on data
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=1)

    # Test play after last train
    score_list, _ = test_play(model=model, reshape_function=reshape_function)
    print("Maximum score after Game %s: " % (epoch + 1,), str(max(test_scores)))
    return train_scores, test_scores


# Choose reshape function based on reshape_type
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
train_scores, test_scores = training(epochs, gamma, model, reshape_function)


# Plot the train_scores and test_scores and save it in a .png file
plt.plot(range(1, epochs+1), train_scores, label='Train scores')
plt.plot(list(range(1, epochs+1, epochs//test_num)), test_scores, label='Test scores averages')
plt.legend(loc='upper left')
plt.title("Training on "+str(epochs)+" games")

plt.savefig(plot_path)