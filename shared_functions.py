import numpy as np
import random

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, Input, concatenate, Add
from keras.layers.core import Permute
from keras.optimizers import Adam


"""
Different functions for reshaping the 20x4x4 array before feeding into the neural network
"""

def onehot_reshape(state, flat=False):
    """
    Reshapes the 4x4 array into a 1x17x4x4 so that the keras model will recognize it properly
    :param state: 4x4 numpy array representing the current state
    :return: 1x17x4x4
    """
    # Change the state values to integer categories from 1 to 17
    log_state = state.copy()
    log_state[log_state==0]=1
    log_state = (np.log(log_state)/np.log(2)).astype(int)
    # Initialize the output state
    reshaped_state = np.zeros((17,16), dtype=np.uint8)
    # Add the ones to the given places
    reshaped_state[log_state.ravel(),np.arange(16)] = 1
    # Reshape the state representation to (17,4,4)
    reshaped_state.shape = (17,)+(4,4)

    if not flat:
        return reshaped_state[np.newaxis, :, :, :]
    if flat:
        return reshaped_state.reshape(1,272)


def linear_reshape(state, flat=False):
    """
    Reshapes the state variable as a 1x4x4 matrix where each tile is represented as number between 0 and 1.
    The higher value tiles are represented by larger numbers. It assumes that the largest tile possible is 2^16
    :param state: 4x4 numpy array
    :return: 1x1x4x4 numpy array
    """
    new_state = state.copy()
    new_state[new_state == 0] = 1
    new_state = np.log(new_state) / np.log(2)
    new_state = new_state / 16
    if not flat:
        return new_state[np.newaxis, np.newaxis, :, :]
    if flat:
        return new_state.reshape(1,16)


def trig_reshape(state, flat = False):
    """
    Reshapes the state into a 1x3x4x4 array where the second axis is a sine, cose and a sigmoid like value.
    It assumes that the largest tile possible is 2^16
    :param state: 20x4x4 numpy array
    :return: 1x3x4x4 numpy array
    """
    # Additional scale that rescales the numbers from one to
    lin_scale = linear_reshape(state)[0,0,:,:]
    # Sine scale
    sine_scale = np.vectorize(lambda x: np.sin(x * 2 * np.pi))(lin_scale)
    # Cosine scale
    cos_scale = np.vectorize(lambda x: np.cos(x * 2 * np.pi))(lin_scale)
    # Sigmoid scale
    sigmoid_scale = np.vectorize(lambda x: 1/(1 + np.exp(-0.5*(x*20-10))))(lin_scale)
    # Stack them together
    new_state = np.stack([sine_scale, cos_scale, sigmoid_scale])

    if not flat:
        return new_state[np.newaxis, :, :, :]
    if flat:
        return new_state.reshape(1,48)


"""
Reward function
"""
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
    elif (state==new_state).sum()==16:
        return -1
    # Else if it reached a new highest tile
    elif (state.max() < new_state.max()) & (new_state.max() in [128, 256, 512]):
        return 0.5
    elif (state.max() < new_state.max()) & (new_state.max() in [1024, 2048, 4096, 8192, 16384, 32768, 65536]):
        return 1
    else:
        return 0


"""
Different neural nets to work with
"""

def init_flat_model(input_size):
    """
    Simple feed forward network
    :return: compiled model
    """
    print("Input size of the flat model: ", input_size)
    model = Sequential()
    model.add(Dense(
        256,
        activation='relu',
        use_bias=False,
        input_shape=(input_size,)
    ))
    model.add(Dense(
        128,
        use_bias=False,
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

def init_conv_model(input_shape):
    """
    2D Convolutional network that makes convolution row and columns wise with shared weights.
    Then flatten, concatenate them and add dense layers.
    :param input_shape: tuple of length 3 that sets the dimensions of the expected input
    :return: compiled model
    """
    filter_size_1 = 32
    filter_size_2 = 32
    state_input = Input((input_shape[0],input_shape[1],input_shape[2]))
    # Switch row and column axis
    permut_input = Permute((1,3,2), input_shape=(input_shape[0],input_shape[1],input_shape[2]))(state_input)

    # Column convolutions
    col_conv_1 = Conv2D(filters=filter_size_1,
                     kernel_size=(4,1),
                     strides=(4,1),
                     use_bias=False,
                     data_format='channels_first',
                     name='col_conv_1',
                     input_shape=(input_shape[0],input_shape[1],input_shape[2]))
    col_conv_2 = Conv2D(filters=filter_size_2,
                     kernel_size=(1,1),
                     strides=(1,1),
                     use_bias=False,
                     data_format='channels_first',
                     name='col_conv_2')

    conv_1 = col_conv_1(state_input)
    conv_1 = col_conv_2(conv_1)
    conv_1 = Flatten()(conv_1)

    # Row convolutions
    row_conv_1 = Conv2D(filters=filter_size_1,
                     kernel_size=(4,1),
                     strides=(4,1),
                     use_bias=False,
                     data_format='channels_first',
                     name='row_conv_1',
                     input_shape=(input_shape[0],input_shape[1],input_shape[2]))
    row_conv_2 = Conv2D(filters=filter_size_2,
                     kernel_size=(1,1),
                     strides=(1,1),
                     use_bias=False,
                     data_format='channels_first',
                     name='row_conv_2')

    # Dense layer
    conv_2 = row_conv_1(permut_input)
    conv_2 = row_conv_2(conv_2)
    conv_2 = Flatten()(conv_2)

    concat = concatenate([conv_1, conv_2])

    # Action stream
    action = Dense(256,
                   activation='relu',
                   use_bias=False
                   )(concat)
    action = Dense(4, activation='linear',use_bias=False)(action)

    # Value stream
    value = Dense(256,
                  activation='relu',
                  use_bias=False)(concat)
    value = Dense(1, activation='linear',use_bias=False)(value)

    # Add the action and the value stream
    output = Add()([action, value])

    model = Model(inputs=state_input, outputs=output)

    print(model.summary())

    opt = Adam()
    model.compile(loss='mse', optimizer=opt)
    return model



"""
Replay training function
"""
def replay_train(reshape_function, model, replay, batch_size, gamma):
    """
    Calculates the target output from the replay variables and updates the model
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
    update_tuples = []

    for state, action, reward, next_state, running in sample_batch:
        # Calculate reward
        target = reward
        if reward != -1:
          target = reward + gamma * np.amax(model.predict(reshape_function(next_state)))
        # Reshape the current state
        reshaped_state = reshape_function(state)
        # Appending the tuple and the reshaped state to the lists
        update_tuples +=[(action, target)]
        X_train += [reshaped_state]

    # Calculate the output of X_train
    X_train = np.concatenate(X_train, axis = 0)
    Y_train = model.predict(X_train)
    # Update Y_train based on the update tuples
    for i,t in enumerate(update_tuples):
        Y_train[i,t[0]] = t[1]

    # Train the network
    model.fit(X_train, Y_train, epochs=1, verbose=0)

    return model
