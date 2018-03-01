import numpy as np

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, concatenate
from keras.layers.core import Permute
from keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l2
from keras import backend as K


"""
Different functions for reshaping the 20x4x4 array before feeding into the neural network
"""

def reshape_state(state):
    """
    Reshapes the 20x4x4 array into a 1x20x4x4 so that the keras model will recognize it properly
    :param state: 20x4x4 numpy array representing the current state
    :return: 1x20x4x4
    """
    return state[np.newaxis, :, :, :]


def linear_reshape(state):
    """
    Reshapes the state variable as a 1x4x4 matrix where each tile is represented as number between 0 and 1.
    The higher value tiles are represented by larger numbers.
    :param state: 20x4x4 numpy array
    :return: 1x1x4x4 numpy array
    """
    # Linear scale from 0 to 1. 20 samples for the 20 different tiles
    linear_scale = np.linspace(0,1,20)
    # New 1x4x4 array for the new state
    new_state = np.zeros((1,4,4))

    for j in range(4): # Iteration for the columns
        for i in range(4): # Iteration for the rows
            # Finds the index of the entry one in the first axis of the i,j tile
            index = np.where(state[:,i,j] == 1)[0][0]
            # Sets the value of the new state at i,j
            new_state[:,i,j] = linear_scale[index]

    return new_state[np.newaxis,:,:,:]


def trig_reshape(state):
    """
    Reshapes the state into a 1x3x4x4 array where the second axis is a sine, cose and a sigmoid like value.
    :param state: 20x4x4 numpy array
    :return: 1x3x4x4 numpy array
    """
    # Linear scale to ease further computations
    lin_scale = np.linspace(0, 1, 20, endpoint=True)
    # First channel's values, sine function's values at the points of lin_scale
    sine_scale = np.sin(lin_scale * 2 * np.pi)
    # Second channel's values, cosine function's values at the points of lin_scale
    cos_scale = np.cos(lin_scale * 2 * np.pi)
    # Third channel's values, sigmoid function's values at the points of lin_scale
    sigmoid_scale = 1/(1 + np.exp(-0.5*(lin_scale*20-10)))
    # Stacking the three channels
    three_chanel_scale = np.stack([sine_scale, cos_scale, sigmoid_scale])

    # New 1x4x4 array for the new state
    new_state = np.zeros((3,4,4))

    # Iterate over the tiles to fill in the new_state variable
    for j in range(4): # Iteration for the columns
        for i in range(4): # Iteration for the rows
            # Finds the index of the entry one in the first axis of the i,j tile
            index = np.where(state[:,i,j] == 1)[0][0]
            # Sets the value of the new state at i,j
            new_state[:,i,j] = three_chanel_scale[:,index]

    return new_state[np.newaxis, :, :, :]

def flat_reshape(state):
    """
    Reshapes the state variable into a flat array
    :param state: 20x4x4 numpy array
    :return: 1x360 numpy array
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

    output = Dense(256,
                   activation='relu'
                   )(output)
    output = Dense(4, activation='linear')(output)

    model = Model(inputs=state_input, outputs=output)

    print(model.summary())

    opt = Adam()
    model.compile(loss='mse', optimizer=opt)
    return model

