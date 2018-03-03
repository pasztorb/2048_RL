import numpy as np

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, Input, concatenate
from keras.layers.core import Permute
from keras.optimizers import Adam, SGD, RMSprop


"""
Different functions for reshaping the 20x4x4 array before feeding into the neural network
"""

def onehot_reshape(state):
    """
    Reshapes the 4x4 array into a 1x17x4x4 so that the keras model will recognize it properly
    :param state: 4x4 numpy array representing the current state
    :return: 1x17x4x4
    """
    # Initialize the output array
    reshaped_state = np.zeros((17,4,4))
    reshaped_state[0,:,:] = 1
    # Find the non-empty tiles
    places = np.where(state != 0)
    # In each tile change the value of the array
    for i in range(places[0].shape[0]):
        value = int(np.log(state[places[0][i], places[1][i]])/np.log(2))
        reshaped_state[value, places[0][i], places[1][i]] = 1
        reshaped_state[0, places[0][i], places[1][i]] = 0

    return reshaped_state[np.newaxis, :, :, :]


def linear_reshape(state):
    """
    Reshapes the state variable as a 1x4x4 matrix where each tile is represented as number between 0 and 1.
    The higher value tiles are represented by larger numbers. It assumes that the largest tile possible is 2^16
    :param state: 4x4 numpy array
    :return: 1x1x4x4 numpy array
    """
    new_state = state/(2**16)
    return new_state[np.newaxis,np.newaxis,:,:]


def trig_reshape(state):
    """
    Reshapes the state into a 1x3x4x4 array where the second axis is a sine, cose and a sigmoid like value.
    It assumes that the largest tile possible is 2^16
    :param state: 20x4x4 numpy array
    :return: 1x3x4x4 numpy array
    """
    # Sine scale
    sine_scale = np.vectorize(lambda x: np.sin(x * 2 * np.pi))(state/(2**16))
    cos_scale = np.vectorize(lambda x: np.cos(x * 2 * np.pi))(state/(2**16))
    sigmoid_scale = np.vectorize(lambda x: 1/(1 + np.exp(-0.5*(x*20-10))))(state/(2**16))

    new_state = np.stack([sine_scale, cos_scale, sigmoid_scale])

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
                     use_bias=False,
                     data_format='channels_first',
                     name='first_conv',
                     input_shape=(input_shape[0],input_shape[1],input_shape[2]))
    conv_1d = Conv2D(filters=filter_size_2,
                     kernel_size=(1,1),
                     strides=(1,1),
                     use_bias=False,
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
                   activation='relu',
                   use_bias=False
                   )(output)
    output = Dense(4, activation='linear')(output)

    model = Model(inputs=state_input, outputs=output)

    print(model.summary())

    opt = RMSprop()
    model.compile(loss='mse', optimizer=opt)
    return model

