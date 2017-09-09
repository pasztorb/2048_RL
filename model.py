import numpy as np
from game import *

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv3D, Input, concatenate
from keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l2

epochs = 20
gamma = 0.9
epsilon = 1
batch_size = 60
buffer = 600
game_shape = (20,4,4)


def reshape_state(state):
    """
    Function that reshapes the given array for a 3D convolution. (i.e. adds two axes, one for the channels and one for the batch siez)
    :param state: numpy array representing the current state
    :return: reshaped array
    """
    return state[np.newaxis, np.newaxis, :, :, :]

def color_scale_reshape(state):
    """
    This function reshapes the state array into a similar 3D array but in the new array RGB color channels represent the tiles
    instead of one-hot vectors.
    :param state: Current state array.
    :return: state: New state array.
    """

    # 20*3 matrix that holds the 20 different colors
    color_matrix = np.array()

    # New array to fill in with the new values
    new_state = np.zeros((3,4,4))

    # Iterate over the tiles to fill in the new_state variable
    state_sum = state.sum(axis=0)
    for j in range(4): # Iteration for the columns
        for i in range(4): # Iteration for the rows
            if state_sum[i,j] == 0: # If there is nothing in that tile skip it
                continue
            else:
                index_of_one = np.where(state[:,i,j]==1)[0][0]
                new_state[:,i,j] = color_matrix[index_of_one,:]


def init_model_1():
    model = Sequential()
    model.add(Dense(512,
                    activation='relu',
                    kernel_regularizer=l2(0.01),
                    input_shape=(game_shape[0]*game_shape[1]*game_shape[2],)
                    ))
    model.add(Dense(512,
                    activation='relu',
                    kernel_regularizer=l2(0.01)
                    ))
    model.add(Dense(256,
                    activation='relu',
                    kernel_regularizer=l2(0.01)
                    ))
    model.add(Dense(4, activation='linear'))
    print(model.summary())

    opt = RMSprop()
    model.compile(loss='mse', optimizer=opt)
    return model

def init_model_2():
    filter_size_1 = 12
    filter_size_2 = 24
    state_input = Input((1,game_shape[0],game_shape[1],game_shape[2]))
    row_conv = Conv3D(filters = filter_size_1,
                      kernel_size=(2,1,2),
                      strides=(1,1,1),
                      data_format="channels_first",
                      activation="relu"
                      )(state_input)
    row_conv = Conv3D(filters = filter_size_2,
                      kernel_size=(2,1,2),
                      strides=(1,1,1),
                      data_format="channels_first",
                      activation="relu"
                      )(row_conv)
    row_flat = Flatten()(row_conv)

    col_conv = Conv3D(filters = filter_size_1,
                      kernel_size=(2, 2, 1),
                      strides=(1, 1, 1),
                      data_format="channels_first",
                      activation="relu"
                      )(state_input)
    col_conv = Conv3D(filters = filter_size_1,
                      kernel_size=(2, 2, 1),
                      strides=(1, 1, 1),
                      data_format="channels_first",
                      activation="relu"
                      )(col_conv)
    col_flat = Flatten()(col_conv)

    output = concatenate([row_flat, col_flat])

    output = Dense(1024,
                   activation='relu',
                   kernel_regularizer=l2(0.002)
                   )(output)

    output = Dense(4, activation='linear')(output)

    model = Model(inputs=state_input, outputs=output)

    print(model.summary())

    opt = RMSprop()
    model.compile(loss='mse', optimizer=opt)
    return model

def getReward(state, new_state, score, new_score, running):
    # if it makes a move that does not change the placement of the tiles
    if running and ((state==new_state).sum()==game_shape[0]*game_shape[1]*game_shape[2]):
        return -3
    # If the game ended
    elif not running:
        return -30
    # The game did not end and it is not an invalid move
    else:
        return (new_score-score)/100

def training(epochs, gamma, model, reshape_function, epsilon=1):
    max_score = 0
    replay = []
    # Play one game with random weights
    print("First game after initialization...")
    test_play(model=model, reshape_function=reshape_function)
    # Looping over the epochs number
    for i in range(epochs):
        print("Epcoh number: ", str(i+1))
        print("Epsilon value: ", str(epsilon))
        game, score = initialize_game()
        running = True
        while running:
            # Evaluate on the current state
            qval = model.predict(reshape_function(game), batch_size=1)
            if (np.random.random() < epsilon):  # choose random action
                action = np.random.randint(0, 4)
            else:  # choose best action from Q(s,a) values
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
            if running == True and reward != -3:  # non-terminal state and not an invalid move
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
                indicies = np.random.choice(buffer,batch_size)
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
        print("Score at the end of the training game: ", str(score))
        # Reduce epsilon value after game finished
        if epsilon > 0.1:
            epsilon -= (1 / epochs)

        if (i % (epochs//100)) == 0:
            print("Current epsilon value: ", str(epsilon))
            # Test play
            score_list, _ = test_play(model=model, reshape_function=reshape_function)
            test_score = score_list[-1]
            if test_score > max_score:
                max_score = test_score
            print("Maximum score after Game %s: " % (i+1,), str(max_score))

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
    test_score = score_list[-1]
    if test_score > max_score:
        max_score = test_score
    print("Maximum score after Game %s: " % (i + 1,), str(max_score))


model = init_model_2()
training(epochs, gamma, model, reshape_state)