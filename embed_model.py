import numpy as np
import sys
from game import *
from embedding_2048 import *

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, concatenate
from keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l2

import matplotlib.pyplot as plt

# Command line given variables
epochs = int(sys.argv[1])
reshape_type = sys.argv[2]
assert reshape_type in ['onehot', 'linear', 'trig']
plot_path = sys.argv[3]

# Fixed variables
gamma = 0.9
epsilon = 1
batch_size = 60
buffer = 600
test_num = 50
game_shape = (20, 4, 4)
embedding_size = 8
pre_train_games = 1000


"""
Different neural nets to work with
"""

def init_model(input_shape, embed_model):
    filter_size_1 = 16
    # Input
    state_input = Input((input_shape[0],input_shape[1],input_shape[2]))
    # Embedding convolution
    embedding_weights = embed_model.get_layer(name='embed_conv').get_weights()
    embed_conv = Conv2D(filters = embedding_size,
                        kernel_size = (1, 1),
                        strides = (1, 1),
                        use_bias = False,
                        trainable = False,
                        name = 'embed_conv',
                        data_format = "channels_first")
    embed_conv.set_weights(embedding_weights)
    embed_input = embed_conv(state_input)

    # Convolution along the rows
    row_conv_2 = Conv2D(filters = filter_size_1,
                      kernel_size=(1,2),
                      strides=(1,1),
                      data_format="channels_first",
                      activation="relu"
                      )(embed_input)
    row_conv_3 = Conv2D(filters = filter_size_1,
                      kernel_size=(1,2),
                      strides=(1,1),
                      data_format="channels_first",
                      dilation_rate=(1,2),
                      activation="relu"
                      )(embed_input)
    row_conv_4 = Conv2D(filters = filter_size_1,
                      kernel_size=(1,2),
                      strides=(1,1),
                      data_format="channels_first",
                      dilation_rate=(1,3),
                      activation="relu"
                      )(embed_input)
    row_flat_2 = Flatten()(row_conv_2)
    row_flat_3 = Flatten()(row_conv_3)
    row_flat_4 = Flatten()(row_conv_4)

    # Convoltuion along the columns
    col_conv_2 = Conv2D(filters = filter_size_1,
                      kernel_size=(2,1),
                      strides=(1,1),
                      data_format="channels_first",
                      activation="relu"
                      )(embed_input)
    col_conv_3 = Conv2D(filters = filter_size_1,
                      kernel_size=(2,1),
                      strides=(1,1),
                      data_format="channels_first",
                      dilation_rate=(2,1),
                      activation="relu"
                      )(embed_input)
    col_conv_4 = Conv2D(filters = filter_size_1,
                      kernel_size=(2,1),
                      strides=(1,1),
                      data_format="channels_first",
                      dilation_rate=(3,1),
                      activation="relu"
                      )(embed_input)
    col_flat_2 = Flatten()(col_conv_2)
    col_flat_3 = Flatten()(col_conv_3)
    col_flat_4 = Flatten()(col_conv_4)

    # Concatenate the outputs
    output = concatenate([row_flat_2, row_flat_3, row_flat_4, col_flat_2, col_flat_3, col_flat_4])

    # Dense Layers
    output = Dense(512,
                   activation='relu',
                   kernel_regularizer=l2(0.002)
                   )(output)
    output = Dense(512,
                   activation='relu',
                   kernel_regularizer=l2(0.002)
                   )(output)

    # Output layer
    output = Dense(4, activation='linear')(output)

    model = Model(inputs=state_input, outputs=output)

    print(model.summary())

    opt = RMSprop()
    model.compile(loss='mse', optimizer=opt)
    return model


"""
Reward function
"""

def getReward(state, new_state, score, new_score, running):
    # if it makes a move that does not change the placement of the tiles
    if running and ((state==new_state).sum()==game_shape[0]*game_shape[1]*game_shape[2]):
        return -10
    # If the game ended
    elif not running:
        return -50
    else:
        return 3


"""
Main training function
"""

def training(epochs, gamma, model, reshape_function, epsilon=1):
    # Variables for storage during training
    replay = [] # List for replay storeage
    train_scores = []
    test_scores = []

    # Play one game with random weights
    print("First game after initialization...")
    test_play(model=model, reshape_function=reshape_function)

    # Looping over the epochs number
    for epoch in range(epochs):
        print("Epcoh number: ", str(epoch+1))
        print("Epsilon value: ", str(epsilon))
        # Initialize game
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

        # If one hundredth of the game has passed play a test game
        if (epoch % (epochs//test_num)) == 0:
            print("Current epsilon value: ", str(epsilon))
            # Test play
            score_list = avg_test_plays(10, model=model, reshape_function=reshape_function)
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


# Initialize embedding model
embed_model = init_and_pretrain_embed_model(embedding_size, pre_train_games)

# Initialize model
model = init_model(game_shape, embed_model)

# Run training on model and reshape function
train_scores, test_scores = training(epochs, gamma, model)


# Plot the train_scores and test_scores and save it in a .png file
plt.plot(range(1, epochs+1), train_scores, label='Train scores')
plt.plot(list(range(1, epochs+1, epochs//test_num)), test_scores, label='Test scores averages')
plt.legend(loc='upper left')
plt.savefig(plot_path)