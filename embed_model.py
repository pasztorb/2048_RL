import sys
import os
import h5py
import time
import datetime

from game import *
from embedding_2048 import *
from shared_functions import onehot_reshape

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, concatenate
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.core import Permute
from keras.regularizers import l2

# Variables given in command lines
epochs = int(sys.argv[1])
embedding_size = int(sys.argv[2])
output_folder = sys.argv[3]

# Fixed variables
gamma = 0.9
epsilon = 1
batch_size = 300
buffer = 1000
test_num = 50
game_shape = (20, 4, 4)

pre_train_games = 100

"""
Initialize a folder in which the training plots, game states and the model will be stored
"""
output_folder = "/"+output_folder
os.mkdir(output_folder)

"""
Implementation of the Convolutional network
"""

def init_model(input_shape, embed_model):
    filter_size_1 = 128
    filter_size_2 = 128
    # Input
    state_input = Input((input_shape[0],input_shape[1],input_shape[2]))
    # Embedding convolution
    embedding_weights = embed_model.get_layer(name='embed_conv').get_weights()
    embed_conv = Conv2D(filters = embedding_weights[0].shape[-1],
                        kernel_size = (1, 1),
                        strides = (1, 1),
                        use_bias = False,
                        trainable = False,
                        name = 'embed_conv',
                        data_format = "channels_first")
    embed_conv.set_weights(embedding_weights)
    embed_input = embed_conv(state_input)

    # Switch row and column axis
    permut_input = Permute((1,3,2),
                           input_shape=(input_shape[0],input_shape[1],input_shape[2]))(embed_input)

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

    conv_1 = conv_2d(embed_input)
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


def replay_to_matrix(reshape_function, model, list, gamma):
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
def training(epochs, gamma, model, embed_model, reshape_function, X_embedding, Y_embedding, epsilon=1):
    # Variables for storage during training
    replay = [] # List for replay storeage
    train_scores = []
    train_games = []

    test_scores_avg = [] # Average scores of the test episodes
    test_scores_min = [] # Minimum scores of the test episodes
    test_scores_max = [] # Maximum scores of the test episodes

    # Lists to store game states for embedding
    X_embed = X_embedding
    Y_embed = Y_embedding

    # Play one game with random weights
    print("First game after initialization...")
    _, _ = test_play(model=model, reshape_function=reshape_function)

    # Looping over the epochs number
    for epoch in range(epochs):
        print("Epcoh number: ", str(epoch+1))
        print("Epsilon value: ", str(epsilon))
        # Initialize game
        game, score = initialize_game()
        # Train game states
        game_states = [game]
        # Variable to keep in charge the running
        running = True
        while running:
            # Evaluate on the current state Q(s_i)
            qval = model.predict(reshape_function(game), batch_size=1)

            # Choose if the action is random or not
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
                replay.append((game, action, reward, new_game, running))
            else:  # Train
                # Choose randomly from the replay list
                indicies = np.random.choice(buffer, batch_size)
                replay_list = []

                # Append chosen entries
                for i in indicies:
                    replay_list.append(replay[i])

                # Remove used entries
                replay = [i for j, i in enumerate(replay) if j not in indicies]

                # Transform the replay list into trainable matrices
                X_train, Y_train = replay_to_matrix(reshape_function, model, replay_list, gamma)

                # Train model on X_train, Y_train
                model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=1)

            # Update game and score variables
            game = new_game
            score = new_score
            # Add game to game_states
            game_states += [game]

        # Store and print score at the end of the training game
        train_scores += [score]
        print("Score at the end of the training game: ", str(score))

        # Add game_states list to train_games list
        train_games += [game_states]

        # Reduce epsilon value after game finished
        if epsilon > 0.1:
            epsilon -= (1 / epochs)

        # If test_num games have passed play test games
        # Add update embedding
        if (epoch % (epochs // test_num)) == 0:
            print("Current epsilon value: ", str(epsilon))
            # Test play
            score_list = avg_test_plays(20, model=model, reshape_function=reshape_function)
            print("Average, min and max of the test scores: ", np.mean(score_list), min(score_list),
                  max(score_list))
            # Store test statistics
            test_scores_avg += [np.mean(score_list)]
            test_scores_min += [min(score_list)]
            test_scores_max += [max(score_list)]
            print("Maximum average score after Game %s: " % (epoch + 1,), str(max(test_scores_avg)))

            # Update embedding
            #Todo: only transform games that are not yet appended to X_embed, Y_embed
            X_embed_new, Y_embed_new = games_to_trainable(train_games)
            # Add the new embeddings to the already existing ones
            X_embed[0] += X_embed_new[0]
            X_embed[1] += X_embed_new[1]
            for i in range(len(Y_embed)):
                Y_embed[i] += Y_embed_new[i]

            # Train embeddings model
            embed_model = train_embed_model(embed_model, X_embed, Y_embed)

            # Update weight of embedding layer
            model.get_layer(name='embed_conv').set_weights(embed_model.get_layer(name='embed_conv').get_weights())


    # Train on the remaining samples in the replay list
    print("Train on the remaining samples")

    # Reshape remaining replay data
    X_train, Y_train = replay_to_matrix(reshape_function, model, replay, gamma)

    # Fit the model on data
    model.fit(X_train, Y_train, batch_size=batch_size//5, epochs=1, verbose=1)

    # Test play after last train
    score_list, _ = test_play(model=model, reshape_function=reshape_function)
    print("Maximum average score after Game %s: " % (epoch + 1,), str(max(test_scores_avg)))

    return train_scores, test_scores_avg, test_scores_min, test_scores_max, model


# Initialize embedding model
embed_model, X_embedding, Y_embedding = init_and_pretrain_embed_model(embedding_size, pre_train_games)

# Initialize model
model = init_model(game_shape, embed_model)

# Run training on model and reshape function
train_scores, test_scores_avg, test_score_min, test_score_max, model = training(epochs, gamma, model, embed_model, onehot_reshape, X_embedding, Y_embedding)

# Write out the generated data and the statistics into the hdf5 file given as the output path
# name is given as the training timestamp
name = datetime.datetime.fromtimestamp(
    int(time.time())
).strftime('%Y-%m-%d_%H:%M:%S')
with h5py.File(output_folder+"training.hdf5", 'a') as f:
    f[name] = train_scores
    f[name].attrs['test_scores_avg'] = test_scores_avg
    f[name].attrs['test_score_min'] = test_score_min
    f[name].attrs['test_score_max'] = test_score_max
    f[name].attrs['gamma'] = gamma
    f[name].attrs['batch_size'] = batch_size
    f[name].attrs['buffer'] = buffer
    f[name].attrs['epochs_num'] = epochs
    f[name].attrs['test_num'] = test_num
    f[name].attrs['embedding_size'] = embedding_size
    f[name].attrs['pre_train_num'] = pre_train_games


# Save trained model
model.save(output_folder+"model_"+name+".hdf5")