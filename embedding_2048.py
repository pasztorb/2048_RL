from game import *
import sys

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, concatenate
from keras.optimizers import Adam, SGD, RMSprop


def initial_random_plays(num_plays):
    """
    This function randomly plays num_plays number of games. Saves the states in all games and return a nested list with the states for all games.
    :param num_plays: integer variable to give how many random games to play
    :return: nested list of games
    """
    list_of_plays = []
    for i in range(num_plays):
        scores, states = test_play(visualize=False)
        list_of_plays += [states]

    return list_of_plays


def games_to_trainable(list_of_games):
    """
    This function takes a list of games and turn it into a trainable format
    :param list_of_games:
    :return: X, Y (two lists required to train the embedding model)
    """
    # Extract the lists from the games
    X_train = [[],[]]
    Y_train = []

    # At each step add the current state to the X_train list's first list and the current state+2 to the second list
    # At each step add the current state + 1 into the Y_train list reshaped to (20,16), i.e. each column represent one tile vector
    for game in list_of_games:
        for g in range(len(game)-2):
            X_train[0].append(game[g])
            X_train[1].append(game[g+2])
            Y_train.append(game[g].reshape((20,16)))

    # Make one array from the several arrays
    X_train = [np.array(x) for x in X_train]
    Y_train = np.array(Y_train)
    # Split the Y_train array into 16 different lists for the different columns
    Y_train = [Y_train[:, :, i] for i in range(16)]

    return X_train, Y_train


def init_embed_model(embed_size):
    # Inputs
    input_1 = Input((20, 4, 4), name = 'input_1') # Input for the state before
    input_2 = Input((20, 4, 4), name = 'input_2') # Input for the state after

    # Embeding 1x1 convolution (shared weights)
    embed_conv = Conv2D(filters = embed_size,
                        kernel_size = (1, 1),
                        strides = (1, 1),
                        use_bias = False,
                        name = 'embed_conv',
                        data_format = "channels_first",
                        input_shape = (20, 4, 4))

    embed_1 = embed_conv(input_1)
    embed_2 = embed_conv(input_2)

    # Flatten the embedded layers
    flat_1 = Flatten()(embed_1)
    flat_2 = Flatten()(embed_2)

    # Concatenate the flat layers
    concat = concatenate([flat_1, flat_2])

    # Output
    outputs = []
    for i in range(16):
        outputs += [Dense(20,
                         activation = 'softmax',
                         name = 'output_'+str(i+1))(concat)]

    # Define model
    model = Model(inputs = [input_1, input_2], outputs = outputs)

    # compile model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    return model


def train_embed_model(model, X_train_list, Y_train_list):
    epoch_num = 30
    batch_size = 30
    print("Training size: ", X_train_list[0].shape)

    model.fit(X_train_list,
              Y_train_list,
              epochs = epoch_num,
              batch_size = batch_size,
              verbose=2)

    return model

def init_and_pretrain_embed_model(embed_size, pre_train_games):
    games = initial_random_plays(pre_train_games)
    X_train, Y_train = games_to_trainable(games)
    model = init_embed_model(embed_size)
    model = train_embed_model(model, X_train, Y_train)
    return model

"""
print("playing the initial random plays")
games = initial_random_plays(int(sys.argv[1]))
print("Reshaping the games")
X_train, Y_train = games_to_trainable(games)
model = init_embed_model(8)
model = train_embed_model(model, X_train, Y_train)
embedding = model.get_layer(name='embed_conv')
print(embedding.get_weights()[0])
print(embedding.get_weights()[0].shape)
"""
