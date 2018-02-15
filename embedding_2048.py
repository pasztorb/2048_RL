from game import *
import sys

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, concatenate
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping


def initial_random_plays(num_plays):
    """
    This function randomly plays num_plays number of games. Saves the states in all games and return a nested list with the states for all games.
    :param num_plays: integer variable to give how many random games to play
    :return: nested list of states of games
    """
    list_of_states = []
    for i in range(num_plays):
        _, states = test_play(visualize=False)
        list_of_states += [states]

    return list_of_states


def games_to_trainable(list_of_games):
    """
    This function takes a list of games and turn it into a trainable format
    :param list_of_games: nested list of games
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
    """
    Initialize the embedding model
    :param embed_size: size of embeddings
    :return: compiled model
    """
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

"""
TODO: Define a Callback function that stops training if all val_loss is below threshold
"""


def train_embed_model(model, X_train_list, Y_train_list):
    """
    Trains the embedding model on the given training data.
    :param model: compiled model
    :param X_train_list: List with X_train entries
    :param Y_train_list: List of Y_train entires
    :return: trained model
    """
    epoch_num = 30
    batch_size = 30
    print("Training size: ", X_train_list[0].shape)

    EarlyStop = EarlyStopping(monitor='val_loss_1', min_delta=0.01)

    model.fit(X_train_list,
              Y_train_list,
              epochs = epoch_num,
              batch_size = batch_size,
              callbacks=[EarlyStop],
              verbose=2)

    return model

def init_and_pretrain_embed_model(embed_size, pre_train_games):
    """
    Initialize and pretrain an embedding model.
    :param embed_size: integer of embedding size
    :param pre_train_games: number of games to pre-train on
    :return: compiled and trained model
    """
    # Plays the initial random games
    games = initial_random_plays(pre_train_games)
    # Transform the games list to trainable data
    X_train, Y_train = games_to_trainable(games)
    # Initialize the embedding model
    model = init_embed_model(embed_size)
    # Train the embedding model on the game data
    model = train_embed_model(model, X_train, Y_train)
    return model, X_train, Y_train

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
