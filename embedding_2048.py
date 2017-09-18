from game import *

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
    for i in range(16):
        Y_train += [[]]

    for game in list_of_games:
        for g in range(len(game)-2):
            X_train[0].append(game[g])
            X_train[1].append(game[g+2])
            for i in range(16):
                Y_train[i].append(game[g + 1][:, i // 4, i % 4])

    X_train = [np.array(x) for x in X_train]
    Y_train = [np.array(y) for y in Y_train]

    for x in X_train:
        print(x.shape)

    for y in Y_train:
        print(y.shape)

    return X_train, Y_train



def init_embed_model():
    embed_size = 12

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
        outputs += [Dense(16,
                         activation = 'softmax',
                         name = 'output_'+str(i))(concat)]

    # Define model
    model = Model(inputs = [input_1, input_2], outputs = outputs)

    # compile model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')

    return model


def train_embed_model(model, X_train_list, Y_train_list):
    epoch_num = 100
    batch_size = 20

    model.fit(X_train_list,
              Y_train_list,
              epochs = epoch_num,
              batch_size = batch_size)

    return model


games = initial_random_plays(3)
X_train, Y_train = games_to_trainable(games)
model = init_embed_model()
train_embed_model(model, X_train, Y_train)