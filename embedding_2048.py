from game import *
import sys
import h5py

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, concatenate
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def random_plays(num_plays, output_folder):
    """
    This function randomly plays num_plays number of games.
    Saves the states in all games and add them to the games.hdf5 file in the output folder.
    :param num_plays: integer variable to give how many random games to play
    :param output_folder: path indicating the folder in which the games.hdf5 can be found
    :return: Adds to the hdf5 file as a (None,20,4,4) array
    """
    with h5py.File(output_folder + "games.hdf5", 'a') as f:
        # Finds the games directory in the hdf5 file. If it does not exists makes one and adds the new_games attribute
        try:
            games_dir = f['games']
        except:
            games_dir = f.create_group('games')
        # Play the given number of games
        for i in range(num_plays):
            # Plays a random game
            _, states = test_play(visualize=False)
            states = np.array(states)
            # Counts the number of games already in the folder
            num_games = len(list(games_dir.keys()))
            # Names the new game as the next one
            name = 'game'+str(num_games)
            # Add the game into the hdf5
            games_dir[name] = states


def games_to_trainable(folder):
    """
    This function takes a list of games and turn it into a trainable format
    :param folder: Folder in which the games.hdf5 can be found.
    :return: Appends the X_train and Y_train to the given datasets in the hdf5 file
    """
    # Extract the lists from the games
    X_train = [[],[]]
    Y_train = []

    # At each step add the current state to the X_train list's first list and the current state+2 to the second list
    # At each step add the current state + 1 into the Y_train list reshaped to (20,16), i.e. each column represent one tile vector
    # First extracts the games_dir from the hdf5 file
    with h5py.File(folder + "games.hdf5", 'a') as f:
        games_dir = f['games']

        # For each (except the first and the last) game it appends the previous and the next game states to X_train the the current to Y_train
        for name in games_dir:
            # current game
            game = games_dir[name]
            # Add the states/rows to the related datasets
            for g in range(game.shape[0]-2):
                X_train[0].append(game[g,:,:,:])
                X_train[1].append(game[g+2,:,:,:])
                Y_train.append(game[g,:,:,:].reshape((20,16)))

        # Remove the games group and add a new empty
        del f['games']
        f.create_group('games')

        # Make one array from the several arrays
        X_train = [np.array(x) for x in X_train]
        Y_train = np.array(Y_train)
        # Split the Y_train array into 16 different lists for the different columns
        Y_train = [Y_train[:, :, i] for i in range(16)]

        # If training datasets are already exists append to them, if not create the datasets
        if 'X_train_0' in f:
            f['X_train_0'].resize(f['X_train_0'].shape[0]+X_train[0].shape[0], axis=0)
            f['X_train_0'][-X_train[0].shape[0]: , :, :, :] = X_train[0]

            f['X_train_1'].resize(f['X_train_1'].shape[0] + X_train[1].shape[0], axis=0)
            f['X_train_1'][-X_train[1].shape[0]:, :, :, :] = X_train[1]

            for i in range(16):
                name = 'Y_train_'+str(i)
                f[name].resize(f[name].shape[0]+Y_train[i].shape[0], axis=0)
                f[name][-Y_train[i].shape[0]:, :] = Y_train[i]
        else:
            f.create_dataset(name='X_train_0',data=X_train[0], maxshape=(None,20,4,4), chunks =(1,20,4,4))
            f.create_dataset(name='X_train_1',data=X_train[1], maxshape=(None,20,4,4), chunks =(1,20,4,4))
            for i in range(16):
                f.create_dataset(name='Y_train_'+str(i),data=Y_train[i],maxshape=(None,20), chunks =(1,20))


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

def train_embed_model(model, folder):
    """
    Trains the embedding model on the given training data.
    :param model: compiled model
    :param folder: Folder in which the games.hdf5 can be found.
    :return: trained model
    """
    # Parameters
    epoch_num = 30
    batch_size = 30

    # Extract the training datasets from the given hdf5 file
    X_train_list = []
    Y_train_list = []

    with h5py.File(folder + "games.hdf5", 'r') as f:
        X_train_list.append(f['X_train_0'][()])
        X_train_list.append(f['X_train_1'][()])
        for i in range(16):
            name = 'Y_train_'+str(i)
            Y_train_list.append(f[name][()])

    print("Training size: ", X_train_list[0].shape)

    EarlyStop = EarlyStopping(monitor='loss', min_delta=0.005,patience=2)

    model.fit(X_train_list,
              Y_train_list,
              epochs = epoch_num,
              batch_size = batch_size,
              callbacks=[EarlyStop],
              verbose= 2)

    return model

def tsne_embedding_plot(embed):
    """
    Plots the weights of the embeddings
    :param embed: embedding layer
    :return: matplotlib plot
    """
    weights = embed.get_weights()[0]
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    for i in range(weights.shape[2]):
        tokens.append(weights[0,0,i,:])
        labels.append(str(2**i))
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(10, 10))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

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
