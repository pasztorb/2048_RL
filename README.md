# Deep Q-Learning with different embeddings for 2048
###### This repository contains my Deep Q-Learning implementation that learns how to play the mobile game 2048.

The agent is based on a vanilla Deep Q-Network which tries to determine which direction to swipe to at the given state.
The main concern, however, is the representation of the game. Since the game consists a 4x4 table, it gives 16 tiles and each tile can attain one state out of about 20 possible states. The most straight-forward solution is then to store the states in a 20x4x4 binary array, but it is rather sparse to feed into a neural network.

In order to reduce the sparsity of this matrix, I have implemented the agent with three simple representation; the simple binary array, a linear embedding that takes values between zero and one in an increasing order and a third embedding that uses three layers with interpolation values of sine, cosine and a logit function. The basic approach is to use a convolutional network as interpret the embeddings as channels, but I have added one more option a flat embeddings which flatten the binary array and uses simple feed-forward network instead.

> game.py

This file contains the functions to play the game. The game can be played via two variables a 20x4x4 numpy array to store the current state and the score variable related to the state. The actions are defined as the integers 0,1,2,3 in a clockwise manner that starts at 12 o'clock.

> reshape_and_NN.py

This file contains the functions to initialize and run the neural networks, also the functions which transforms the original state into the embeddings are in this file.

> model.py epochs reshape_type output_path

This file contains the implementation of the Deep Q-Network. The additional sys arguments define the number of learning epochs, the reshape_type is the variable of the embedding chosen from the set {'onehot', 'linear', 'trig', 'flat'} and the output_path to use in saving the training data in hdf5 format.

> embedding_2048.py

This file contains the implementation of the word-embedding like approach. It is a network that combines the continuous bag-of-words and the skip-gram models. Its input is 32 vectors from the two neighbour states and it predicts 16 output vectors which are the current state.

> embed_model.py epochs embedding_size output_path

This file contains the implementation of the word-embedding based Deep Q-Network. It is similar to the model.py but it has an embedding layer first on the 20x4x4 array that reduces the number of channels. The agent runs a given number of random games and then learns the basic embedding then it updates it after given timesteps.
