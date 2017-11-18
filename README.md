# 2048_RL
This repository is made to track my progress with a reinforcement learning agent that learns to play the mobile game called 2048.

The agent is based on a vanilla deep Q-network which simply tries to determine which direction to swipe to at the given state.
The main concern, however, is the representation of the game.
Since the game consists a 4x4 table, it gives 16 tiles and each tile can attain one state out of about 20 possible states.
The most straight-forward solution is then to store the states in a 20x4x4 binary array, which can be easily feed into the approximation network.
Either as a flat 320 long vector after flattening the state array or as a 4x4 image with 20 channels for a convolutional network.
In both cases, the input is quite sparse (20 ones and 300 zeros), which makes the training slow,
therefore I am trying to find a new representation that reduces the sparsity of the matrix.
The easier ideas are to convert the 20 channels to either a linear channel or several channels in which each channels consists interpolation values of the a sine, a cosine and a sigmoid function.
Hopefully these solutions already helps the learning, but I have implemented an other idea that uses similar approach as the word-embedding models.

