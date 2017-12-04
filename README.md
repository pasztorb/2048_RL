# Deep Q-Learning with different embeddings for 2048
###### This repository contains my Deep Q-Learning implementation that learns how to play the mobile game 2048.

The agent is based on a vanilla Deep Q-network which tries to determine which direction to swipe to at the given state.
The main concern, however, is the representation of the game. Since the game consists a 4x4 table, it gives 16 tiles and each tile can attain one state out of about 20 possible states. The most straight-forward solution is then to store the states in a 20x4x4 binary array, but it is rather sparse to feed into a neural network.

In order to reduce the sparsity of this matrix, I have implemented the agent with three simple representation; the simple binary array, a linear embedding that takes values between zero and one in an increasing order and a third embedding that uses three layers with interpolation values of sine, cosine and a logit function. Also, there is a more complex embedding implementation that uses similar technique as word-embeddings to reduce the dimensionality based on hisotrical states.
