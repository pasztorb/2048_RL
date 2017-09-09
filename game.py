"""
Functions to play 2048
"""

import numpy as np

game_shape = (20,4,4)

def visualize_state(state, score):
    """
    Visualize the current given state
    :param state: current state of the game
    :return: None
    """
    print("Current score: ",str(score))
    for i in range(4): # Iterate over the rows
        row = [0,0,0,0]
        for j in range(4): # Iterate over the columns
            value = np.where(state[:,i,j]==1)[0]
            if len(value)>1:
                print("Not valid state")
            if len(value) == 1:
                row[j] = 2**(value[0]+1)
        print(row)

def add_tile(state):
    """
    Funtion that adds a tile randomly to the given state
    :param state: current state of the game
    :return: new state of the game with the added tile
    """
    # sum of the array along the first array, if 0 it is empty if 1 it has a number in it
    used_tiles = state.sum(axis=0)
    # possible index pairs to put number in it
    possible_tiles = np.where(used_tiles==0)
    # If there is no possible tile, it returns the original state
    if len(possible_tiles[0]) == 0:
        return state
    else:
        # choose one randomly
        index = np.random.randint(len(possible_tiles[0]))
        # add a tile 2 to the chosen tile
        state[0,possible_tiles[0][index],possible_tiles[1][index]] = 1
        return state

def initialize_game():
    """
    This function initialize a 4x4 game with two randomly added tile
    :return: 20X4X4 array of the current state, score integer: 0
    """
    # empty game grid
    state = np.zeros((game_shape[0],game_shape[1],game_shape[2]))
    # add two tile randomly
    for i in range(2):
        state = add_tile(state)
    return state, 0

def shift_row(row, score):
    """
    Function that makes a move on the given row
    :param row: 20x4 numpy array that represents a row
    :return: row_updated
    """
    # new array to store the changed row and score
    row_updated = row.copy()
    score_updated = score

    # Sums up the first axis
    row_sum = row_updated.sum(axis=0)
    # If there is no entry in the row:
    if row_sum.sum() == 0:
        return row_updated, score_updated
    # If there is at least one entry continue
    # Shift the entries
    columns_to_drop = []
    for i in range(4):
        # If there is no value in the entry delete that
        if row_sum[i] == 0:
            columns_to_drop += [i]
    row_updated = np.delete(row_updated, columns_to_drop, 1)
    # Add rows to fill the dropped
    row_updated = np.concatenate([row_updated, np.zeros((game_shape[0],len(columns_to_drop)))], axis=1)

    # Iterate over the all possible merging place
    temp_row = row_updated.copy() # Additional copy to avoid several merging of one tile
    for i in range(1,row_updated.shape[0]):
        for j in range(row_updated.shape[1]-1):
            # If the merge happens update the tiles
            if row_updated[i-1,j]*row_updated[i-1,j+1] == 1:
                temp_row[i,j] = 1
                temp_row[i - 1, j] = 0
                temp_row[i - 1, j + 1] = 0
                row_updated[i-1,j+1] = 0 # Avoid to merge 3 similar into two new tiles
                score_updated += 2**(i+1)
    row_updated = temp_row # update row_update variable

    # Shift again to fill the holes made by merging
    row_sum = row_updated.sum(axis=0)
    columns_to_drop = []
    for i in range(4):
        # If there is no value in the entry delete that
        if row_sum[i] == 0:
            columns_to_drop += [i]
    row_updated = np.delete(row_updated, columns_to_drop, 1)
    # Add rows to fill the dropped
    row_updated = np.concatenate([row_updated, np.zeros((game_shape[0],len(columns_to_drop)))], axis=1)

    return row_updated, score_updated


def action_move(state, score, move, end_check = True):
    """
    Function to make the given move, add a new tile, update the score value and return continuing = False if there is no possible further move
    move: 0 - up, 1 - right, 2 - down, 3 - left
    :param state: current state, score: current score,move: move to make on the state
    :return: state, score, running
    """
    assert move in [0,1,2,3]
    # Copy the state in order to keep the move from immediately change the state of the game
    new_state = state.copy()
    # Copy the original score so it won't update it automatically the original variable
    new_score = score
    for i in range(4): # Iterate over the chosen axis
        if move == 0: # Up move
            # Shifts each column
            new_state[:,:,i], new_score = shift_row(new_state[:,:,i], new_score)

        if move == 2: # Down move
            # Shifts the revered columns since the shift is downward
            new_state[:, :, i], new_score = shift_row(new_state[:, ::-1, i], new_score)
            new_state[:, :, i] = new_state[:, ::-1, i] # Reverts back the column

        if move == 3: # Left move
            # Shifts each row
            new_state[:,i,:], new_score = shift_row(new_state[:,i,:], new_score)

        if move == 1: # Right move
            # Shifts each reverted rows and the revert them back
            new_state[:, i, :], new_score = shift_row(new_state[:, i, ::-1], new_score)
            new_state[:,i,:] = new_state[:,i,::-1]

    # Add a new tile only if there has been change in the placement of the original tiles
    if (new_state == state).sum() != game_shape[0]*game_shape[1]*game_shape[2]:
        new_state = add_tile(new_state)

    # Check if the game can continue (no change can happen after the 4 different move)
    if end_check:
        running = False
        # If there is difference after one of the move then there is possible move so it is not yet the end of the game
        for i in range(4):
            state_next, _ , _ = action_move(new_state, score, i, end_check=False)
            if (new_state == state_next).sum() != game_shape[0]*game_shape[1]*game_shape[2]:
                running = True
    else:
        running = True

    return new_state, new_score, running


def test_play(model = None, reshape_function = None,visualize=True):
    """
    Function to play a game based on the feeded model or randomly.
    :param model: If given the function plays a game based on its output.
    :param reshape_function: If model is given, a reshape function has to be given as well that transforms the array representing the state into an array that can be fed into the neural net
    :param visualize: If set to be True it visualizes the game after each step with the Q-values.
    :return: Returns two list with the score at each timestep and the arrays representing the states.
    """
    # Set up a new game
    game, score = initialize_game()
    running = True
    score_list = [0]
    game_list = [game]
    # Play the game based on the neural network
    while running:
        if model != None:
            qval = model.predict(reshape_function(game), batch_size=1)
            action = np.argmax(qval)
        else:
            action = np.random.randint(0, 4)
        game, score, running = action_move(game, score, action)
        score_list += [score]
        game_list += [game]
        if visualize:
            print(qval)
            visualize_state(game, score)
        # If it traps into a pit where it makes the same move over and over it terminates the game
        if score_list[-10:] == [score]*10:
            print("Not good enough, trapped")
            running = False
    return score_list, game_list
