"""
Functions to play 2048
The game stores current state as a numpy array and the score as an integer
"""

import numpy as np

def visualize_state(state, score):
    """
    Visualize the current given state
    :param state: 4x4 numpy array
    :return: None
    """
    print("Current score: ",str(score))
    for i in range(4): # Iterate over the rows
        print(list(state[i,:].astype(int)))

def add_tile(state):
    """
    Funtion that adds a tile randomly to the given state
    :param state: 4x4 numpy array
    :return: 4x4 numpy array
    """
    # possible index pairs to put number in it
    possible_tiles = np.where(state==0)
    # If there is no possible tile, it returns the original state
    if len(possible_tiles[0]) != 0:
        # choose one randomly
        index = np.random.randint(len(possible_tiles[0]))
        # add a tile 2 to the chosen tile
        state[possible_tiles[0][index], possible_tiles[1][index]] = 2
    return state

def initialize_game():
    """
    This function initialize a 4x4 game with two randomly added tile
    :return: 4X4 numpy array, score: 0
    """
    # empty game table
    state = np.zeros((4,4))
    # add two tile randomly
    for i in range(2):
        state = add_tile(state)
    return state, 0

def shift_row(row, score):
    """
    Function that makes a move on the given row.
    :param row: (4,) numpy array
    :return: (4, updated numpy array
    """
    # new array to store the changed row and score
    row_updated = row.copy()
    score_updated = score

    # Checks if there is possible merge in the row
    possible_merge = False
    unique, counts = np.unique(row, return_counts=True)
    for u,c in zip(unique, counts):
        if (c > 1) and (u != 0):
            possible_merge = True
    # If merge is possible, iterates over the possible patterns
    if possible_merge:
        for case in [(0,3),(0,2),(0,1),(1,3),(1,2),(2,3)]:
            # If the tiles are the same at the two position and they are not zeros, but all the entries between them are zero
            if (row_updated[case[0]] == row_updated[case[1]]) and (row_updated[case[0]] != 0) and (row_updated[range(case[0]+1,case[1])].sum()==0):
                # Update tiles
                row_updated[case[0]] = row[case[0]]*2
                row_updated[case[1]] = 0
                # Update score
                score_updated += row[case[0]]*2

    # Shift tiles
    for i in reversed(range(3)):
        if row_updated[i] == 0:
            row_updated[i:] = np.roll(row_updated[i:],-1)

    return row_updated, score_updated


def action_move(state, score, move, end_check = True):
    """
    Function to make the given move, add a new tile, update the score value and return continuing = False if there is no possible further move
    move: 0 - up, 1 - right, 2 - down, 3 - left
    :param state: 4x4 array, score: current score, move: move to make on the state
    :param end_check: checks if the game has ended or not
    :return: state, score, running
    """
    assert move in [0,1,2,3]
    # Copy the state in order to keep the move from immediately change the state of the game
    new_state = state.copy()
    # Copy the original score so it won't update it automatically the original variable
    new_score = score

    # updates all row given the move
    for i in range(4):
        if move == 0:
            new_state[:,i], new_score = shift_row(new_state[:,i], new_score)
        elif move == 1:
            new_state[i,::-1], new_score = shift_row(new_state[i,::-1], new_score)
        elif move == 2:
            new_state[::-1, i], new_score = shift_row(new_state[::-1, i], new_score)
        else:
            new_state[i, :], new_score = shift_row(new_state[i, :], new_score)

    # If something has changed add a new tile
    if (new_state==state).sum() != 16:
        new_state = add_tile(new_state)

    running = True
    # Checks if any further move is possible
    if end_check:
        running = False
        # If there is at least one zero in the state go on, since move is possible
        if (new_state==0).sum() == 0:
            # Check horizontal moves
            if (new_state[:,:-1]==new_state[:,1:]).sum() != 0:
                running = True
            # Check vertical moves
            elif (new_state[:-1,:]==new_state[1:,:]).sum() != 0:
                running = True
        else:
            running = True

    return new_state, new_score, running


def test_play(model = None, reshape_function = None, visualize=True):
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
    action_list = []
    # Play the game based on the neural network
    while running:
        # If model is given made move based on it, otherwise randomly
        if model != None:
            qval = model.predict(reshape_function(game), batch_size=1)
            action = np.argmax(qval)
        else:
            action = np.random.randint(0, 4)
        # Append action to the list
        action_list += [action]
        # Make move
        game, score, running = action_move(game, score, action)
        score_list += [score]
        game_list += [game]
        # if visualize is given, then print state after each move
        if visualize:
            if model == None:
                print(action)
            else:
                print(qval)
            visualize_state(game, score)
        # If it traps into a pit where it makes the same move over and over it terminates the game (no change for 20 moves)
        if score_list[-20:] == [score]*20:
            if visualize:
                print("Not good enough, trapped")
            running = False
    return game_list, score_list, action_list


def avg_test_plays(plays_num, model=None ,reshape_function=None):
    """
    Plays the given number of plays and returns statistics based on them.
    :param plays_num: integer, number of plays to play
    :param model: network to play based on, if none it plays randomly
    :param reshape_function: reshape function that works with the model
    :return: list of scores
    """
    list_of_scores = []
    for i in range(plays_num):
        _, scores, _ = test_play(model, reshape_function, visualize=False)
        list_of_scores += [scores[-1]]

    return list_of_scores
