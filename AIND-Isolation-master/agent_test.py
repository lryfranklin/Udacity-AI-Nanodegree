"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent

from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = "Player1"
        self.player2 = "Player2"
        self.game = isolation.Board(self.player1, self.player2)

    def test_minimax_interface(self):
        """ Test CustomPlayer.minimax interface with simple input """
        h, w = 7, 7  # board size
        test_depth = 1
        starting_location = (5, 3)
        adversary_location = (0, 0)  # top left corner
        iterative_search = False
        search_method = "minimax"
        heuristic = lambda g, p: 0.  # return 0 everywhere

        # create a player agent & a game board
        agentUT = game_agent.MinimaxPlayer(game_agent.IsolationPlayer)
        agentUT.time_left = lambda: 99  # ignore timeout for fixed-depth search
        board = isolation.Board(agentUT, 'null_agent', w, h)

        # place two "players" on the board at arbitrary (but fixed) locations
        board.apply_move(starting_location)
        board.apply_move(adversary_location)

        for move in board.get_legal_moves():
            next_state = board.forecast_move(move)
            op_move = agentUT.minimax(next_state, test_depth)
            print("op_move = ")
            print(op_move)
            self.assertTrue(type(op_move) == tuple,
                            ("Minimax function should return a tuple " +
                             "point value approximating the score for the " +
                             "branch being searched."))

    def test_alphabeta_interface(self):
        """ Test AlphaBetaPlayer.alphabeta interface with simple input """
        h, w = 7, 7  # board size
        test_depth = 1
        starting_location = (5, 3)
        adversary_location = (0, 0)  # top left corner
        iterative_search = False
        search_method = "minimax"
        heuristic = lambda g, p: 0.  # return 0 everywhere

        # create a player agent & a game board
        agentUT = game_agent.AlphaBetaPlayer(game_agent.IsolationPlayer)
        agentUT.time_left = lambda: 99  # ignore timeout for fixed-depth search
        board = isolation.Board(agentUT, 'null_agent', w, h)

        # place two "players" on the board at arbitrary (but fixed) locations
        board.apply_move(starting_location)
        board.apply_move(adversary_location)

        for move in board.get_legal_moves():
            next_state = board.forecast_move(move)
            op_move = agentUT.alphabeta(next_state, test_depth)
            print("op_move = ")
            print(op_move)
            self.assertTrue(type(op_move) == tuple,
                            ("Minimax function should return a tuple " +
                            "point value approximating the score for the " +
                            "branch being searched."))



if __name__ == '__main__':
    unittest.main()
    #IsolationTest.test_minimax_interface(IsolationTest)
