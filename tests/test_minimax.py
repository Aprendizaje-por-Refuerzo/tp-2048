"""
Tests for Parte 1 (minimax).
The expected values were computed by hand or with a reference implementation,
using the heuristic provided in Game2048Problem (sum of the tiles).
"""

import unittest
from problems import Game2048Problem, Game2048State
from algorithms import MinimaxAgent


def board(*cells):
    return Game2048State(tuple(cells), 2, 2)


class TestMinimax(unittest.TestCase):

    def setUp(self):
        self.problem = Game2048Problem()
        self.agent = MinimaxAgent(self.problem, depth=1)

    def test_terminal_state_is_worth_zero(self):
        for depth in (0, 1, 2):
            self.assertEqual(self.agent.max_value(board(1, 2, 2, 1), depth), 0,
                             "Un estado final vale 0, cualquiera sea la profundidad")

    def test_depth_zero_uses_the_heuristic(self):
        state = board(1, 1, 0, 2)
        self.assertEqual(self.agent.max_value(state, 0), self.problem.heuristic(state),
                         "Con profundidad 0, un estado no final vale la heurística")

    def test_min_node_takes_the_worst_outcome(self):
        # | 2 4 |  Left  | 2 4 |  The new tile appears in the empty cell:
        # | 4 4 |  --->  | 8 . |  a 2 ends the game (0), a 4 leaves a live board (18).
        # Minimax assumes the worst outcome: 0.
        self.assertEqual(self.agent.chance_value(board(1, 2, 3, 0), 1), 0)
        # After Right the cases are reversed: a 2 leaves a live board (16), a 4 ends it.
        self.assertEqual(self.agent.chance_value(board(1, 2, 0, 3), 1), 0)

    def test_values_depth_1(self):
        self.assertEqual(self.agent.max_value(board(1, 1, 0, 2), 1), 14)
        self.assertEqual(self.agent.max_value(board(0, 1, 0, 2), 1), 8)
        self.assertEqual(self.agent.max_value(board(1, 0, 0, 1), 1), 6)
        self.assertEqual(self.agent.max_value(board(3, 2, 1, 1), 1), 22)

    def test_values_depth_2_and_3(self):
        self.assertEqual(self.agent.max_value(board(1, 1, 0, 2), 2), 24)
        self.assertEqual(self.agent.max_value(board(1, 1, 0, 2), 3), 12)
        self.assertEqual(self.agent.max_value(board(1, 0, 0, 1), 2), 12)
        self.assertEqual(self.agent.max_value(board(1, 0, 0, 1), 3), 14)
        self.assertEqual(self.agent.max_value(board(3, 2, 1, 1), 3), 28)

    def test_choose_action(self):
        self.assertEqual(MinimaxAgent(self.problem, depth=2).choose_action(board(3, 2, 1, 1)), "Right")
        self.assertEqual(MinimaxAgent(self.problem, depth=2).choose_action(board(1, 1, 0, 2)), "Left")

    def test_expanded_nodes_are_counted(self):
        agent = MinimaxAgent(self.problem, depth=2)
        agent.choose_action(board(1, 0, 0, 1))
        self.assertGreater(agent.expanded_nodes, 0, "No borrar la línea que cuenta los nodos en max_value")



if __name__ == "__main__":
    unittest.main()
