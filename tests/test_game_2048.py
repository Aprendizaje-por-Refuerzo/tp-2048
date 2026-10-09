"""
Tests of the provided 2048 implementation. They pass from the start: use them to
understand the rules and the interface of the game.
"""

import random
import unittest
from problems import Game2048Problem, Game2048State


def board(*cells):
    return Game2048State(tuple(cells), 2, 2)


class TestSlideLine(unittest.TestCase):

    def test_merge_two_equal_tiles(self):
        self.assertEqual(Game2048Problem.slide_line([1, 1]), ([2, 0], 4))

    def test_slide_without_merge(self):
        self.assertEqual(Game2048Problem.slide_line([0, 3]), ([3, 0], 0))

    def test_each_tile_merges_once(self):
        # 2 2 2 2 -> 4 4 (not 8), and 4 2 2 -> 4 4 (the new 4 does not merge again)
        self.assertEqual(Game2048Problem.slide_line([1, 1, 1, 1]), ([2, 2, 0, 0], 8))
        self.assertEqual(Game2048Problem.slide_line([2, 1, 1]), ([2, 2, 0], 4))

    def test_gaps_are_removed_before_merging(self):
        self.assertEqual(Game2048Problem.slide_line([0, 1, 0, 1]), ([2, 0, 0, 0], 4))


class TestGame2048Problem(unittest.TestCase):

    def setUp(self):
        self.problem = Game2048Problem()

    def test_apply_action(self):
        # | 2 2 |   Left   | 4 . |
        # | . 4 |  ----->  | 4 . |   and 4 points
        afterstate, reward = self.problem.apply_action(board(1, 1, 0, 2), "Left")
        self.assertEqual(afterstate, board(2, 0, 2, 0))
        self.assertEqual(reward, 4)

    def test_legal_actions(self):
        # Up does not change | 2 2 / . 4 |, so it is not legal
        self.assertEqual(self.problem.get_actions(board(1, 1, 0, 2)), ["Left", "Right", "Down"])

    def test_terminal_state(self):
        # | 2 4 |
        # | 4 2 |  no move changes the board
        self.assertTrue(self.problem.is_terminal(board(1, 2, 2, 1)))
        self.assertEqual(self.problem.get_actions(board(1, 2, 2, 1)), [])

    def test_chance_outcomes(self):
        outcomes = dict(self.problem.get_chance_outcomes(board(2, 0, 2, 0)))
        self.assertEqual(len(outcomes), 4)
        self.assertAlmostEqual(outcomes[board(2, 1, 2, 0)], 0.45)
        self.assertAlmostEqual(outcomes[board(2, 2, 2, 0)], 0.05)
        self.assertAlmostEqual(outcomes[board(2, 0, 2, 1)], 0.45)
        self.assertAlmostEqual(outcomes[board(2, 0, 2, 2)], 0.05)
        self.assertAlmostEqual(sum(outcomes.values()), 1.0)

    def test_step_returns_a_possible_outcome(self):
        rng = random.Random(0)
        state = board(1, 1, 0, 2)
        afterstate, expected_reward = self.problem.apply_action(state, "Left")
        possible = {s for s, _ in self.problem.get_chance_outcomes(afterstate)}
        for _ in range(50):
            reward, next_state, done = self.problem.step(state, "Left", rng)
            self.assertEqual(reward, expected_reward)
            self.assertIn(next_state, possible)
            self.assertEqual(done, self.problem.is_terminal(next_state))

    def test_initial_state_has_two_tiles(self):
        rng = random.Random(0)
        for _ in range(20):
            state = self.problem.get_initial_state(rng)
            self.assertEqual(len(state.empty_cells()), 2)

    def test_larger_boards(self):
        problem = Game2048Problem(rows=1, cols=4)
        afterstate, reward = problem.apply_action(Game2048State((1, 1, 1, 1), 1, 4), "Left")
        self.assertEqual(afterstate, Game2048State((2, 2, 0, 0), 1, 4))
        self.assertEqual(reward, 8)


if __name__ == "__main__":
    unittest.main()
