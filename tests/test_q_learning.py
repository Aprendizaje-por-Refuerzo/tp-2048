"""
Tests for Parte 2 (Q-learning).
"""

import unittest
from collections import Counter
from problems import Game2048Problem, Game2048State
from algorithms import QLearningAgent
from core import evaluate_agent


def board(*cells):
    return Game2048State(tuple(cells), 2, 2)


class NoModelProblem(Game2048Problem):
    """The same game, but asking for the probabilities is an error."""

    def get_chance_outcomes(self, afterstate):
        raise AssertionError("Q-learning no debe usar get_chance_outcomes: no conoce las probabilidades")


class TestQLearningUpdate(unittest.TestCase):

    def setUp(self):
        self.problem = Game2048Problem()
        self.agent = QLearningAgent(self.problem, alpha=0.5, gamma=0.9, epsilon=0.1, seed=0)
        self.s = board(1, 0, 0, 1)
        self.agent.q[(self.s, "Left")] = 2.0

    def test_non_terminal_update_uses_max_over_legal_actions(self):
        next_state = board(1, 1, 0, 2)                 # legal: Left, Right, Down
        self.agent.q[(next_state, "Left")] = 10.0
        self.agent.q[(next_state, "Down")] = -5.0
        self.agent.q[(next_state, "Up")] = 100.0       # Up is NOT legal in next_state
        self.agent.update(self.s, "Left", 4.0, next_state, False)
        # target = 4 + 0.9 * 10 = 13;  Q = 2 + 0.5 * (13 - 2) = 7.5
        self.assertAlmostEqual(self.agent.get_q(self.s, "Left"), 7.5,
                               msg="El máximo debe tomarse sólo sobre las acciones legales del siguiente estado")

    def test_terminal_update_uses_only_the_reward(self):
        next_state = board(1, 2, 2, 1)                 # terminal
        self.agent.q[(next_state, "Left")] = 100.0     # must be ignored
        self.agent.update(self.s, "Left", 4.0, next_state, True)
        # target = 4;  Q = 2 + 0.5 * (4 - 2) = 3
        self.assertAlmostEqual(self.agent.get_q(self.s, "Left"), 3.0,
                               msg="Si el siguiente estado es final, el objetivo es sólo la recompensa")

    def test_unknown_pairs_are_worth_zero(self):
        next_state = board(0, 1, 0, 2)                 # never updated
        self.agent.update(self.s, "Right", 4.0, next_state, False)
        # Q(s, Right) = 0 + 0.5 * (4 + 0.9 * 0 - 0) = 2
        self.assertAlmostEqual(self.agent.get_q(self.s, "Right"), 2.0)

    def test_update_changes_only_one_entry(self):
        next_state = board(1, 1, 0, 2)
        self.agent.q[(next_state, "Left")] = 10.0
        before = dict(self.agent.q)
        self.agent.update(self.s, "Left", 4.0, next_state, False)
        changed = {k for k in set(before) | set(self.agent.q) if before.get(k) != self.agent.q.get(k)}
        self.assertEqual(changed, {(self.s, "Left")})


class TestEpsilonGreedy(unittest.TestCase):

    def setUp(self):
        self.problem = Game2048Problem()
        self.state = board(1, 1, 0, 2)                 # legal: Left, Right, Down

    def test_epsilon_zero_is_greedy(self):
        agent = QLearningAgent(self.problem, epsilon=0.0, seed=0)
        agent.q[(self.state, "Left")] = 1.0
        agent.q[(self.state, "Right")] = 3.0
        agent.q[(self.state, "Down")] = 2.0
        agent.q[(self.state, "Up")] = 50.0             # not legal
        for _ in range(20):
            self.assertEqual(agent.epsilon_greedy_action(self.state), "Right")

    def test_epsilon_one_explores_only_legal_actions(self):
        agent = QLearningAgent(self.problem, epsilon=1.0, seed=0)
        agent.q[(self.state, "Right")] = 3.0
        counts = Counter(agent.epsilon_greedy_action(self.state) for _ in range(3000))
        self.assertEqual(set(counts), {"Left", "Right", "Down"},
                         "Con epsilon = 1 debe elegir al azar entre todas las acciones legales, y sólo entre ellas")
        for action, count in counts.items():
            self.assertGreater(count / 3000, 0.25, f"La acción {action} se elige muy poco: ¿es uniforme?")


class TestQLearningTraining(unittest.TestCase):

    def test_does_not_use_the_model(self):
        agent = QLearningAgent(NoModelProblem(), seed=0)
        agent.train(200)

    def test_learns_to_play(self):
        problem = Game2048Problem()
        agent = QLearningAgent(problem, seed=1)
        agent.train(5000)
        result = evaluate_agent(problem, agent, games=2000, seed=7)
        # A random agent scores about 20 points; the reference implementation, about 59.
        self.assertGreater(result["mean_score"], 50,
                           "Tras 5000 episodios, la política voraz debería superar los 50 puntos de media")


if __name__ == "__main__":
    unittest.main()
