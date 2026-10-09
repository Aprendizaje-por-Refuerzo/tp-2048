import random
from typing import Optional
from core.agent import Agent
from core.game_state import GameState
from core.stochastic_problem import StochasticProblem


class RandomAgent(Agent):
    """
    Agent that chooses uniformly at random among the legal actions.
    Useful as a baseline: any reasonable agent should beat it.
    """

    def __init__(self, problem: StochasticProblem, seed: Optional[int] = None):
        super().__init__(problem)
        self.rng = random.Random(seed)

    def choose_action(self, state: GameState) -> str:
        return self.rng.choice(self.problem.get_actions(state))
