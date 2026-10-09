"""
Abstract base class for all agents.
"""

from abc import ABC, abstractmethod
from core.stochastic_problem import StochasticProblem
from core.game_state import GameState


class Agent(ABC):
    """
    Abstract base class for agents.
    All agents should inherit from this class.
    """

    def __init__(self, problem: StochasticProblem):
        """
        Initialize the agent with the problem it plays.

        Args:
            problem: The stochastic problem (game) the agent plays
        """
        self.problem = problem

    @abstractmethod
    def choose_action(self, state: GameState) -> str:
        """
        Choose the action to perform in a non-terminal state.

        Args:
            state: Current state (it has at least one legal action)

        Returns:
            One of problem.get_actions(state)
        """
        pass

    @property
    def name(self) -> str:
        """Name used when printing results"""
        return self.__class__.__name__
