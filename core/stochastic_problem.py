"""
Abstract base class for stochastic sequential decision problems (Markov decision processes).
"""

import random
from abc import ABC, abstractmethod
from typing import List, Tuple
from core.game_state import GameState


class StochasticProblem(ABC):
    """
    A problem in which an agent repeatedly chooses an action and the environment answers
    with a reward and a (possibly random) next state.

    Every transition is split in two parts:
      1. apply_action: the deterministic effect of the agent's action. It produces an
         *afterstate* and the immediate reward.
      2. a chance event that turns the afterstate into the next state
         (get_chance_outcomes lists every possible result with its probability;
         sample_chance_outcome draws one of them at random).

    Search agents (such as minimax) may use the whole model.
    Learning agents (Q-learning) do NOT know the probabilities: they must only use
    get_initial_state, get_actions, is_terminal and step.
    """

    @abstractmethod
    def get_initial_state(self, rng: random.Random) -> GameState:
        """Return a (possibly random) initial state"""
        pass

    @abstractmethod
    def get_actions(self, state: GameState) -> List[str]:
        """Return the legal actions in a state (an empty list if the state is terminal)"""
        pass

    def is_terminal(self, state: GameState) -> bool:
        """A state is terminal when no legal action is left"""
        return not self.get_actions(state)

    @abstractmethod
    def apply_action(self, state: GameState, action: str) -> Tuple[GameState, float]:
        """
        Deterministic part of a transition.
        Returns (afterstate, reward): the state right after the action, before the
        chance event, and the reward obtained by the action.
        """
        pass

    @abstractmethod
    def get_chance_outcomes(self, afterstate: GameState) -> List[Tuple[GameState, float]]:
        """
        Random part of a transition.
        Returns every possible next state as a list of (next_state, probability) pairs.
        Probabilities add up to 1.
        """
        pass

    def sample_chance_outcome(self, afterstate: GameState, rng: random.Random) -> GameState:
        """Draw one next state according to the probabilities of get_chance_outcomes"""
        outcomes = self.get_chance_outcomes(afterstate)
        u = rng.random()
        accumulated = 0.0
        for next_state, probability in outcomes:
            accumulated += probability
            if u < accumulated:
                return next_state
        return outcomes[-1][0]

    def step(self, state: GameState, action: str, rng: random.Random) -> Tuple[float, GameState, bool]:
        """
        Environment interface used by learning agents: perform the action and return
        (reward, next_state, done), where done tells whether next_state is terminal.
        """
        afterstate, reward = self.apply_action(state, action)
        next_state = self.sample_chance_outcome(afterstate, rng)
        return reward, next_state, self.is_terminal(next_state)

    def heuristic(self, state: GameState) -> float:
        """
        Estimate of the reward still to be obtained from a non-terminal state.
        Used by depth-limited search when the depth runs out.
        """
        # Default implementation: no information
        return 0
