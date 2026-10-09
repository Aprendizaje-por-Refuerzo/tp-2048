"""
Helpers to play games and evaluate agents.
"""

import random
from collections import Counter
from typing import Any, Dict, Tuple
from core.agent import Agent
from core.game_state import GameState
from core.stochastic_problem import StochasticProblem


def play_game(problem: StochasticProblem, agent: Agent, rng: random.Random,
              verbose: bool = False) -> Tuple[float, GameState, int]:
    """
    Play one complete game.

    Args:
        problem: The game to play
        agent: The agent that chooses the actions
        rng: Random number generator used by the environment
        verbose: Whether to print every move

    Returns:
        (total reward, final state, number of moves)
    """
    state = problem.get_initial_state(rng)
    score = 0.0
    moves = 0
    if verbose:
        print("Initial state:")
        print(state)
    while not problem.is_terminal(state):
        action = agent.choose_action(state)
        reward, state, _ = problem.step(state, action, rng)
        score += reward
        moves += 1
        if verbose:
            print(f"Move {moves:3}: {action:6} (+{reward:g}) → score {score:g}")
            print(state)
    return score, state, moves


def evaluate_agent(problem: StochasticProblem, agent: Agent, games: int = 1000,
                   seed: int = 7) -> Dict[str, Any]:
    """
    Play several games and summarize the results.

    Returns:
        Dictionary with the mean score, the mean number of moves and, if the
        final states have a max_tile() method, how often each maximum tile was reached.
    """
    rng = random.Random(seed)
    total_score = 0.0
    total_moves = 0
    max_tiles = Counter()
    for _ in range(games):
        score, final_state, moves = play_game(problem, agent, rng)
        total_score += score
        total_moves += moves
        if hasattr(final_state, "max_tile"):
            max_tiles[final_state.max_tile()] += 1
    return {
        "games": games,
        "mean_score": total_score / games,
        "mean_moves": total_moves / games,
        "max_tiles": dict(sorted(max_tiles.items())),
    }
