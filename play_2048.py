"""
Play one game of (simplified) 2048 with an agent, printing every move.
Change the configuration below to try other agents or board sizes.
"""

import random
from problems import Game2048Problem
from algorithms import RandomAgent, MinimaxAgent, QLearningAgent
from core import play_game

# Configuration
ROWS, COLS = 2, 2
AGENT = "minimax"          # "random", "minimax" or "qlearning"
DEPTH = 2                  # for minimax
TRAINING_EPISODES = 30_000 # for qlearning
SEED = 7

problem = Game2048Problem(ROWS, COLS)
if AGENT == "random":
    agent = RandomAgent(problem, seed=0)
elif AGENT == "minimax":
    agent = MinimaxAgent(problem, depth=DEPTH)
elif AGENT == "qlearning":
    agent = QLearningAgent(problem)
    print(f"Training Q-learning for {TRAINING_EPISODES} episodes...")
    agent.train(TRAINING_EPISODES)
else:
    raise ValueError(f"Unknown agent: {AGENT}")

print("=" * 70)
print(f"2048 ({ROWS}x{COLS}) played by {agent.name}")
print("=" * 70)
score, final_state, moves = play_game(problem, agent, random.Random(SEED), verbose=True)
print("-" * 70)
print(f"Game over after {moves} moves. Score: {score:g}. Max tile: {final_state.max_tile()}")
