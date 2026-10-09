"""
Compare several agents on many games of (simplified) 2048.
Agents whose code is not implemented yet are skipped.
"""

import time
from problems import Game2048Problem
from algorithms import RandomAgent, MinimaxAgent, QLearningAgent
from core import evaluate_agent

# Configuration
ROWS, COLS = 2, 2
GAMES = 1000
DEPTHS = [1, 2, 3]          # minimax is evaluated with each of these depths
TRAINING_EPISODES = 30_000

problem = Game2048Problem(ROWS, COLS)


def build_q_learning_agent():
    agent = QLearningAgent(problem)
    agent.train(TRAINING_EPISODES)
    return agent


candidates = [("Random", lambda: RandomAgent(problem, seed=0))]
for depth in DEPTHS:
    candidates.append((f"Minimax (depth {depth})", lambda depth=depth: MinimaxAgent(problem, depth=depth)))
candidates.append((f"Q-learning ({TRAINING_EPISODES} episodes)", build_q_learning_agent))

print("=" * 70)
print(f"2048 ({ROWS}x{COLS}): {GAMES} games per agent")
print("=" * 70)
for name, build in candidates:
    start = time.time()
    try:
        agent = build()
        result = evaluate_agent(problem, agent, games=GAMES)
    except NotImplementedError as e:
        print(f"{name:32} not implemented yet ({e})")
        continue
    elapsed = time.time() - start
    tiles = ", ".join(f"{tile}: {100 * n / GAMES:.1f}%" for tile, n in result["max_tiles"].items())
    print(f"{name:32} mean score {result['mean_score']:7.2f}   ({elapsed:5.1f} s)")
    print(f"{'':32} max tile   {tiles}")
    if hasattr(agent, "expanded_nodes"):
        total_moves = result["mean_moves"] * result["games"]
        print(f"{'':32} expanded nodes per move {agent.expanded_nodes / total_moves:.1f}")
