"""
Train a Q-learning agent on (simplified) 2048 and print its learning curve:
after some numbers of training episodes, the greedy policy is evaluated.
"""

import time
from problems import Game2048Problem
from algorithms import QLearningAgent
from core import evaluate_agent

# Configuration
ROWS, COLS = 2, 2
ALPHA, GAMMA, EPSILON = 0.1, 1.0, 0.1
SEED = 42
CHECKPOINTS = [100, 300, 1_000, 3_000, 10_000, 30_000]
EVALUATION_GAMES = 2000

problem = Game2048Problem(ROWS, COLS)
agent = QLearningAgent(problem, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, seed=SEED)

print("=" * 70)
print(f"Q-learning on 2048 ({ROWS}x{COLS}): alpha={ALPHA}, gamma={GAMMA}, epsilon={EPSILON}")
print("=" * 70)
print(f"{'episodes':>10} {'mean score':>12} {'Q entries':>10} {'time (s)':>9}")
start = time.time()
trained = 0
for checkpoint in CHECKPOINTS:
    agent.train(checkpoint - trained)
    trained = checkpoint
    result = evaluate_agent(problem, agent, games=EVALUATION_GAMES)
    print(f"{trained:>10} {result['mean_score']:>12.2f} {len(agent.q):>10} {time.time() - start:>9.1f}")
