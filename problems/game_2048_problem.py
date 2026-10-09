"""
Simplified 2048 game as a stochastic problem (a Markov decision process).
"""

import random
from typing import Dict, List, Tuple
from core.stochastic_problem import StochasticProblem
from problems.game_2048_state import Game2048State


class Game2048Problem(StochasticProblem):
    """
    2048 on a small board (2x2 by default).

    Rules:
      - Four actions: Left, Right, Up, Down. All tiles slide as far as possible.
      - Two equal tiles that collide merge into one (2 + 2 = 4). A tile merges at
        most once per move. The reward of a move is the sum of the merged tiles.
      - Only moves that change the board are legal. With no legal move, the game ends.
      - After every move a new tile appears on a random empty cell: a 2 with
        probability prob_two, a 4 otherwise.
      - The game starts with two random tiles.

    The deterministic part of a move (slide and merge) is apply_action; the random
    part (the new tile) is described by get_chance_outcomes.
    """

    ACTIONS = ("Left", "Right", "Up", "Down")

    def __init__(self, rows: int = 2, cols: int = 2, prob_two: float = 0.9):
        """
        Initialize the game.

        Args:
            rows: Number of rows of the board
            cols: Number of columns of the board
            prob_two: Probability that a new tile is a 2 (otherwise it is a 4)
        """
        assert rows >= 1 and cols >= 1 and rows * cols >= 2, "The board needs at least two cells"
        self.rows = rows
        self.cols = cols
        self.prob_two = prob_two
        self._lines = self._build_lines()
        self._actions_cache: Dict[Game2048State, List[str]] = {}

    def _build_lines(self) -> Dict[str, List[List[int]]]:
        """
        For each action, the cell indices of every line of the board, starting with
        the cell on the border the tiles slide towards.
        """
        rows, cols = self.rows, self.cols
        by_row = [[r * cols + c for c in range(cols)] for r in range(rows)]
        by_col = [[r * cols + c for r in range(rows)] for c in range(cols)]
        return {
            "Left": by_row,
            "Right": [list(reversed(line)) for line in by_row],
            "Up": by_col,
            "Down": [list(reversed(line)) for line in by_col],
        }

    @staticmethod
    def slide_line(line: List[int]) -> Tuple[List[int], int]:
        """
        Slide and merge one line towards its first element.

        Args:
            line: Exponents of the line, starting with the destination border

        Returns:
            (new line, points obtained by the merges)
        """
        tiles = [e for e in line if e]
        result, points, i = [], 0, 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                result.append(tiles[i] + 1)          # 2^k + 2^k = 2^(k+1)
                points += 2 ** (tiles[i] + 1)
                i += 2                               # each tile merges at most once
            else:
                result.append(tiles[i])
                i += 1
        return result + [0] * (len(line) - len(result)), points

    def get_initial_state(self, rng: random.Random) -> Game2048State:
        """Empty board plus two random tiles."""
        empty = Game2048State(tuple([0] * (self.rows * self.cols)), self.rows, self.cols)
        return self.sample_chance_outcome(self.sample_chance_outcome(empty, rng), rng)

    def apply_action(self, state: Game2048State, action: str) -> Tuple[Game2048State, float]:
        """Slide and merge all lines. Returns (board before the new tile, points)."""
        cells = list(state.cells)
        points = 0
        for indices in self._lines[action]:
            new_line, p = self.slide_line([state.cells[i] for i in indices])
            points += p
            for i, e in zip(indices, new_line):
                cells[i] = e
        return Game2048State(tuple(cells), self.rows, self.cols), float(points)

    def get_actions(self, state: Game2048State) -> List[str]:
        """Actions that change the board."""
        actions = self._actions_cache.get(state)
        if actions is None:
            actions = [a for a in self.ACTIONS if self.apply_action(state, a)[0] != state]
            if len(self._actions_cache) > 1_000_000:
                self._actions_cache.clear()
            self._actions_cache[state] = actions
        return list(actions)

    def get_chance_outcomes(self, afterstate: Game2048State) -> List[Tuple[Game2048State, float]]:
        """Every board that can result from placing the new tile, with its probability."""
        empty = afterstate.empty_cells()
        outcomes = []
        for i in empty:
            for exponent, probability in ((1, self.prob_two), (2, 1 - self.prob_two)):
                cells = list(afterstate.cells)
                cells[i] = exponent
                outcomes.append((Game2048State(tuple(cells), self.rows, self.cols), probability / len(empty)))
        return outcomes

    def sample_chance_outcome(self, afterstate: Game2048State, rng: random.Random) -> Game2048State:
        """Place a 2 or a 4 on a random empty cell."""
        i = rng.choice(afterstate.empty_cells())
        cells = list(afterstate.cells)
        cells[i] = 1 if rng.random() < self.prob_two else 2
        return Game2048State(tuple(cells), self.rows, self.cols)

    def heuristic(self, state: Game2048State) -> float:
        """
        Rough estimate of the points still to be obtained from a non-terminal board:
        the sum of its tiles. What matters most is that it is positive, so that a
        search that runs out of depth prefers boards that are still alive to boards
        where the game is over (which are worth 0).
        """
        return float(state.tile_sum())
