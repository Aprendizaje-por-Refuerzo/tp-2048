"""
Board state for the 2048 game.
"""

from typing import List, Tuple
from core.game_state import GameState


class Game2048State(GameState):
    """
    Board of a (possibly small) 2048 game.

    Cells are stored row by row as exponents: 0 = empty cell, k = tile 2**k.
    For example, on a 2x2 board the tuple (1, 1, 0, 2) represents

        | 2 2 |
        | . 4 |

    Using exponents, merging two equal tiles is just adding 1 to the exponent.
    States are immutable and hashable, so they can be used as dictionary keys.
    """

    def __init__(self, cells: Tuple[int, ...], rows: int, cols: int):
        """
        Initialize a board.

        Args:
            cells: Exponents of the tiles, row by row (0 = empty)
            rows: Number of rows of the board
            cols: Number of columns of the board
        """
        assert len(cells) == rows * cols, "The number of cells must be rows * cols"
        self.cells = tuple(cells)
        self.rows = rows
        self.cols = cols
        self._hash = hash((self.rows, self.cols, self.cells))

    def __hash__(self) -> int:
        """Hash based on the cells."""
        return self._hash

    def __eq__(self, other: object) -> bool:
        """Equality based on the cells."""
        if not isinstance(other, Game2048State):
            return False
        return self.cells == other.cells and self.rows == other.rows and self.cols == other.cols

    def __str__(self) -> str:
        width = max(4, len(str(self.max_tile())))
        lines = []
        for row in range(self.rows):
            values = [self.tile(row, col) for col in range(self.cols)]
            lines.append("|" + " ".join(f"{v if v else '.':>{width}}" for v in values) + " |")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Game2048State({self.cells}, {self.rows}, {self.cols})"

    def tile(self, row: int, col: int) -> int:
        """Value of the tile at (row, col), or 0 if the cell is empty."""
        exponent = self.cells[row * self.cols + col]
        return 2 ** exponent if exponent else 0

    def max_tile(self) -> int:
        """Value of the largest tile on the board."""
        return 2 ** max(self.cells) if any(self.cells) else 0

    def tile_sum(self) -> int:
        """Sum of the values of all tiles on the board."""
        return sum(2 ** e for e in self.cells if e)

    def empty_cells(self) -> List[int]:
        """Indices (row * cols + col) of the empty cells."""
        return [i for i, e in enumerate(self.cells) if e == 0]
