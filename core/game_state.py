from abc import ABC, abstractmethod


class GameState(ABC):
    """
    Abstract base class for any game state.
    States must be hashable: search agents use them as keys of memo tables and
    learning agents use them as keys of their Q tables.
    """

    @abstractmethod
    def __hash__(self) -> int:
        """Hash value of a state (states must be hashable for use in sets/dictionaries)"""
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Equality comparison for state checking"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Human-readable representation of the state"""
        pass
