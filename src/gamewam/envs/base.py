"""Base game environment interface for GameWAM deployment."""

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class GameObservation:
    """Observation returned by every game environment."""
    image: Optional[np.ndarray] = None
    proprio: Optional[np.ndarray] = None
    proprio_field_valid: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    timestamp: int = 0


class GameEnvBase(abc.ABC):
    """Abstract base for all game environments used with GameWAM."""

    @abc.abstractmethod
    def reset(self, **kwargs) -> GameObservation:
        """Reset environment and return initial observation."""
        ...

    @abc.abstractmethod
    def step(self, action: Any) -> GameObservation:
        """Execute action and return resulting observation."""
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Release environment resources."""
        ...

    @abc.abstractmethod
    def get_screenshot(self) -> np.ndarray:
        """Capture the current game frame as [H, W, 3] RGB array."""
        ...

    @property
    @abc.abstractmethod
    def game_type(self) -> str:
        """Return the game type identifier (e.g. 'minecraft', 'vizdoom')."""
        ...

    @property
    def proprio_dim(self) -> int:
        """Dimension of proprioceptive state. 0 if not available."""
        return 0
