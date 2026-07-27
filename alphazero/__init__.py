"""Independent, optimized AlphaZero implementation for COSMOS Othello."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .board import BLACK, EMPTY, WHITE, BitBoard

if TYPE_CHECKING:
    from .model import AlphaZeroModelConfig, AlphaZeroNetwork, AlphaZeroPlayer

_MODEL_EXPORTS = {
    "AlphaZeroModelConfig",
    "AlphaZeroNetwork",
    "AlphaZeroPlayer",
}


def __getattr__(name: str) -> Any:
    """Keep bitboards/search importable without eagerly importing PyTorch."""

    if name in _MODEL_EXPORTS:
        from . import model

        value = getattr(model, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BLACK",
    "EMPTY",
    "WHITE",
    "AlphaZeroModelConfig",
    "AlphaZeroNetwork",
    "AlphaZeroPlayer",
    "BitBoard",
]
