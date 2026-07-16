"""Compatibility layer exposing the original game.py engine to newer tools."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, Sequence

from game import Game, LegalMove


EMPTY = Game.EMPTY
BLACK = Game.BLACK
WHITE = Game.WHITE
BOARD_SIZE = 8
DIRECTIONS = tuple(Game.DIRS)

# Newer modules historically imported this name. It now points directly to the
# original Game class so there is only one rules implementation.
HeadlessOthello = Game


def opponent(color: int) -> int:
    return WHITE if color == BLACK else BLACK


class Player(Protocol):
    name: str

    def choose_move(
        self,
        game: Game,
        color: int,
        legal_moves: Sequence[LegalMove],
        rng: random.Random,
    ) -> tuple[int, int]: ...


@dataclass(frozen=True)
class GameOutcome:
    black_score: int
    white_score: int
    winner: int | None
    moves: int


def play_game(
    black_player: Player,
    white_player: Player,
    rng: random.Random,
    save: bool = False
) -> GameOutcome:
    """Play one complete game with the original Game rules implementation."""

    game = Game(save = save)
    players = {BLACK: black_player, WHITE: white_player}
    color = BLACK
    consecutive_passes = 0
    moves_played = 0

    while consecutive_passes < 2:
        legal_moves = game.legal_moves(color)
        if not legal_moves:
            consecutive_passes += 1
            color = opponent(color)
            continue

        consecutive_passes = 0
        selected = players[color].choose_move(game, color, legal_moves, rng)
        legal_by_coordinate = {(move.x, move.y): move for move in legal_moves}
        if selected not in legal_by_coordinate:
            raise RuntimeError(
                f"{players[color].name} selected illegal move {selected}"
            )

        game.play(color, legal_by_coordinate[selected])
        moves_played += 1
        color = opponent(color)

    scores = game.get_score()
    if scores[BLACK] == scores[WHITE]:
        winner = None
    else:
        winner = BLACK if scores[BLACK] > scores[WHITE] else WHITE

    if game.save:
        game.save_game()
        
    return GameOutcome(
        black_score=scores[BLACK],
        white_score=scores[WHITE],
        winner=winner,
        moves=moves_played,
    )
