"""Headless Othello rules shared by training and benchmark programs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, Sequence


EMPTY = 0
BLACK = 1
WHITE = 2
BOARD_SIZE = 8
DIRECTIONS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)


def opponent(color: int) -> int:
    return WHITE if color == BLACK else BLACK


@dataclass(frozen=True)
class LegalMove:
    x: int
    y: int
    flips: tuple[tuple[int, int], ...]


class HeadlessOthello:
    """Minimal 8x8 rules engine with no GUI dependency."""

    def __init__(self) -> None:
        self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.board[3][3] = WHITE
        self.board[4][4] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK

    @staticmethod
    def _inside(x: int, y: int) -> bool:
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def _flips_for(self, color: int, x: int, y: int) -> tuple[tuple[int, int], ...]:
        if not self._inside(x, y) or self.board[y][x] != EMPTY:
            return ()

        flips: list[tuple[int, int]] = []
        other = opponent(color)

        for dx, dy in DIRECTIONS:
            line: list[tuple[int, int]] = []
            scan_x, scan_y = x + dx, y + dy

            while self._inside(scan_x, scan_y) and self.board[scan_y][scan_x] == other:
                line.append((scan_x, scan_y))
                scan_x += dx
                scan_y += dy

            if (
                line
                and self._inside(scan_x, scan_y)
                and self.board[scan_y][scan_x] == color
            ):
                flips.extend(line)

        return tuple(flips)

    def legal_moves(self, color: int) -> list[LegalMove]:
        moves: list[LegalMove] = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                flips = self._flips_for(color, x, y)
                if flips:
                    moves.append(LegalMove(x=x, y=y, flips=flips))
        return moves

    def play(self, color: int, move: LegalMove) -> None:
        if self.board[move.y][move.x] != EMPTY or not move.flips:
            raise ValueError(f"Illegal move ({move.x}, {move.y})")

        self.board[move.y][move.x] = color
        for x, y in move.flips:
            self.board[y][x] = color

    def undo(self, color: int, move: LegalMove) -> None:
        """Undo a move previously applied with play()."""
        self.board[move.y][move.x] = EMPTY
        other = opponent(color)
        for x, y in move.flips:
            self.board[y][x] = other

    def score(self) -> dict[int, int]:
        return {
            BLACK: sum(row.count(BLACK) for row in self.board),
            WHITE: sum(row.count(WHITE) for row in self.board),
        }

    def clone(self) -> HeadlessOthello:
        """Return an independent copy suitable for search on another thread."""
        copied = HeadlessOthello()
        copied.board = [row.copy() for row in self.board]
        return copied


class Player(Protocol):
    name: str

    def choose_move(
        self,
        game: HeadlessOthello,
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
) -> GameOutcome:
    """Play one complete game, including all forced passes."""
    game = HeadlessOthello()
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

    scores = game.score()
    if scores[BLACK] == scores[WHITE]:
        winner = None
    else:
        winner = BLACK if scores[BLACK] > scores[WHITE] else WHITE
    return GameOutcome(
        black_score=scores[BLACK],
        white_score=scores[WHITE],
        winner=winner,
        moves=moves_played,
    )
