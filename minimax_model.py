"""Minimax Othello player with alpha-beta pruning."""

from __future__ import annotations

import random
from typing import Sequence

from othello_engine import (
    BOARD_SIZE,
    EMPTY,
    HeadlessOthello, #Game
    LegalMove,
    opponent,
)


DEFAULT_MINIMAX_DEPTH = 2  # Kept low enough for large benchmark batches.

# Corners are highly valuable. Squares beside an unclaimed corner are risky.
POSITION_WEIGHTS = (
    (120, -25, 20, 5, 5, 20, -25, 120),
    (-25, -45, -5, -5, -5, -5, -45, -25),
    (20, -5, 15, 3, 3, 15, -5, 20),
    (5, -5, 3, 3, 3, 3, -5, 5),
    (5, -5, 3, 3, 3, 3, -5, 5),
    (20, -5, 15, 3, 3, 15, -5, 20),
    (-25, -45, -5, -5, -5, -5, -45, -25),
    (120, -25, 20, 5, 5, 20, -25, 120),
)


class MinimaxPlayer:
    """Depth-limited minimax player with alpha-beta pruning."""

    WIN_SCORE = 1_000_000

    def __init__(self, depth: int = DEFAULT_MINIMAX_DEPTH) -> None:
        if depth < 1:
            raise ValueError("Minimax depth must be at least 1")
        self.depth = depth
        self.name = f"Minimax (depth {depth}, alpha-beta)"

    @staticmethod
    def _ordered_moves(legal_moves: Sequence[LegalMove]) -> list[LegalMove]:
        return sorted(
            legal_moves,
            key=lambda move: (
                POSITION_WEIGHTS[move.y][move.x],
                len(move.flips),
                -move.y,
                -move.x,
            ),
            reverse=True,
        )

    @classmethod
    def _terminal_value(cls, game: HeadlessOthello, root_color: int) -> int:
        scores = game.score()
        difference = scores[root_color] - scores[opponent(root_color)]
        if difference > 0:
            return cls.WIN_SCORE + difference
        if difference < 0:
            return -cls.WIN_SCORE + difference
        return 0

    @staticmethod
    def _heuristic_value(game: HeadlessOthello, root_color: int) -> int:
        other = opponent(root_color)
        root_discs = 0
        other_discs = 0
        positional = 0

        for y, row in enumerate(game.board):
            for x, square in enumerate(row):
                if square == root_color:
                    root_discs += 1
                    positional += POSITION_WEIGHTS[y][x]
                elif square == other:
                    other_discs += 1
                    positional -= POSITION_WEIGHTS[y][x]

        empty = BOARD_SIZE * BOARD_SIZE - root_discs - other_discs
        disc_difference = root_discs - other_discs
        mobility_difference = len(game.legal_moves(root_color)) - len(
            game.legal_moves(other)
        )

        if empty > 20:
            disc_weight = 1
        elif empty > 10:
            disc_weight = 4
        else:
            disc_weight = 12

        return positional + 10 * mobility_difference + disc_weight * disc_difference

    def _alpha_beta(
        self,
        game: HeadlessOthello,
        color: int,
        depth: int,
        alpha: float,
        beta: float,
        root_color: int,
    ) -> int:
        legal_moves = game.legal_moves(color)

        if not legal_moves:
            if not game.legal_moves(opponent(color)):
                return self._terminal_value(game, root_color)
            if depth <= 0:
                return self._heuristic_value(game, root_color)
            return self._alpha_beta(
                game,
                opponent(color),
                depth,
                alpha,
                beta,
                root_color,
            )

        if depth <= 0:
            return self._heuristic_value(game, root_color)

        ordered_moves = self._ordered_moves(legal_moves)
        if color == root_color:
            value = -self.WIN_SCORE * 2
            for move in ordered_moves:
                game.play(color, move)
                try:
                    child_value = self._alpha_beta(
                        game,
                        opponent(color),
                        depth - 1,
                        alpha,
                        beta,
                        root_color,
                    )
                finally:
                    game.undo(color, move)
                value = max(value, child_value)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = self.WIN_SCORE * 2
        for move in ordered_moves:
            game.play(color, move)
            try:
                child_value = self._alpha_beta(
                    game,
                    opponent(color),
                    depth - 1,
                    alpha,
                    beta,
                    root_color,
                )
            finally:
                game.undo(color, move)
            value = min(value, child_value)
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def choose_move(
        self,
        game: HeadlessOthello,
        color: int,
        legal_moves: Sequence[LegalMove],
        rng: random.Random,
    ) -> tuple[int, int]:
        del rng

        game = game.copy()
        
        ordered_moves = self._ordered_moves(legal_moves)
        best_move = ordered_moves[0]
        best_value = -self.WIN_SCORE * 2
        alpha = float("-inf")

        for move in ordered_moves:
            game.play(color, move)
            try:
                value = self._alpha_beta(
                    game,
                    opponent(color),
                    self.depth - 1,
                    alpha,
                    float("inf"),
                    color,
                )
            finally:
                game.undo(color, move)

            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)

        return best_move.x, best_move.y
