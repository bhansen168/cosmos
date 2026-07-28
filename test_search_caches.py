"""Regression tests for semantics-preserving search transposition caches."""

from __future__ import annotations

import random
import unittest

from genetic_model import DEFAULT_SEED_GENOME, GeneticPlayer
from minimax_model import MinimaxPlayer
from othello_engine import BLACK, EMPTY, HeadlessOthello, opponent


def _random_position(seed: int, target_empty: int) -> tuple[HeadlessOthello, int]:
    rng = random.Random(seed)
    game = HeadlessOthello()
    color = BLACK
    passes = 0
    while (
        sum(square == EMPTY for row in game.board for square in row)
        > target_empty
        and passes < 2
    ):
        legal_moves = game.legal_moves(color)
        if not legal_moves:
            passes += 1
            color = opponent(color)
            continue
        passes = 0
        game.play(color, rng.choice(legal_moves))
        color = opponent(color)
    if not game.legal_moves(color):
        color = opponent(color)
    return game, color


def _genetic_root_values(
    player: GeneticPlayer,
    game: HeadlessOthello,
    color: int,
    use_cache: bool,
) -> list[float]:
    legal_moves = game.legal_moves(color)
    remaining_depth = (
        sum(square == EMPTY for row in game.board for square in row) - 1
    )
    transposition = {} if use_cache else None
    values: list[float] = []
    alpha = float("-inf")
    working = game.copy()
    for move in player._ordered_moves(legal_moves):
        working.play(color, move)
        try:
            value = player._alpha_beta(
                working,
                opponent(color),
                remaining_depth,
                alpha,
                float("inf"),
                color,
                transposition,
            )
        finally:
            working.undo(color, move)
        values.append(value)
        alpha = max(alpha, value)
    return values


def _minimax_root_values(
    player: MinimaxPlayer,
    game: HeadlessOthello,
    color: int,
    use_cache: bool,
) -> list[int]:
    legal_moves = game.legal_moves(color)
    transposition = {} if use_cache else None
    values: list[int] = []
    alpha = float("-inf")
    working = game.copy()
    for move in player._ordered_moves(legal_moves):
        working.play(color, move)
        try:
            value = player._alpha_beta(
                working,
                opponent(color),
                player.depth - 1,
                alpha,
                float("inf"),
                color,
                transposition,
            )
        finally:
            working.undo(color, move)
        values.append(value)
        alpha = max(alpha, value)
    return values


class SearchCacheTests(unittest.TestCase):
    def test_genetic_exact_endgame_cache_preserves_root_values(self) -> None:
        game, color = _random_position(seed=37, target_empty=6)
        player = GeneticPlayer(
            DEFAULT_SEED_GENOME,
            search_depth=2,
            endgame_exact_empties=6,
        )

        self.assertEqual(
            _genetic_root_values(player, game, color, use_cache=True),
            _genetic_root_values(player, game, color, use_cache=False),
        )

    def test_minimax_cache_preserves_root_values(self) -> None:
        game, color = _random_position(seed=19, target_empty=44)
        player = MinimaxPlayer(depth=3)

        self.assertEqual(
            _minimax_root_values(player, game, color, use_cache=True),
            _minimax_root_values(player, game, color, use_cache=False),
        )


if __name__ == "__main__":
    unittest.main()
