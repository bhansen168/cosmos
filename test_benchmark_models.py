"""Regression tests for the headless COSMOS model benchmark."""

from __future__ import annotations

import random
import sys
import tempfile
import types
import unittest
from pathlib import Path


# game.py imports Pygame for rendering, but its rules engine does not need it.
# A stub lets this test compare both rules engines in headless environments.
sys.modules.setdefault("pygame", types.ModuleType("pygame"))

from benchmark_models import (  # noqa: E402
    BLACK,
    WHITE,
    GreedyPlayer,
    HeadlessOthello,
    MinimaxPlayer,
    RandomPlayer,
    build_player,
    discover_models,
    opponent,
    run_match,
)
from genetic_model import GeneticPlayer, TrainingConfig, load_checkpoint, train  # noqa: E402
from game import Game  # noqa: E402
from minimax_model import MinimaxPlayer as StandaloneMinimaxPlayer  # noqa: E402
from watch_models import SpectatorMatch  # noqa: E402


class HeadlessRulesTests(unittest.TestCase):
    def test_rules_match_existing_game_across_complete_games(self) -> None:
        for seed in range(10):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                benchmark_game = HeadlessOthello()
                existing_game = Game(8)
                color = BLACK
                consecutive_passes = 0

                while consecutive_passes < 2:
                    benchmark_moves = benchmark_game.legal_moves(color)
                    existing_moves = existing_game.get_all_legal_moves(color)
                    benchmark_coordinates = [(move.x, move.y) for move in benchmark_moves]

                    self.assertEqual(benchmark_coordinates, existing_moves)
                    self.assertEqual(benchmark_game.board, existing_game.board)

                    if not benchmark_moves:
                        consecutive_passes += 1
                    else:
                        consecutive_passes = 0
                        move = rng.choice(benchmark_moves)
                        benchmark_game.play(color, move)
                        self.assertTrue(existing_game.place_piece(color, move.x, move.y))

                    color = opponent(color)

                self.assertEqual(benchmark_game.board, existing_game.board)
                self.assertEqual(benchmark_game.score(), existing_game.get_score())

    def test_starting_position_has_four_legal_moves_for_each_color(self) -> None:
        game = HeadlessOthello()
        self.assertEqual(len(game.legal_moves(BLACK)), 4)
        self.assertEqual(len(game.legal_moves(WHITE)), 4)


class MatchTests(unittest.TestCase):
    def test_match_totals_and_color_alternation(self) -> None:
        stats, _ = run_match(
            (RandomPlayer(), GreedyPlayer()),
            games=5,
            seed=123,
            show_progress=False,
        )

        self.assertEqual(stats.games, 5)
        self.assertEqual(stats.games_as_black, [3, 2])
        self.assertEqual(sum(stats.wins) + stats.draws, 5)
        self.assertGreater(sum(stats.total_discs), 0)

    def test_checkpoint_discovery_includes_builtin_and_dqn_models(self) -> None:
        options = discover_models()
        specs = [option.spec for option in options]
        self.assertEqual(
            specs[:6],
            ["random", "greedy", "minimax:1", "minimax", "minimax:3", "minimax:4"],
        )
        self.assertTrue(any(spec.lower().endswith(".pth") for spec in specs[6:]))

    def test_minimax_selects_a_legal_move_without_changing_the_board(self) -> None:
        game = HeadlessOthello()
        original_board = [row.copy() for row in game.board]
        legal_moves = game.legal_moves(BLACK)
        legal_coordinates = {(move.x, move.y) for move in legal_moves}

        selected = MinimaxPlayer(depth=3).choose_move(
            game,
            BLACK,
            legal_moves,
            random.Random(0),
        )

        self.assertIn(selected, legal_coordinates)
        self.assertEqual(game.board, original_board)

    def test_custom_minimax_depth_spec(self) -> None:
        player = build_player("minimax:2")
        self.assertIsInstance(player, MinimaxPlayer)
        self.assertIsInstance(player, StandaloneMinimaxPlayer)
        self.assertEqual(player.__class__.__module__, "minimax_model")
        self.assertEqual(player.depth, 2)

        with self.assertRaisesRegex(ValueError, "at least 1"):
            build_player("minimax:0")

    def test_genetic_training_checkpoint_and_benchmark_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = train(
                TrainingConfig(
                    generations=1,
                    population_size=4,
                    games_per_pair=1,
                    baseline_games=1,
                    minimax_games=0,
                    elite_count=1,
                    tournament_size=2,
                    checkpoint_every=1,
                    output_directory=Path(temporary_directory),
                    seed=9,
                )
            )

            payload = load_checkpoint(checkpoint)
            self.assertEqual(payload["generation"], 0)
            self.assertEqual(len(payload["population"]), 4)

            player = build_player(f"genetic:{checkpoint}")
            self.assertIsInstance(player, GeneticPlayer)
            game = HeadlessOthello()
            original_board = [row.copy() for row in game.board]
            legal_moves = game.legal_moves(BLACK)
            selected = player.choose_move(
                game,
                BLACK,
                legal_moves,
                random.Random(0),
            )
            self.assertIn(selected, {(move.x, move.y) for move in legal_moves})
            self.assertEqual(game.board, original_board)

            benchmark_stats, _ = run_match(
                (player, GreedyPlayer()),
                games=2,
                seed=4,
                show_progress=False,
            )
            self.assertEqual(benchmark_stats.games, 2)
            self.assertEqual(sum(benchmark_stats.wins) + benchmark_stats.draws, 2)

            resumed_checkpoint = train(
                TrainingConfig(
                    generations=2,
                    population_size=4,
                    games_per_pair=1,
                    baseline_games=1,
                    minimax_games=0,
                    elite_count=1,
                    tournament_size=2,
                    checkpoint_every=1,
                    output_directory=Path(temporary_directory),
                    seed=9,
                    resume=checkpoint,
                )
            )
            self.assertEqual(load_checkpoint(resumed_checkpoint)["generation"], 1)


class SpectatorTests(unittest.TestCase):
    def test_spectator_state_can_step_through_a_complete_game(self) -> None:
        match = SpectatorMatch()
        rng = random.Random(17)

        while not match.game_over:
            legal_moves = match.legal_moves()
            if not legal_moves:
                match.pass_turn()
                continue
            selected = rng.choice(legal_moves)
            match.apply_move((selected.x, selected.y))

        scores = match.scores()
        self.assertGreater(len(match.history), 0)
        self.assertGreater(scores[BLACK] + scores[WHITE], 4)
        self.assertFalse(match.game.legal_moves(BLACK))
        self.assertFalse(match.game.legal_moves(WHITE))
        if scores[BLACK] == scores[WHITE]:
            self.assertIsNone(match.winner)
        else:
            expected = BLACK if scores[BLACK] > scores[WHITE] else WHITE
            self.assertEqual(match.winner, expected)


if __name__ == "__main__":
    unittest.main()
