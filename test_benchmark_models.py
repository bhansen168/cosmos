"""Regression tests for the headless COSMOS model benchmark."""

from __future__ import annotations

import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


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
from computer import Computer, RandomComputer, create_minimax_computer  # noqa: E402
from computer2 import Computer3  # noqa: E402
from genetic_model import (  # noqa: E402
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    GENOME_SIZE,
    LEGACY_GENOME_SIZE,
    GeneticPlayer,
    TrainingConfig,
    load_checkpoint,
    train,
)
from game import Game  # noqa: E402
from minimax_model import MinimaxPlayer as StandaloneMinimaxPlayer  # noqa: E402
from watch_models import SpectatorMatch  # noqa: E402


class HeadlessRulesTests(unittest.TestCase):
    def test_new_engine_name_uses_the_original_game_class(self) -> None:
        self.assertIs(HeadlessOthello, Game)

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

    def test_original_computers_supply_benchmark_player_interfaces(self) -> None:
        self.assertIs(GreedyPlayer, Computer)
        self.assertIs(RandomPlayer, RandomComputer)
        game = Game()
        greedy_move = GreedyPlayer().choose_move(
            game,
            BLACK,
            game.legal_moves(BLACK),
            random.Random(1),
        )
        random_move = RandomPlayer().choose_move(
            game,
            BLACK,
            game.legal_moves(BLACK),
            random.Random(1),
        )
        legal = set(game.get_all_legal_moves(BLACK))
        self.assertIn(greedy_move, legal)
        self.assertIn(random_move, legal)

    def test_bound_computer_places_the_requested_color(self) -> None:
        game = Game()
        computer = Computer(game, WHITE)
        computer.pick_greedy(color=BLACK, place=True)
        self.assertEqual(game.get_score(), {BLACK: 4, WHITE: 1})

        minimax = create_minimax_computer(game, WHITE, depth=1)
        original_board = [row.copy() for row in game.board]
        selected = minimax.pick_minimax(color=WHITE, place=False)
        self.assertIn(selected, game.get_all_legal_moves(WHITE))
        self.assertEqual(game.board, original_board)


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
        self.assertIn("bard", specs)
        self.assertTrue(any(spec.startswith("bard:") for spec in specs))
        self.assertTrue(any(spec.lower().endswith(".pth") for spec in specs[6:]))

    def test_bard_adapter_selects_a_legal_move(self) -> None:
        class FakeBardAgent:
            @staticmethod
            def pick(legal_moves, board):
                self.assertEqual(board.shape, (64,))
                y, x = divmod(legal_moves[0], 8)
                return x, y

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "test.bard"
            checkpoint.write_bytes(b"test checkpoint")
            with mock.patch(
                "computer2.load_agent_sup",
                return_value=FakeBardAgent(),
            ):
                player = build_player(f"bard:{checkpoint}")

            self.assertIsInstance(player, Computer3)
            game = Game()
            legal_moves = game.legal_moves(BLACK)
            selected = player.choose_move(
                game,
                BLACK,
                legal_moves,
                random.Random(0),
            )
            self.assertIn(selected, {(move.x, move.y) for move in legal_moves})

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
                    coevolution_opponents=1,
                    baseline_games=0,
                    minimax_games=0,
                    search_depth=1,
                    opening_plies=2,
                    validation_candidates=1,
                    validation_openings=1,
                    elite_count=1,
                    tournament_size=2,
                    random_immigrants=1,
                    checkpoint_every=1,
                    output_directory=Path(temporary_directory),
                    seed=9,
                )
            )

            payload = load_checkpoint(checkpoint)
            self.assertEqual(payload["version"], CHECKPOINT_VERSION)
            self.assertEqual(payload["generation"], 0)
            self.assertEqual(len(payload["population"]), 4)
            self.assertIn("champion", payload)
            self.assertEqual(len(payload["champion"]["genome"]), GENOME_SIZE)

            player = build_player(f"genetic:{checkpoint}")
            self.assertIsInstance(player, GeneticPlayer)
            self.assertEqual(player.genome, tuple(payload["champion"]["genome"]))
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
                    coevolution_opponents=1,
                    baseline_games=0,
                    minimax_games=0,
                    search_depth=1,
                    opening_plies=2,
                    validation_candidates=1,
                    validation_openings=1,
                    elite_count=1,
                    tournament_size=2,
                    random_immigrants=1,
                    checkpoint_every=1,
                    output_directory=Path(temporary_directory),
                    seed=9,
                    resume=checkpoint,
                )
            )
            self.assertEqual(load_checkpoint(resumed_checkpoint)["generation"], 1)

    def test_legacy_genetic_checkpoint_upgrades_and_uses_current_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy.json"
            stale_best = [0.75 for _ in range(LEGACY_GENOME_SIZE)]
            current_best = [0.25 for _ in range(LEGACY_GENOME_SIZE)]
            payload = {
                "format": CHECKPOINT_FORMAT,
                "version": 1,
                "generation": 34,
                "generation_best": {
                    "genome": current_best,
                    "fitness": 0.7,
                    "games": 44,
                },
                "best_ever": {
                    "genome": stale_best,
                    "fitness": 0.95,
                    "games": 44,
                },
                "population": [
                    {"genome": current_best, "fitness": 0.7, "games": 44},
                    {"genome": stale_best, "fitness": 0.6, "games": 44},
                ],
                "config": {"minimax_depth": 1},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            upgraded = load_checkpoint(path)
            player = GeneticPlayer.from_checkpoint(path)

            self.assertEqual(upgraded["source_version"], 1)
            self.assertEqual(upgraded["version"], CHECKPOINT_VERSION)
            self.assertEqual(len(upgraded["champion"]["genome"]), GENOME_SIZE)
            self.assertEqual(
                player.genome,
                tuple(upgraded["champion"]["genome"]),
            )
            self.assertNotEqual(
                upgraded["champion"]["genome"],
                upgraded["best_ever"]["genome"],
            )


class SpectatorTests(unittest.TestCase):
    def test_spectator_state_can_step_through_a_complete_game(self) -> None:
        match = SpectatorMatch()
        self.assertIsInstance(match.game, Game)
        rng = random.Random(17)

        while not match.game_over:
            legal_moves = match.legal_moves()
            if not legal_moves:
                match.pass_turn()
                continue
            selected = rng.choice(legal_moves)
            match.apply_move((selected.x, selected.y))

        scores = match.scores()
        final_board = [row.copy() for row in match.game.board]
        final_turn = len(match.history)
        self.assertGreater(len(match.history), 0)
        self.assertTrue(match.at_latest)
        self.assertEqual(match.timeline_index, final_turn)
        self.assertEqual(len(match.timeline), final_turn + 1)
        self.assertGreater(scores[BLACK] + scores[WHITE], 4)
        self.assertFalse(match.game.legal_moves(BLACK))
        self.assertFalse(match.game.legal_moves(WHITE))
        if scores[BLACK] == scores[WHITE]:
            self.assertIsNone(match.winner)
        else:
            expected = BLACK if scores[BLACK] > scores[WHITE] else WHITE
            self.assertEqual(match.winner, expected)

        self.assertTrue(match.seek(0))
        self.assertFalse(match.game_over)
        self.assertEqual(match.game.board, Game().board)
        self.assertEqual(match.visible_history, [])

        self.assertTrue(match.seek_relative(1))
        self.assertEqual(len(match.visible_history), 1)
        self.assertEqual(match.last_move, match.history[0].coordinate)

        self.assertTrue(match.seek(final_turn))
        self.assertTrue(match.at_latest)
        self.assertTrue(match.game_over)
        self.assertEqual(match.game.board, final_board)
        self.assertEqual(match.scores(), scores)

    def test_spectator_can_branch_after_rewinding(self) -> None:
        match = SpectatorMatch()
        first = match.legal_moves()[0]
        match.apply_move((first.x, first.y))
        original_second = match.legal_moves()[0]
        match.apply_move((original_second.x, original_second.y))

        self.assertEqual(len(match.history), 2)
        self.assertTrue(match.seek(1))
        alternatives = [
            move
            for move in match.legal_moves()
            if (move.x, move.y) != (original_second.x, original_second.y)
        ]
        self.assertTrue(alternatives)
        replacement = alternatives[-1]
        match.apply_move((replacement.x, replacement.y))

        self.assertTrue(match.at_latest)
        self.assertEqual(len(match.history), 2)
        self.assertEqual(len(match.timeline), 3)
        self.assertEqual(match.history[-1].coordinate, (replacement.x, replacement.y))


if __name__ == "__main__":
    unittest.main()
