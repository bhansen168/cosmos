"""Focused correctness tests for the independent AlphaZero subsystem."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from alphazero.board import (
    ACTION_DIM,
    ACTION_TRANSFORMS,
    BitBoard,
    action_to_coord,
    encode_boards,
    transform_action,
    transform_board,
)
from alphazero.mcts import ExactSolver, MCTSConfig, MCTSNode, NeuralMCTS
from alphazero.model import (
    AlphaZeroEvaluator,
    AlphaZeroModelConfig,
    AlphaZeroNetwork,
    AlphaZeroPlayer,
    load_checkpoint,
    save_checkpoint,
)
from alphazero.replay import ReplayBuffer, SelfPlayRecord
from alphazero.training import (
    TrainingConfig,
    collect_self_play,
    collect_self_play_parallel,
    evaluate_candidate,
    reanalyse_replay,
    train,
    update_network,
)
from benchmark_models import build_player
from computer import ComputerAlphaZero
from game import Game


class BitBoardTests(unittest.TestCase):
    def test_random_games_match_reference_engine(self) -> None:
        for seed in range(20):
            rng = random.Random(seed)
            game = Game()
            color = Game.BLACK
            for _ in range(60):
                board = BitBoard.from_game(game, color)
                reference_moves = {
                    move.y * 8 + move.x for move in game.legal_moves(color)
                }
                self.assertEqual(set(board.legal_actions()), reference_moves)
                if not reference_moves:
                    other = Game.WHITE if color == Game.BLACK else Game.BLACK
                    if not game.legal_moves(other):
                        break
                    color = other
                    continue

                action = rng.choice(tuple(reference_moves))
                x, y = action_to_coord(action)
                move = next(
                    move
                    for move in game.legal_moves(color)
                    if (move.x, move.y) == (x, y)
                )
                expected = board.play(action)
                game.play(color, move)
                color = Game.WHITE if color == Game.BLACK else Game.BLACK
                self.assertEqual(expected, BitBoard.from_game(game, color))

    def test_symmetries_preserve_legal_actions(self) -> None:
        board = BitBoard.initial()
        legal = set(board.legal_actions())
        for symmetry in range(8):
            with self.subTest(symmetry=symmetry):
                transformed = transform_board(board, symmetry)
                expected = {transform_action(action, symmetry) for action in legal}
                self.assertEqual(set(transformed.legal_actions()), expected)
                self.assertEqual(
                    sorted(ACTION_TRANSFORMS[symmetry]),
                    list(range(ACTION_DIM)),
                )

    def test_encoding_contains_phase_and_legal_planes(self) -> None:
        board = BitBoard.initial()
        encoded = encode_boards([board])
        self.assertEqual(encoded.shape, (1, 4, 8, 8))
        self.assertEqual(int(encoded[0, 0].sum()), 2)
        self.assertEqual(int(encoded[0, 1].sum()), 2)
        self.assertEqual(int(encoded[0, 2].sum()), 4)
        self.assertAlmostEqual(float(encoded[0, 3, 0, 0]), 1.0)

    def test_precomputed_flip_fast_path_rejects_an_incorrect_mask(self) -> None:
        board = BitBoard.initial()
        action = board.legal_actions()[0]
        flips = board.flips_for_action(action)
        self.assertEqual(board.play_with_flips(action, flips), board.play(action))
        unrelated_opponent = board.opponent & ~flips
        bogus = unrelated_opponent & -unrelated_opponent
        with self.assertRaises(ValueError):
            board.play_with_flips(action, bogus)


class SearchTests(unittest.TestCase):
    class UniformEvaluator:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def evaluate(self, boards):
            self.calls.append(len(boards))
            results = []
            for board in boards:
                policy = np.zeros(ACTION_DIM, dtype=np.float32)
                actions = board.legal_actions()
                policy[list(actions)] = 1.0 / len(actions)
                results.append((policy, 0.0))
            return results

    def test_batched_search_visits_only_legal_moves(self) -> None:
        evaluator = self.UniformEvaluator()
        mcts = NeuralMCTS(
            evaluator,
            MCTSConfig(simulations=8, exact_endgame_empties=0),
        )
        roots = [MCTSNode(BitBoard.initial()) for _ in range(4)]
        results = mcts.search_many(
            roots,
            add_root_noise=False,
            rng=np.random.default_rng(2),
        )
        legal = set(BitBoard.initial().legal_actions())
        for result in results:
            self.assertEqual(int(result.visit_counts.sum()), 8)
            self.assertEqual(
                set(np.flatnonzero(result.visit_counts)),
                legal,
            )
            self.assertAlmostEqual(float(result.policy.sum()), 1.0)
        self.assertLessEqual(len(evaluator.calls), 9)
        self.assertEqual(evaluator.calls[0], 1)

    def test_within_root_batching_has_no_virtual_visit_residue(self) -> None:
        evaluator = self.UniformEvaluator()
        simulations = 12
        root = MCTSNode(BitBoard.initial())
        result = NeuralMCTS(
            evaluator,
            MCTSConfig(
                simulations=simulations,
                exact_endgame_empties=0,
                leaf_batch_size=4,
                virtual_loss=3.0,
            ),
        ).search(root, add_root_noise=False)

        self.assertGreater(max(evaluator.calls), 1)
        self.assertEqual(int(result.visit_counts.sum()), simulations)
        self.assertEqual(root.total_visits, simulations)

        pending = [root]
        while pending:
            node = pending.pop()
            self.assertEqual(node.total_visits, int(node.visit_counts.sum()))
            self.assertTrue(np.all(node.visit_counts >= 0))
            self.assertTrue(np.all(np.isfinite(node.value_sums)))
            np.testing.assert_allclose(node.value_sums, 0.0, atol=1e-7)
            pending.extend(node.children.values())
            if node.pass_child is not None:
                pending.append(node.pass_child)

    def test_malformed_evaluator_output_falls_back_to_finite_legal_priors(
        self,
    ) -> None:
        class MalformedEvaluator:
            def evaluate(self, boards):
                output = []
                bad_values = (np.nan, np.inf, -np.inf, -3.0)
                for board in boards:
                    priors = np.full(ACTION_DIM, np.inf, dtype=np.float32)
                    for index, action in enumerate(board.legal_actions()):
                        priors[action] = bad_values[index % len(bad_values)]
                    output.append((priors, np.nan))
                return output

        root = MCTSNode(BitBoard.initial())
        result = NeuralMCTS(
            MalformedEvaluator(),
            MCTSConfig(
                simulations=8,
                exact_endgame_empties=0,
                leaf_batch_size=4,
            ),
        ).search(root, add_root_noise=False)
        legal = set(root.board.legal_actions())

        self.assertEqual(int(result.visit_counts.sum()), 8)
        self.assertTrue(set(np.flatnonzero(result.visit_counts)).issubset(legal))
        self.assertTrue(np.all(np.isfinite(result.policy)))
        self.assertTrue(np.all(result.policy >= 0))
        self.assertAlmostEqual(float(result.policy.sum()), 1.0)
        self.assertTrue(np.isfinite(result.root_value))
        self.assertIsNotNone(root.priors)
        self.assertTrue(np.all(np.isfinite(root.priors)))
        self.assertTrue(np.all(root.priors >= 0))
        self.assertAlmostEqual(float(root.priors.sum()), 1.0)

    def test_selection_error_releases_all_virtual_reservations(self) -> None:
        evaluator = self.UniformEvaluator()
        root = MCTSNode(BitBoard.initial())
        mcts = NeuralMCTS(
            evaluator,
            MCTSConfig(
                simulations=8,
                exact_endgame_empties=0,
                leaf_batch_size=4,
            ),
        )
        original_select = mcts._select_leaf
        calls = 0

        def fail_after_one(selected_root):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected selection failure")
            return original_select(selected_root)

        with (
            mock.patch.object(mcts, "_select_leaf", side_effect=fail_after_one),
            self.assertRaisesRegex(RuntimeError, "injected selection failure"),
        ):
            mcts.search(root, add_root_noise=False)

        self.assertEqual(root.total_visits, 0)
        np.testing.assert_array_equal(root.visit_counts, 0)
        np.testing.assert_allclose(root.value_sums, 0.0)

    def test_exact_solver_matches_reference_recursion(self) -> None:
        rng = random.Random(23)
        game = Game()
        color = Game.BLACK
        while sum(square == Game.EMPTY for row in game.board for square in row) > 4:
            moves = game.legal_moves(color)
            if not moves:
                color = Game.WHITE if color == Game.BLACK else Game.BLACK
                if not game.legal_moves(color):
                    break
                continue
            game.play(color, rng.choice(moves))
            color = Game.WHITE if color == Game.BLACK else Game.BLACK

        board = BitBoard.from_game(game, color)

        def reference_value(state: Game, to_play: int, perspective: int) -> int:
            moves = state.legal_moves(to_play)
            other = Game.WHITE if to_play == Game.BLACK else Game.BLACK
            if not moves:
                if not state.legal_moves(other):
                    scores = state.get_score()
                    return (
                        scores[perspective]
                        - scores[
                            Game.WHITE if perspective == Game.BLACK else Game.BLACK
                        ]
                    )
                return reference_value(state, other, perspective)
            values = []
            for move in moves:
                state.play(to_play, move)
                try:
                    values.append(reference_value(state, other, perspective))
                finally:
                    state.undo(to_play, move)
            return max(values) if to_play == perspective else min(values)

        expected = reference_value(game.copy(), color, color)
        self.assertEqual(ExactSolver().solve(board), expected)

    def test_backup_negates_once_for_moves_and_passes(self) -> None:
        parent = MCTSNode(BitBoard.initial())
        child = parent.child_for_action(parent.board.legal_actions()[0])
        action = parent.board.legal_actions()[0]
        NeuralMCTS._backup(
            [(parent, action), (child, None)],
            leaf_value=0.75,
        )
        self.assertEqual(parent.visit_counts[action], 1)
        self.assertAlmostEqual(float(parent.value_sums[action]), 0.75)


class NetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)
        self.model = AlphaZeroNetwork(AlphaZeroModelConfig(8, 1, 16))

    def test_network_shapes_and_policy_mask(self) -> None:
        board = BitBoard.initial()
        states = torch.from_numpy(encode_boards([board]))
        policy, wdl, margin, ownership = self.model(states)
        inference_policy, inference_wdl = self.model.inference(states)
        self.assertEqual(tuple(policy.shape), (1, 64))
        self.assertEqual(tuple(wdl.shape), (1, 3))
        self.assertEqual(tuple(margin.shape), (1,))
        self.assertEqual(tuple(ownership.shape), (1, 3, 8, 8))
        torch.testing.assert_close(inference_policy, policy)
        torch.testing.assert_close(inference_wdl, wdl)
        mask = torch.zeros((1, 64), dtype=torch.bool)
        mask[0, list(board.legal_actions())] = True
        probabilities = torch.softmax(
            self.model.masked_policy_logits(policy, mask),
            dim=-1,
        )
        self.assertTrue(torch.all(probabilities[~mask] == 0))
        self.assertAlmostEqual(float(probabilities.sum().detach()), 1.0, places=6)

    def test_checkpoint_and_player_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.az"
            save_checkpoint(
                path,
                self.model,
                training_state={"generation": 4},
            )
            loaded, payload = load_checkpoint(path)
            self.assertEqual(loaded.config, self.model.config)
            self.assertEqual(payload["training_state"]["generation"], 4)
            player = AlphaZeroPlayer(
                path,
                device="cpu",
                simulations=2,
                exact_endgame_empties=0,
            )
            game = Game()
            legal = game.legal_moves(Game.BLACK)
            original = [row[:] for row in game.board]
            coordinate = player.choose_move(
                game,
                Game.BLACK,
                legal,
                random.Random(1),
            )
            self.assertIn(
                coordinate,
                {(move.x, move.y) for move in legal},
            )
            self.assertEqual(game.board, original)
            self.assertIn("generation 4", player.name)

            benchmark_player = build_player(f"az:{path}")
            self.assertIsInstance(benchmark_player, AlphaZeroPlayer)
            bound_game = Game()
            bound = ComputerAlphaZero(
                bound_game,
                Game.BLACK,
                path=path,
                device="cpu",
                simulations=1,
                exact_endgame_empties=0,
            )
            bound_move = bound.pick_model(place=False)
            self.assertIn(bound_move, bound_game.get_all_legal_moves(Game.BLACK))

            first_move = next(move for move in legal if (move.x, move.y) == coordinate)
            game.play(Game.BLACK, first_move)
            white_move = game.legal_moves(Game.WHITE)[0]
            game.play(Game.WHITE, white_move)
            second_legal = game.legal_moves(Game.BLACK)
            second_coordinate = player.choose_move(
                game,
                Game.BLACK,
                second_legal,
                random.Random(2),
            )
            self.assertIn(
                second_coordinate,
                {(move.x, move.y) for move in second_legal},
            )

    def test_checkpoint_can_select_raw_or_ema_weights(self) -> None:
        ema_model = AlphaZeroNetwork(self.model.config)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.fill_(0.125)
            for parameter in ema_model.parameters():
                parameter.fill_(-0.375)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ema-model.az"
            save_checkpoint(path, self.model, ema_model=ema_model)
            raw_model, raw_payload = load_checkpoint(path)
            loaded_ema, ema_payload = load_checkpoint(path, use_ema=True)

        self.assertIn("ema_model_state", raw_payload)
        self.assertIn("ema_model_state", ema_payload)
        for name, expected in self.model.state_dict().items():
            torch.testing.assert_close(raw_model.state_dict()[name], expected)
        for name, expected in ema_model.state_dict().items():
            torch.testing.assert_close(loaded_ema.state_dict()[name], expected)
        first_raw = next(raw_model.parameters())
        first_ema = next(loaded_ema.parameters())
        self.assertFalse(torch.equal(first_raw, first_ema))

    def test_evaluator_batches_duplicate_states_once(self) -> None:
        evaluator = AlphaZeroEvaluator(
            self.model,
            "cpu",
            maximum_batch_size=16,
            cache_size=16,
        )
        board = BitBoard.initial()
        with (
            mock.patch.object(
                self.model,
                "inference",
                wraps=self.model.inference,
            ) as inference,
            mock.patch.object(
                self.model.ownership_conv,
                "forward",
                wraps=self.model.ownership_conv.forward,
            ) as ownership,
        ):
            first = evaluator.evaluate([board, board, board])
            second = evaluator.evaluate([board])
        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 1)
        self.assertEqual(inference.call_count, 1)
        self.assertEqual(ownership.call_count, 0)


class TrainerTests(unittest.TestCase):
    @staticmethod
    def replay_record(generation: int = 1) -> SelfPlayRecord:
        board = BitBoard.initial()
        visits = np.zeros(ACTION_DIM, dtype=np.int32)
        visits[board.legal_actions()[0]] = 1
        return SelfPlayRecord(
            player=board.player,
            opponent=board.opponent,
            visit_counts=visits,
            outcome=0,
            margin=0.0,
            ownership=np.ones(ACTION_DIM, dtype=np.int8),
            generation=generation,
        )

    @staticmethod
    def tiny_config(output: Path) -> TrainingConfig:
        return TrainingConfig(
            generations=1,
            games_per_generation=2,
            simulations=1,
            parallel_games=2,
            self_play_workers=1,
            channels=8,
            residual_blocks=1,
            value_hidden=16,
            exact_endgame_empties=4,
            inference_batch_size=64,
            inference_cache_size=1_000,
            replay_capacity=1_000,
            minimum_replay_size=1,
            batch_size=16,
            training_steps=1,
            evaluation_every=0,
            checkpoint_every=1,
            snapshot_every=0,
            replay_save_every=1,
            seed=17,
            device="cpu",
            output_directory=output,
        )

    def test_self_play_replay_and_update_are_finite(self) -> None:
        config = self.tiny_config(Path("unused"))
        model = AlphaZeroNetwork(AlphaZeroModelConfig(8, 1, 16))
        records, metrics = collect_self_play(
            model,
            torch.device("cpu"),
            config,
            generation=1,
        )
        self.assertEqual(metrics.games, 2)
        self.assertGreater(len(records), 80)

        replay = ReplayBuffer(1_000)
        replay.add(records)
        batch = replay.sample(
            16,
            np.random.default_rng(3),
        )
        self.assertEqual(batch.states.shape, (16, 4, 8, 8))
        self.assertTrue(np.allclose(batch.policy_targets.sum(axis=1), 1.0))
        self.assertTrue(
            np.all(
                np.logical_or(
                    batch.policy_targets == 0,
                    batch.legal_masks,
                )
            )
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        update = update_network(
            model,
            optimizer,
            replay,
            torch.device("cpu"),
            config,
            np.random.default_rng(4),
        )
        for value in (
            update.policy_loss,
            update.wdl_loss,
            update.margin_loss,
            update.ownership_loss,
            update.total_loss,
            update.gradient_norm,
            update.wdl_accuracy,
            update.explained_variance,
        ):
            self.assertTrue(np.isfinite(value))

    def test_replay_round_trip(self) -> None:
        config = self.tiny_config(Path("unused"))
        model = AlphaZeroNetwork(AlphaZeroModelConfig(8, 1, 16))
        records, _ = collect_self_play(
            model,
            "cpu",
            config,
            generation=2,
            games=1,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "replay.npz"
            replay = ReplayBuffer(1_000)
            replay.add(records)
            replay.save(path)
            restored = ReplayBuffer(1_000)
            restored.load(path)
            self.assertEqual(len(restored), len(replay))
            np.testing.assert_array_equal(
                restored.visit_counts[: len(restored)],
                replay.visit_counts[: len(replay)],
            )
            compressed_path = Path(temporary_directory) / "replay-compressed.npz"
            replay.save(compressed_path, compressed=True)
            compressed = ReplayBuffer(1_000)
            compressed.load(compressed_path)
            np.testing.assert_array_equal(
                compressed.priority[: len(compressed)],
                replay.priority[: len(replay)],
            )

    def test_replay_sample_size_is_exact_for_small_batches_and_fractions(
        self,
    ) -> None:
        fractions = (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 0.5),
            (0.5, 0.2),
            (0.34, 0.33),
        )
        for recent_fraction, prioritized_fraction in fractions:
            replay = ReplayBuffer(
                8,
                recent_fraction=recent_fraction,
                prioritized_fraction=prioritized_fraction,
            )
            replay.add([self.replay_record(), self.replay_record(2)])
            for batch_size in range(1, 10):
                with self.subTest(
                    recent_fraction=recent_fraction,
                    prioritized_fraction=prioritized_fraction,
                    batch_size=batch_size,
                ):
                    batch = replay.sample(
                        batch_size,
                        np.random.default_rng(batch_size),
                        augment_symmetry=False,
                    )
                    self.assertEqual(batch.size, batch_size)
                    self.assertEqual(batch.states.shape[0], batch_size)
                    self.assertEqual(batch.policy_targets.shape[0], batch_size)

    def test_duplicate_priority_updates_are_order_independent(self) -> None:
        replay = ReplayBuffer(8)
        replay.add([self.replay_record(), self.replay_record(2)])
        replay.update_priorities(
            np.asarray([0, 1, 0], dtype=np.int64),
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        )
        first = replay.priority[:2].copy()
        replay.update_priorities(
            np.asarray([0, 0, 1], dtype=np.int64),
            np.asarray([3.0, 1.0, 2.0], dtype=np.float32),
        )
        second = replay.priority[:2].copy()

        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(second, np.asarray([2.0, 2.0]))

    def test_reanalysis_refreshes_old_search_targets(self) -> None:
        config = self.tiny_config(Path("unused"))
        config.reanalysis_positions = 12
        config.reanalysis_minimum_age = 1
        config.reanalysis_batch_size = 6
        model = AlphaZeroNetwork(AlphaZeroModelConfig(8, 1, 16))
        records, _ = collect_self_play(
            model,
            "cpu",
            config,
            generation=1,
            games=1,
        )
        replay = ReplayBuffer(1_000)
        replay.add(records)
        metrics = reanalyse_replay(
            model,
            torch.device("cpu"),
            replay,
            config,
            generation=3,
            rng=np.random.default_rng(8),
        )
        self.assertEqual(metrics.positions, 12)
        self.assertEqual(int(np.sum(replay.generation == 3)), 12)

    def test_spawned_cpu_self_play_workers_return_complete_games(self) -> None:
        config = self.tiny_config(Path("unused"))
        config.self_play_workers = 2
        config.parallel_games = 1
        model = AlphaZeroNetwork(AlphaZeroModelConfig(8, 1, 16))
        records, metrics = collect_self_play_parallel(
            model,
            torch.device("cpu"),
            config,
            generation=3,
        )
        self.assertEqual(metrics.games, 2)
        self.assertGreater(len(records), 80)

    def test_batched_paired_evaluation_is_color_balanced(self) -> None:
        config = self.tiny_config(Path("unused"))
        config.evaluation_pairs = 2
        config.evaluation_simulations = 1
        model = AlphaZeroNetwork(AlphaZeroModelConfig(8, 1, 16))
        metrics = evaluate_candidate(
            model,
            model,
            torch.device("cpu"),
            config,
            generation=2,
        )
        self.assertEqual(metrics["evaluation_games"], 4.0)
        self.assertAlmostEqual(metrics["evaluation_score"], 0.5)

    def test_tiny_training_writes_independent_checkpoint_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            checkpoint = train(self.tiny_config(output))
            self.assertEqual(checkpoint, output.resolve() / "latest.az")
            self.assertTrue((output / "best.az").is_file())
            self.assertTrue((output / "replay.npz").is_file())
            self.assertTrue((output / "metrics.jsonl").is_file())
            _, payload = load_checkpoint(checkpoint)
            self.assertEqual(payload["training_state"]["generation"], 1)
            self.assertNotIn("teacher", payload["training_config"])
            self.assertNotIn("wthor", str(payload).lower())


if __name__ == "__main__":
    unittest.main()
