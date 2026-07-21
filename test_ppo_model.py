"""Focused tests for the PPO Othello model and trainer."""

from __future__ import annotations

import math
import os
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from benchmark_models import build_player
from computer import ComputerPPO, create_ppo_computer, find_latest_ppo_checkpoint
from game import Game
from ppo_model import (
    ACTION_DIM,
    ACTION_TRANSFORMS,
    ModelConfig,
    PPOActorCritic,
    PPOPlayer,
    SearchConfig,
    choose_search_move,
    encode_state,
    inverse_transform_action,
    legal_moves_mask,
    load_checkpoint,
    save_checkpoint,
    transform_action,
    transform_mask,
    transform_planes,
)
from train_ppo import (
    Experience,
    TrainingConfig,
    _finish_trajectory,
    collect_rollout,
    train,
    update_policy,
)


class EncodingAndSymmetryTests(unittest.TestCase):
    def test_encoding_is_from_active_players_perspective(self) -> None:
        game = Game()
        black = encode_state(game, Game.BLACK)
        white = encode_state(game, Game.WHITE)

        self.assertEqual(black.shape, (3, 8, 8))
        np.testing.assert_array_equal(black[0], white[1])
        np.testing.assert_array_equal(black[1], white[0])
        self.assertEqual(int(black[2].sum()), 4)
        self.assertEqual(int(white[2].sum()), 4)

    def test_all_symmetries_are_invertible_and_preserve_legal_actions(self) -> None:
        game = Game()
        legal_moves = game.legal_moves(Game.BLACK)
        planes = encode_state(game, Game.BLACK, legal_moves)
        mask = legal_moves_mask(legal_moves)
        action = int(np.flatnonzero(mask)[0])

        for symmetry in range(8):
            with self.subTest(symmetry=symmetry):
                transformed_planes = transform_planes(planes, symmetry)
                transformed_mask = transform_mask(mask, symmetry)
                transformed_action = transform_action(action, symmetry)
                self.assertEqual(transformed_planes.shape, planes.shape)
                self.assertEqual(int(transformed_mask.sum()), int(mask.sum()))
                self.assertTrue(transformed_mask[transformed_action])
                self.assertEqual(
                    inverse_transform_action(transformed_action, symmetry),
                    action,
                )
                self.assertEqual(
                    sorted(ACTION_TRANSFORMS[symmetry].tolist()),
                    list(range(ACTION_DIM)),
                )


class NetworkAndCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = PPOActorCritic(ModelConfig(channels=8, residual_blocks=1))

    def test_policy_mask_excludes_every_illegal_action(self) -> None:
        game = Game()
        legal_moves = game.legal_moves(Game.BLACK)
        state = torch.from_numpy(encode_state(game, Game.BLACK)).unsqueeze(0)
        mask = torch.from_numpy(legal_moves_mask(legal_moves)).unsqueeze(0)

        distribution, value = self.model.distribution_and_value(state, mask)
        probabilities = distribution.probs.squeeze(0)
        self.assertEqual(tuple(probabilities.shape), (ACTION_DIM,))
        self.assertEqual(tuple(value.shape), (1,))
        self.assertTrue(torch.all(probabilities[~mask.squeeze(0)] == 0))
        self.assertAlmostEqual(float(probabilities.sum().detach()), 1.0, places=6)
        for _ in range(100):
            self.assertTrue(bool(mask[0, distribution.sample().item()]))

    def test_checkpoint_round_trip_and_benchmark_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "test.ppo"
            save_checkpoint(
                checkpoint,
                self.model,
                training_state={"iteration": 3},
                metrics={"score": 0.25},
            )
            loaded, payload = load_checkpoint(checkpoint)
            self.assertEqual(loaded.config, self.model.config)
            self.assertEqual(payload["training_state"]["iteration"], 3)
            for expected, actual in zip(
                self.model.parameters(), loaded.parameters(), strict=True
            ):
                self.assertTrue(torch.equal(expected, actual))

            player = build_player(f"ppo:{checkpoint}")
            self.assertIsInstance(player, PPOPlayer)
            game = Game()
            original_board = [row.copy() for row in game.board]
            legal_moves = game.legal_moves(Game.BLACK)
            coordinate = player.choose_move(
                game,
                Game.BLACK,
                legal_moves,
                random.Random(1),
            )
            self.assertIn(
                coordinate,
                {(move.x, move.y) for move in legal_moves},
            )
            self.assertEqual(game.board, original_board)
            self.assertGreaterEqual(player.get_value_prediction(game, Game.BLACK), -1)
            self.assertLessEqual(player.get_value_prediction(game, Game.BLACK), 1)

    def test_bound_computer_adapter_and_checkpoint_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy = root / "latest.ppo"
            current = root / "ppo" / "latest.ppo"
            current.parent.mkdir()
            save_checkpoint(legacy, self.model)
            save_checkpoint(current, self.model)
            os.utime(legacy, (1, 1))
            os.utime(current, (2, 2))
            self.assertEqual(find_latest_ppo_checkpoint(root), current.resolve())

            game = Game()
            original_board = [row.copy() for row in game.board]
            computer = ComputerPPO(
                game,
                Game.BLACK,
                path=current,
                device="cpu",
                search_depth=0,
                endgame_exact_empties=0,
            )
            coordinate = computer.pick_model(place=False)
            self.assertIn(coordinate, game.get_all_legal_moves(Game.BLACK))
            self.assertEqual(game.board, original_board)
            self.assertGreaterEqual(computer.get_value_prediction(), -1.0)
            self.assertLessEqual(computer.get_value_prediction(), 1.0)

            factory_computer = create_ppo_computer(
                game,
                Game.BLACK,
                checkpoint_path=current,
                device="cpu",
                search_depth=0,
                endgame_exact_empties=0,
            )
            self.assertIsInstance(factory_computer, ComputerPPO)
            factory_computer.pick()
            self.assertEqual(game.get_score(), {Game.BLACK: 4, Game.WHITE: 1})

    def test_policy_guided_search_preserves_board(self) -> None:
        game = Game()
        legal_moves = game.legal_moves(Game.BLACK)
        original_board = [row.copy() for row in game.board]
        with mock.patch.object(
            self.model,
            "forward",
            wraps=self.model.forward,
        ) as forward:
            coordinate = choose_search_move(
                self.model,
                torch.device("cpu"),
                game,
                Game.BLACK,
                legal_moves,
                SearchConfig(depth=2, endgame_exact_empties=0),
            )
        self.assertIn(coordinate, {(move.x, move.y) for move in legal_moves})
        self.assertEqual(game.board, original_board)
        self.assertEqual(forward.call_count, 2)

    def test_exact_endgame_search_preserves_board(self) -> None:
        game = Game()
        rng = random.Random(19)
        color = Game.BLACK
        while sum(square == Game.EMPTY for row in game.board for square in row) > 6:
            legal_moves = game.legal_moves(color)
            if not legal_moves:
                color = Game.WHITE if color == Game.BLACK else Game.BLACK
                legal_moves = game.legal_moves(color)
                if not legal_moves:
                    break
            game.play(color, rng.choice(legal_moves))
            color = Game.WHITE if color == Game.BLACK else Game.BLACK

        legal_moves = game.legal_moves(color)
        if not legal_moves:
            color = Game.WHITE if color == Game.BLACK else Game.BLACK
            legal_moves = game.legal_moves(color)
        self.assertTrue(legal_moves)
        original_board = [row.copy() for row in game.board]
        coordinate = choose_search_move(
            self.model,
            torch.device("cpu"),
            game,
            color,
            legal_moves,
            SearchConfig(depth=0, endgame_exact_empties=6),
        )
        self.assertIn(coordinate, {(move.x, move.y) for move in legal_moves})
        self.assertEqual(game.board, original_board)


class TrainerTests(unittest.TestCase):
    @staticmethod
    def tiny_config(output_directory: Path) -> TrainingConfig:
        return TrainingConfig(
            iterations=1,
            rollout_steps=8,
            ppo_epochs=1,
            minibatch_size=16,
            parallel_games=4,
            teacher_fraction=0.0,
            self_play_fraction=1.0,
            league_fraction=0.0,
            baseline_fraction=0.0,
            checkpoint_every=0,
            snapshot_every=0,
            validation_every=0,
            champion_every=0,
            validation_pairs=1,
            channels=8,
            residual_blocks=1,
            seed=11,
            device="cpu",
            output_directory=output_directory,
            genetic_checkpoint=output_directory / "missing-genetic.json",
        )

    def test_value_targets_use_final_outcome_instead_of_critic_bootstrap(self) -> None:
        config = replace(
            self.tiny_config(Path("unused")),
            score_target_weight=0.10,
            gae_lambda=0.95,
        )
        trajectory = [
            Experience(
                state=np.zeros((3, 8, 8), dtype=np.float32),
                legal_mask=np.ones(64, dtype=np.bool_),
                action=0,
                old_log_probability=-1.0,
                old_value=value,
            )
            for value in (-0.5, 0.1, 0.7)
        ]
        _finish_trajectory(trajectory, outcome=1.0, score_margin=0.25, config=config)
        expected = 0.9 + 0.1 * 0.25
        for experience in trajectory:
            self.assertAlmostEqual(experience.return_value, expected)
            self.assertTrue(math.isfinite(experience.advantage))

    def test_rollout_and_update_are_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = self.tiny_config(Path(temporary_directory))
            torch.manual_seed(config.seed)
            model = PPOActorCritic(ModelConfig(8, 1))
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            batch = collect_rollout(
                model,
                torch.device("cpu"),
                random.Random(config.seed),
                config,
                league=[],
                league_cache={},
            )

            self.assertGreaterEqual(batch.size, config.rollout_steps)
            self.assertEqual(batch.games, 1)
            legal_taken = batch.legal_masks.gather(1, batch.actions.unsqueeze(1))
            self.assertTrue(bool(torch.all(legal_taken)))
            metrics = update_policy(
                model,
                optimizer,
                batch,
                torch.device("cpu"),
                config,
                iteration=0,
            )
            for value in (
                metrics.policy_loss,
                metrics.value_loss,
                metrics.teacher_loss,
                metrics.entropy,
                metrics.normalized_entropy,
                metrics.approximate_kl,
                metrics.clip_fraction,
                metrics.explained_variance,
            ):
                self.assertTrue(math.isfinite(value))

    def test_parallel_rollout_batches_policy_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = replace(
                self.tiny_config(Path(temporary_directory)),
                rollout_steps=180,
                parallel_games=4,
            )
            model = PPOActorCritic(ModelConfig(8, 1))
            with mock.patch.object(model, "forward", wraps=model.forward) as forward:
                batch = collect_rollout(
                    model,
                    torch.device("cpu"),
                    random.Random(config.seed),
                    config,
                    league=[],
                    league_cache={},
                )

            self.assertGreaterEqual(batch.size, config.rollout_steps)
            self.assertGreaterEqual(batch.games, 4)
            self.assertLess(forward.call_count, batch.size / 2)

    def test_tiny_training_run_writes_resumable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            config = self.tiny_config(output)
            checkpoint = train(config)
            self.assertEqual(checkpoint, output.resolve() / "latest.ppo")
            self.assertTrue(checkpoint.is_file())
            _, payload = load_checkpoint(checkpoint)
            self.assertEqual(payload["training_state"]["iteration"], 1)
            self.assertGreater(payload["training_state"]["total_steps"], 0)
            self.assertIn("optimizer_state", payload)

            first_steps = payload["training_state"]["total_steps"]
            resumed = train(replace(config, iterations=2, resume=checkpoint))
            _, resumed_payload = load_checkpoint(resumed)
            self.assertEqual(resumed_payload["training_state"]["iteration"], 2)
            self.assertGreater(
                resumed_payload["training_state"]["total_steps"],
                first_steps,
            )
            self.assertGreater(
                resumed_payload["metrics"]["learning_rate"],
                0.0,
            )

    def test_validation_and_league_snapshot_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            config = replace(
                self.tiny_config(output),
                checkpoint_every=1,
                snapshot_every=1,
                validation_every=1,
                champion_every=1,
                champion_pairs=1,
                champion_search_depth=0,
                champion_endgame_exact_empties=0,
                champion_max_minimax_depth=2,
            )
            latest = train(config)
            best = output / "best.ppo"
            snapshots = list((output / "league").glob("*.ppo"))
            self.assertTrue(best.is_file())
            self.assertEqual(len(snapshots), 1)
            _, payload = load_checkpoint(latest)
            self.assertIn("validation_composite", payload["metrics"])
            self.assertIn("champion_composite", payload["metrics"])
            self.assertEqual(len(payload["league"]), 1)
            self.assertEqual(len(payload["hall_of_fame"]), 1)

            already_complete = train(replace(config, resume=latest))
            self.assertEqual(already_complete, latest.resolve())


if __name__ == "__main__":
    unittest.main()
