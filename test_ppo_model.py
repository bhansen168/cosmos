"""Focused tests for the PPO Othello model and trainer."""

from __future__ import annotations

import math
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from benchmark_models import build_player
from game import Game
from ppo_model import (
    ACTION_DIM,
    ACTION_TRANSFORMS,
    ModelConfig,
    PPOActorCritic,
    PPOPlayer,
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
    TrainingConfig,
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
            self.assertGreaterEqual(player.get_value_prediction(game, Game.BLACK), -1)
            self.assertLessEqual(player.get_value_prediction(game, Game.BLACK), 1)


class TrainerTests(unittest.TestCase):
    @staticmethod
    def tiny_config(output_directory: Path) -> TrainingConfig:
        return TrainingConfig(
            iterations=1,
            rollout_steps=8,
            ppo_epochs=1,
            minibatch_size=16,
            self_play_fraction=1.0,
            league_fraction=0.0,
            baseline_fraction=0.0,
            checkpoint_every=0,
            snapshot_every=0,
            validation_every=0,
            validation_pairs=1,
            channels=8,
            residual_blocks=1,
            seed=11,
            device="cpu",
            output_directory=output_directory,
        )

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
                metrics.entropy,
                metrics.approximate_kl,
                metrics.clip_fraction,
                metrics.explained_variance,
            ):
                self.assertTrue(math.isfinite(value))

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
            )
            latest = train(config)
            best = output / "best.ppo"
            snapshots = list((output / "league").glob("*.ppo"))
            self.assertTrue(best.is_file())
            self.assertEqual(len(snapshots), 1)
            _, payload = load_checkpoint(latest)
            self.assertIn("validation_composite", payload["metrics"])
            self.assertEqual(len(payload["league"]), 1)

            already_complete = train(replace(config, resume=latest))
            self.assertEqual(already_complete, latest.resolve())


if __name__ == "__main__":
    unittest.main()
