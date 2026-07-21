"""Focused regression tests for the strength-oriented genetic trainer."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from genetic_model import (
    DEFAULT_SEED_GENOME,
    GENOME_SIZE,
    GeneticPlayer,
    Individual,
    TrainingConfig,
    _challenge_passed,
    _checkpoint_filename,
    _eligible_for_challenge,
    _latest_checkpoint_filename,
    _normalize_genome_scale,
    _validation_opponents,
    load_checkpoint,
    train,
)


class StrengthLeagueTests(unittest.TestCase):
    def test_validation_league_is_diverse_and_excludes_unready_models(self) -> None:
        config = TrainingConfig(population_size=6)
        archive = [
            Individual([value + offset for value in DEFAULT_SEED_GENOME])
            for offset in (0.1, 0.2, 0.3, 0.4)
        ]

        first = _validation_opponents(config, generation=8, hall_of_fame=archive)
        second = _validation_opponents(config, generation=8, hall_of_fame=archive)
        names = [name for name, _, _ in first]

        self.assertEqual(names, [name for name, _, _ in second])
        self.assertEqual(
            [getattr(player, "genome", None) for _, player, _ in first],
            [getattr(player, "genome", None) for _, player, _ in second],
        )
        self.assertIn("random", names)
        self.assertIn("greedy", names)
        self.assertIn("seed_genetic", names)
        self.assertIn("minimax_depth_1", names)
        self.assertIn("minimax_depth_2", names)
        self.assertIn("minimax_depth_3", names)
        self.assertEqual(
            sum(name.startswith("historical_genetic_") for name in names),
            config.validation_hall_of_fame_opponents,
        )
        self.assertFalse(any("bard" in name or "dqn" in name for name in names))

    def test_near_equal_leader_can_challenge_and_ties_use_margin(self) -> None:
        config = TrainingConfig(population_size=6)

        self.assertTrue(_eligible_for_challenge(-0.002, config))
        self.assertFalse(_eligible_for_challenge(-0.02, config))
        self.assertTrue(_challenge_passed(0.50, 1.0, 0.50))
        self.assertFalse(_challenge_passed(0.50, -1.0, 0.50))
        self.assertTrue(_challenge_passed(0.55, -3.0, 0.50))

    def test_genome_normalization_removes_only_global_scale(self) -> None:
        genome = [float(index + 1) for index in range(GENOME_SIZE)]
        ratios = [genome[index] / genome[0] for index in range(GENOME_SIZE)]

        _normalize_genome_scale(genome, gene_limit=100.0)

        rms = math.sqrt(sum(value * value for value in genome) / len(genome))
        self.assertAlmostEqual(rms, 1.0)
        for index, ratio in enumerate(ratios):
            self.assertAlmostEqual(genome[index] / genome[0], ratio)


class VersionedCheckpointTests(unittest.TestCase):
    def test_training_writes_only_v2_checkpoint_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            checkpoint = train(
                TrainingConfig(
                    generations=1,
                    population_size=4,
                    games_per_pair=1,
                    coevolution_opponents=1,
                    baseline_games=0,
                    minimax_games=0,
                    minimax_depth=1,
                    training_search_depth=1,
                    search_depth=1,
                    endgame_exact_empties=0,
                    opening_plies=2,
                    validation_candidates=1,
                    validation_openings=1,
                    validation_folds=2,
                    validation_hall_of_fame_opponents=0,
                    challenge_openings=1,
                    hall_of_fame_opponents=0,
                    elite_count=1,
                    tournament_size=2,
                    random_immigrants=1,
                    checkpoint_every=1,
                    output_directory=output_directory,
                    seed=91,
                )
            )

            self.assertEqual(checkpoint.name, _checkpoint_filename(0, "v2"))
            self.assertTrue(
                (output_directory / _latest_checkpoint_filename("v2")).is_file()
            )
            self.assertFalse((output_directory / "genetic_gen_0000.json").exists())
            self.assertFalse((output_directory / "latest.json").exists())

            payload = load_checkpoint(checkpoint)
            self.assertIsNotNone(payload["validation_leader"])
            self.assertEqual(payload["config"]["checkpoint_suffix"], "v2")
            self.assertTrue(payload["config"]["normalize_genomes"])

            player = GeneticPlayer.from_checkpoint(checkpoint)
            self.assertIn("champion generation 0", player.name)
            self.assertIn("checkpoint generation 0", player.name)

    def test_checkpoint_suffix_rejects_paths(self) -> None:
        for invalid in ("../v2", "v2.json", "", "v2/latest"):
            with self.subTest(invalid=invalid):
                config = TrainingConfig(
                    population_size=6,
                    checkpoint_suffix=invalid,
                )
                with self.assertRaisesRegex(ValueError, "checkpoint suffix"):
                    config.validate()


if __name__ == "__main__":
    unittest.main()
