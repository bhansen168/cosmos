"""Focused regression tests for the strength-oriented genetic trainer."""

from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

from genetic_model import (
    DEFAULT_SEED_GENOME,
    GENOME_SIZE,
    GeneticPlayer,
    Individual,
    LegacyGeneticPlayer,
    TrainingConfig,
    _challenge_passed,
    _checkpoint_filename,
    _crossover,
    _eligible_for_challenge,
    _latest_checkpoint_filename,
    _normalize_genome_scale,
    _promotion_guardrails,
    _restore_hall_of_fame,
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
        self.assertIn("reference_pre_v2_gen24_early", names)
        self.assertIn("reference_v2_gen24", names)
        self.assertEqual(
            sum(name.startswith("reference_") for name in names),
            2,
        )
        self.assertEqual(
            sum(name.startswith("historical_genetic_") for name in names),
            config.validation_hall_of_fame_opponents,
        )
        self.assertFalse(any("bard" in name or "dqn" in name for name in names))

    def test_reference_opponents_rotate_and_can_be_disabled(self) -> None:
        config = TrainingConfig(population_size=6, validation_every=1)
        historical_names = []
        for generation in range(3):
            names = [
                name
                for name, _, _ in _validation_opponents(
                    config,
                    generation=generation,
                )
                if name.startswith("reference_")
                and name != "reference_v2_gen24"
            ]
            self.assertEqual(len(names), 1)
            historical_names.extend(names)

        self.assertEqual(len(set(historical_names)), 3)
        disabled = TrainingConfig(population_size=6, reference_weight=0.0)
        self.assertFalse(
            any(
                name.startswith("reference_")
                for name, _, _ in _validation_opponents(
                    disabled,
                    generation=0,
                )
            )
        )

    def test_promotion_requires_external_and_head_to_head_improvement(self) -> None:
        config = TrainingConfig(population_size=6)

        self.assertFalse(_eligible_for_challenge(-0.002, config))
        self.assertTrue(_eligible_for_challenge(0.01, config))
        self.assertFalse(_challenge_passed(0.50, 12.0, 0.55))
        self.assertTrue(_challenge_passed(0.55, -3.0, 0.55))

    def test_promotion_guardrails_reject_baseline_regression(self) -> None:
        config = TrainingConfig(population_size=6)
        names = (
            "random",
            "greedy",
            "seed_genetic",
            "minimax_depth_3",
            "reference_original_v1",
            "reference_pre_v2_gen24_early",
            "reference_pre_v2_gen24_late",
            "reference_v2_gen24",
        )
        incumbent = Individual(
            list(DEFAULT_SEED_GENOME),
            validation_breakdown={
                name: {"score": 0.7} for name in names
            },
        )
        challenger = incumbent.copy()
        challenger.validation_breakdown["minimax_depth_3"] = {"score": 0.4}

        result = _promotion_guardrails(challenger, incumbent, config)

        self.assertFalse(result["passed"])
        self.assertIn("minimax_depth_3", result["regressions"])

        recovering = incumbent.copy()
        recovering.validation_breakdown = {
            name: {"score": 0.7} for name in names
        }
        incumbent.validation_breakdown["reference_v2_gen24"] = {"score": 0.2}
        recovering.validation_breakdown["reference_v2_gen24"] = {"score": 0.3}
        self.assertTrue(
            _promotion_guardrails(recovering, incumbent, config)["passed"]
        )

    def test_genome_normalization_removes_only_global_scale(self) -> None:
        genome = [float(index + 1) for index in range(GENOME_SIZE)]
        ratios = [genome[index] / genome[0] for index in range(GENOME_SIZE)]

        _normalize_genome_scale(genome, gene_limit=100.0)

        rms = math.sqrt(sum(value * value for value in genome) / len(genome))
        self.assertAlmostEqual(rms, 1.0)
        for index, ratio in enumerate(ratios):
            self.assertAlmostEqual(genome[index] / genome[0], ratio)

    def test_crossover_is_scale_invariant(self) -> None:
        first = [float(index + 1) for index in range(GENOME_SIZE)]
        second = [float(GENOME_SIZE - index) for index in range(GENOME_SIZE)]

        baseline = _crossover(first, second, random.Random(17), 100.0)
        scaled = _crossover(
            [value * 7.0 for value in first],
            [value * 0.2 for value in second],
            random.Random(17),
            100.0,
        )

        for first_value, second_value in zip(baseline, scaled):
            self.assertAlmostEqual(first_value, second_value)


class VersionedCheckpointTests(unittest.TestCase):
    def test_training_writes_only_v3_checkpoint_names(self) -> None:
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

            self.assertEqual(checkpoint.name, _checkpoint_filename(0, "v3"))
            self.assertTrue(
                (output_directory / _latest_checkpoint_filename("v3")).is_file()
            )
            self.assertFalse((output_directory / "genetic_gen_0000.json").exists())
            self.assertFalse((output_directory / "latest.json").exists())

            payload = load_checkpoint(checkpoint)
            self.assertIsNotNone(payload["validation_leader"])
            self.assertEqual(payload["config"]["checkpoint_suffix"], "v3")
            self.assertTrue(payload["config"]["normalize_genomes"])
            self.assertEqual(
                payload["training_state"]["hall_of_fame_policy"],
                "accepted_champions_only",
            )

            player = GeneticPlayer.from_checkpoint(checkpoint)
            self.assertIn("champion generation", player.name)
            self.assertIn("checkpoint generation 0", player.name)

    def test_original_reference_uses_exact_legacy_best_ever_player(self) -> None:
        reference = (
            Path(__file__).resolve().parent
            / "models"
            / "history"
            / "genetic"
            / "original_gen24_v1_reference.json"
        )

        player = GeneticPlayer.from_checkpoint(reference)

        self.assertIsInstance(player, LegacyGeneticPlayer)
        self.assertIn("best-ever", player.name)

    def test_old_contaminated_archive_restores_only_incumbent(self) -> None:
        champion = {
            "genome": list(DEFAULT_SEED_GENOME),
            "fitness": 0.7,
            "games": 10,
        }
        rejected = {
            "genome": [value + 0.1 for value in DEFAULT_SEED_GENOME],
            "fitness": 0.8,
            "games": 10,
        }
        payload = {
            "champion": champion,
            "hall_of_fame": [rejected],
            "training_state": {},
        }

        restored = _restore_hall_of_fame(
            payload,
            Path("unused.json"),
            maximum_size=8,
        )

        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].genome, champion["genome"])

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
