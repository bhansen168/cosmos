#!/usr/bin/env python3
"""Train and load an evolutionary Othello evaluation model.

The genome contains opening and endgame weights for six normalized board
features. A genetic player evaluates every legal move one ply ahead. Training
uses co-evolution within the population plus random, greedy, and shallow
minimax baselines so the population does not specialize against one opponent.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from minimax_model import MinimaxPlayer, POSITION_WEIGHTS
from othello_engine import (
    BLACK,
    BOARD_SIZE,
    DIRECTIONS,
    EMPTY,
    WHITE,
    GameOutcome,
    HeadlessOthello,
    LegalMove,
    Player,
    opponent,
    play_game,
)


CHECKPOINT_FORMAT = "cosmos-genetic-othello"
CHECKPOINT_VERSION = 1
FEATURE_NAMES = (
    "disc_difference",
    "opponent_mobility",
    "corners",
    "edges",
    "frontier_safety",
    "positional_value",
)
GENOME_SIZE = len(FEATURE_NAMES) * 2
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "models" / "genetic"
CORNER_COORDINATES = ((0, 0), (7, 0), (0, 7), (7, 7))
EDGE_COORDINATES = tuple(
    (x, y)
    for y in range(BOARD_SIZE)
    for x in range(BOARD_SIZE)
    if (x in (0, 7) or y in (0, 7)) and (x, y) not in CORNER_COORDINATES
)
POSITION_SCALE = float(sum(abs(value) for row in POSITION_WEIGHTS for value in row))

# One informed seed gives evolution a reasonable starting point without
# preventing the remaining population from exploring unrelated strategies.
DEFAULT_SEED_GENOME = (
    -0.20,  # opening disc count: taking too many pieces early can reduce mobility
    1.20,   # opening mobility
    2.00,   # opening corners
    0.30,   # opening edges
    0.80,   # opening frontier safety
    1.00,   # opening positional value
    1.50,   # endgame disc count
    0.40,   # endgame mobility
    2.00,   # endgame corners
    0.60,   # endgame edges
    0.35,   # endgame frontier safety
    0.80,   # endgame positional value
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _frontier_counts(game: HeadlessOthello, color: int) -> tuple[int, int]:
    other = opponent(color)
    own_frontier = 0
    other_frontier = 0

    for y, row in enumerate(game.board):
        for x, square in enumerate(row):
            if square == EMPTY:
                continue
            touches_empty = any(
                0 <= x + dx < BOARD_SIZE
                and 0 <= y + dy < BOARD_SIZE
                and game.board[y + dy][x + dx] == EMPTY
                for dx, dy in DIRECTIONS
            )
            if touches_empty:
                if square == color:
                    own_frontier += 1
                elif square == other:
                    other_frontier += 1
    return own_frontier, other_frontier


def extract_features(
    game: HeadlessOthello,
    color: int,
    opponent_move_count: int | None = None,
) -> tuple[float, ...]:
    """Return six normalized features from ``color``'s perspective."""
    other = opponent(color)
    scores = game.score()
    own_discs = scores[color]
    other_discs = scores[other]

    if opponent_move_count is None:
        opponent_move_count = len(game.legal_moves(other))

    own_corners = sum(game.board[y][x] == color for x, y in CORNER_COORDINATES)
    other_corners = sum(game.board[y][x] == other for x, y in CORNER_COORDINATES)
    own_edges = sum(game.board[y][x] == color for x, y in EDGE_COORDINATES)
    other_edges = sum(game.board[y][x] == other for x, y in EDGE_COORDINATES)
    own_frontier, other_frontier = _frontier_counts(game, color)
    total_frontier = own_frontier + other_frontier

    positional = 0
    for y, row in enumerate(game.board):
        for x, square in enumerate(row):
            if square == color:
                positional += POSITION_WEIGHTS[y][x]
            elif square == other:
                positional -= POSITION_WEIGHTS[y][x]

    return (
        (own_discs - other_discs) / (BOARD_SIZE * BOARD_SIZE),
        -min(opponent_move_count, 20) / 20.0,
        (own_corners - other_corners) / len(CORNER_COORDINATES),
        (own_edges - other_edges) / len(EDGE_COORDINATES),
        (other_frontier - own_frontier) / max(1, total_frontier),
        _clamp(positional / POSITION_SCALE, -1.0, 1.0),
    )


class GeneticPlayer:
    """One-ply player driven by an evolved, phase-aware linear evaluator."""

    def __init__(
        self,
        genome: Sequence[float],
        name: str = "Genetic",
    ) -> None:
        if len(genome) != GENOME_SIZE:
            raise ValueError(
                f"Genetic genome requires {GENOME_SIZE} values; received {len(genome)}"
            )
        self.genome = tuple(float(value) for value in genome)
        self.name = name

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> GeneticPlayer:
        checkpoint_path = Path(path).expanduser().resolve()
        payload = load_checkpoint(checkpoint_path)
        best = payload["best_ever"]
        generation = int(payload["generation"])
        fitness = float(best.get("fitness", 0.0))
        return cls(
            best["genome"],
            name=(
                f"Genetic (generation {generation}, fitness {fitness:.3f}, "
                f"{checkpoint_path.name})"
            ),
        )

    def evaluate(self, game: HeadlessOthello, color: int) -> float:
        other_moves = game.legal_moves(opponent(color))
        if not other_moves and not game.legal_moves(color):
            scores = game.score()
            difference = scores[color] - scores[opponent(color)]
            if difference > 0:
                return 1_000.0 + difference
            if difference < 0:
                return -1_000.0 + difference
            return 0.0

        scores = game.score()
        progress = (scores[BLACK] + scores[WHITE]) / (BOARD_SIZE * BOARD_SIZE)
        phase = progress * progress
        features = extract_features(game, color, len(other_moves))
        opening = self.genome[: len(FEATURE_NAMES)]
        endgame = self.genome[len(FEATURE_NAMES) :]
        weights = (
            (1.0 - phase) * opening[index] + phase * endgame[index]
            for index in range(len(FEATURE_NAMES))
        )
        return sum(weight * feature for weight, feature in zip(weights, features))

    def choose_move(
        self,
        game: HeadlessOthello,
        color: int,
        legal_moves: Sequence[LegalMove],
        rng: random.Random,
    ) -> tuple[int, int]:
        best_value = float("-inf")
        best_moves: list[LegalMove] = []

        for move in legal_moves:
            game.play(color, move)
            try:
                value = self.evaluate(game, color)
            finally:
                game.undo(color, move)

            if value > best_value + 1e-12:
                best_value = value
                best_moves = [move]
            elif abs(value - best_value) <= 1e-12:
                best_moves.append(move)

        selected = rng.choice(best_moves)
        return selected.x, selected.y


class RandomBaseline:
    name = "Random"

    def choose_move(
        self,
        game: HeadlessOthello,
        color: int,
        legal_moves: Sequence[LegalMove],
        rng: random.Random,
    ) -> tuple[int, int]:
        del game, color
        move = rng.choice(legal_moves)
        return move.x, move.y


class GreedyBaseline:
    name = "Greedy"

    def choose_move(
        self,
        game: HeadlessOthello,
        color: int,
        legal_moves: Sequence[LegalMove],
        rng: random.Random,
    ) -> tuple[int, int]:
        del game, color, rng
        move = max(legal_moves, key=lambda candidate: len(candidate.flips))
        return move.x, move.y


@dataclass
class Individual:
    genome: list[float]
    fitness: float = 0.0
    games: int = 0

    def copy(self) -> Individual:
        return Individual(self.genome.copy(), self.fitness, self.games)


@dataclass
class TrainingConfig:
    generations: int = 50
    population_size: int = 20
    games_per_pair: int = 2
    baseline_games: int = 2
    minimax_games: int = 2
    minimax_depth: int = 1
    elite_count: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.90
    mutation_rate: float = 0.20
    mutation_sigma: float = 0.20
    gene_limit: float = 3.0
    margin_weight: float = 0.05
    checkpoint_every: int = 5
    output_directory: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIRECTORY)
    seed: int = 0
    resume: Path | None = None

    def validate(self) -> None:
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if self.population_size < 4:
            raise ValueError("population size must be at least 4")
        if self.games_per_pair < 1:
            raise ValueError("games per pair must be at least 1")
        if self.baseline_games < 0 or self.minimax_games < 0:
            raise ValueError("baseline game counts cannot be negative")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("elite count must be between 1 and population size - 1")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament size must be between 2 and population size")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover rate must be between 0 and 1")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation rate must be between 0 and 1")
        if self.mutation_sigma < 0.0:
            raise ValueError("mutation sigma cannot be negative")
        if self.gene_limit <= 0.0:
            raise ValueError("gene limit must be positive")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint interval must be at least 1")
        if self.minimax_games and self.minimax_depth < 1:
            raise ValueError("minimax depth must be at least 1")


def _result_points(outcome: GameOutcome, color: int, margin_weight: float) -> float:
    if outcome.winner is None:
        outcome_points = 0.5
    else:
        outcome_points = 1.0 if outcome.winner == color else 0.0
    own_score = outcome.black_score if color == BLACK else outcome.white_score
    other_score = outcome.white_score if color == BLACK else outcome.black_score
    return outcome_points + margin_weight * (own_score - other_score) / 64.0


def evaluate_population(
    population: Sequence[Individual],
    config: TrainingConfig,
    rng: random.Random,
) -> None:
    """Score every genome through co-evolution and fixed baseline games."""
    players = [GeneticPlayer(individual.genome, f"Genome {index}") for index, individual in enumerate(population)]
    points = [0.0 for _ in population]
    games = [0 for _ in population]

    for first in range(len(population)):
        for second in range(first + 1, len(population)):
            for game_index in range(config.games_per_pair):
                if game_index % 2 == 0:
                    outcome = play_game(players[first], players[second], rng)
                    colors = (BLACK, WHITE)
                else:
                    outcome = play_game(players[second], players[first], rng)
                    colors = (WHITE, BLACK)

                points[first] += _result_points(outcome, colors[0], config.margin_weight)
                points[second] += _result_points(outcome, colors[1], config.margin_weight)
                games[first] += 1
                games[second] += 1

    baselines: list[tuple[Player, int]] = [
        (RandomBaseline(), config.baseline_games),
        (GreedyBaseline(), config.baseline_games),
    ]
    if config.minimax_games:
        baselines.append((MinimaxPlayer(config.minimax_depth), config.minimax_games))

    for index, player in enumerate(players):
        for baseline, game_count in baselines:
            for game_index in range(game_count):
                if game_index % 2 == 0:
                    outcome = play_game(player, baseline, rng)
                    color = BLACK
                else:
                    outcome = play_game(baseline, player, rng)
                    color = WHITE
                points[index] += _result_points(outcome, color, config.margin_weight)
                games[index] += 1

    for index, individual in enumerate(population):
        individual.games = games[index]
        individual.fitness = points[index] / max(1, games[index])


def _tournament_select(
    population: Sequence[Individual],
    tournament_size: int,
    rng: random.Random,
) -> Individual:
    contenders = rng.sample(list(population), tournament_size)
    return max(contenders, key=lambda individual: individual.fitness)


def _crossover(
    first: Sequence[float],
    second: Sequence[float],
    rng: random.Random,
    gene_limit: float,
) -> list[float]:
    child: list[float] = []
    for first_gene, second_gene in zip(first, second):
        blend = rng.uniform(-0.25, 1.25)
        value = blend * first_gene + (1.0 - blend) * second_gene
        child.append(_clamp(value, -gene_limit, gene_limit))
    return child


def _mutate(genome: list[float], config: TrainingConfig, rng: random.Random) -> None:
    for index in range(len(genome)):
        if rng.random() < config.mutation_rate:
            genome[index] = _clamp(
                genome[index] + rng.gauss(0.0, config.mutation_sigma),
                -config.gene_limit,
                config.gene_limit,
            )


def reproduce(
    population: Sequence[Individual],
    config: TrainingConfig,
    rng: random.Random,
) -> list[Individual]:
    ranked = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    next_population = [ranked[index].copy() for index in range(config.elite_count)]
    for elite in next_population:
        elite.fitness = 0.0
        elite.games = 0

    while len(next_population) < config.population_size:
        first = _tournament_select(ranked, config.tournament_size, rng)
        second = _tournament_select(ranked, config.tournament_size, rng)
        if rng.random() < config.crossover_rate:
            genome = _crossover(first.genome, second.genome, rng, config.gene_limit)
        else:
            genome = first.genome.copy()
        _mutate(genome, config, rng)
        next_population.append(Individual(genome))
    return next_population


def create_population(config: TrainingConfig, rng: random.Random) -> list[Individual]:
    population = [Individual(list(DEFAULT_SEED_GENOME))]
    while len(population) < config.population_size:
        population.append(
            Individual(
                [
                    rng.uniform(-config.gene_limit / 2, config.gene_limit / 2)
                    for _ in range(GENOME_SIZE)
                ]
            )
        )
    return population


def _individual_payload(individual: Individual) -> dict[str, Any]:
    return {
        "genome": individual.genome,
        "fitness": individual.fitness,
        "games": individual.games,
    }


def _config_payload(config: TrainingConfig) -> dict[str, Any]:
    return {
        "generations": config.generations,
        "population_size": config.population_size,
        "games_per_pair": config.games_per_pair,
        "baseline_games": config.baseline_games,
        "minimax_games": config.minimax_games,
        "minimax_depth": config.minimax_depth,
        "elite_count": config.elite_count,
        "tournament_size": config.tournament_size,
        "crossover_rate": config.crossover_rate,
        "mutation_rate": config.mutation_rate,
        "mutation_sigma": config.mutation_sigma,
        "gene_limit": config.gene_limit,
        "margin_weight": config.margin_weight,
        "checkpoint_every": config.checkpoint_every,
        "seed": config.seed,
    }


def save_checkpoint(
    path: Path,
    generation: int,
    population: Sequence[Individual],
    generation_best: Individual,
    best_ever: Individual,
    config: TrainingConfig,
) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "generation": generation,
        "feature_names": FEATURE_NAMES,
        "genome_layout": "opening weights followed by endgame weights",
        "generation_best": _individual_payload(generation_best),
        "best_ever": _individual_payload(best_ever),
        "population": [_individual_payload(individual) for individual in population],
        "config": _config_payload(config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read genetic checkpoint {checkpoint_path}: {exc}") from exc

    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Not a COSMOS genetic checkpoint: {checkpoint_path}")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported genetic checkpoint version {payload.get('version')!r}"
        )
    best = payload.get("best_ever", {})
    if len(best.get("genome", [])) != GENOME_SIZE:
        raise ValueError(f"Checkpoint contains an invalid genome: {checkpoint_path}")
    return payload


def _restore_population(payload: dict[str, Any]) -> list[Individual]:
    population = []
    for stored in payload.get("population", []):
        genome = stored.get("genome", [])
        if len(genome) != GENOME_SIZE:
            raise ValueError("Checkpoint population contains an invalid genome")
        population.append(
            Individual(
                [float(value) for value in genome],
                float(stored.get("fitness", 0.0)),
                int(stored.get("games", 0)),
            )
        )
    if len(population) < 2:
        raise ValueError("Checkpoint does not contain a usable population")
    return population


def train(config: TrainingConfig) -> Path:
    """Run evolution and return the final checkpoint path."""
    config.validate()
    rng = random.Random(config.seed)
    config.output_directory = config.output_directory.expanduser().resolve()
    config.output_directory.mkdir(parents=True, exist_ok=True)

    best_ever: Individual | None = None
    if config.resume is not None:
        resume_payload = load_checkpoint(config.resume)
        evaluated_population = _restore_population(resume_payload)
        stored_best = resume_payload["best_ever"]
        best_ever = Individual(
            [float(value) for value in stored_best["genome"]],
            float(stored_best.get("fitness", 0.0)),
            int(stored_best.get("games", 0)),
        )
        start_generation = int(resume_payload["generation"]) + 1
        if start_generation >= config.generations:
            return Path(config.resume).expanduser().resolve()
        population = reproduce(evaluated_population, config, rng)
        print(
            f"Resuming after generation {start_generation} from "
            f"{Path(config.resume).expanduser().resolve()}"
        )
    else:
        population = create_population(config, rng)
        start_generation = 0

    final_checkpoint: Path | None = None
    training_started = time.perf_counter()

    for generation in range(start_generation, config.generations):
        generation_started = time.perf_counter()
        evaluate_population(population, config, rng)
        ranked = sorted(population, key=lambda individual: individual.fitness, reverse=True)
        generation_best = ranked[0].copy()
        if best_ever is None or generation_best.fitness > best_ever.fitness:
            best_ever = generation_best.copy()

        fitnesses = [individual.fitness for individual in population]
        print(
            f"Generation {generation + 1}/{config.generations}: "
            f"best={generation_best.fitness:.4f}, "
            f"mean={statistics.mean(fitnesses):.4f}, "
            f"median={statistics.median(fitnesses):.4f}, "
            f"games/genome={generation_best.games}, "
            f"time={time.perf_counter() - generation_started:.2f}s"
        )

        should_save = (
            generation == start_generation
            or (generation + 1) % config.checkpoint_every == 0
            or generation == config.generations - 1
        )
        if should_save:
            checkpoint = config.output_directory / f"genetic_gen_{generation:04d}.json"
            save_checkpoint(
                checkpoint,
                generation,
                population,
                generation_best,
                best_ever,
                config,
            )
            save_checkpoint(
                config.output_directory / "latest.json",
                generation,
                population,
                generation_best,
                best_ever,
                config,
            )
            final_checkpoint = checkpoint
            print(f"  Saved checkpoint: {checkpoint}")

        if generation < config.generations - 1:
            population = reproduce(population, config, rng)

    assert final_checkpoint is not None
    print(
        f"Training complete in {time.perf_counter() - training_started:.2f}s. "
        f"Best fitness: {best_ever.fitness:.4f}."
    )
    return final_checkpoint


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolve a phase-aware Othello evaluation function.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--games-per-pair", type=int, default=2)
    parser.add_argument("--baseline-games", type=int, default=2)
    parser.add_argument("--minimax-games", type=int, default=2)
    parser.add_argument("--minimax-depth", type=int, default=1)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--crossover-rate", type=float, default=0.90)
    parser.add_argument("--mutation-rate", type=float, default=0.20)
    parser.add_argument("--mutation-sigma", type=float, default=0.20)
    parser.add_argument("--gene-limit", type=float, default=3.0)
    parser.add_argument("--margin-weight", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "Resume from a checkpoint. --generations remains the total target "
            "generation count, not an additional count."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = TrainingConfig(
        generations=args.generations,
        population_size=args.population,
        games_per_pair=args.games_per_pair,
        baseline_games=args.baseline_games,
        minimax_games=args.minimax_games,
        minimax_depth=args.minimax_depth,
        elite_count=args.elite_count,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        mutation_sigma=args.mutation_sigma,
        gene_limit=args.gene_limit,
        margin_weight=args.margin_weight,
        checkpoint_every=args.checkpoint_every,
        output_directory=args.output_directory,
        seed=args.seed,
        resume=args.resume,
    )
    try:
        train(config)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
