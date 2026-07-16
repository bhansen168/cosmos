#!/usr/bin/env python3
"""Train and load an evolutionary Othello evaluation model.

The current genome contains opening, middlegame, and endgame weights for ten
normalized board features. Training screens the full population with a fast
search, then uses full-depth alpha-beta against a fixed validation suite to
choose the checkpoint champion. Games combine sampled co-evolution with paired
randomized openings against fixed baselines.

Version-1 checkpoints containing the original twelve-gene, one-ply evaluator
are upgraded in memory when loaded or resumed. Existing checkpoint files are
never modified.
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

from computer import Computer as GreedyBaseline
from computer import RandomComputer as RandomBaseline
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
)


CHECKPOINT_FORMAT = "cosmos-genetic-othello"
CHECKPOINT_VERSION = 2
SUPPORTED_CHECKPOINT_VERSIONS = (1, CHECKPOINT_VERSION)
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "models" / "genetic"
DEFAULT_SEARCH_DEPTH = 2
PHASE_NAMES = ("opening", "middlegame", "endgame")
FEATURE_NAMES = (
    "disc_difference",
    "mobility_difference",
    "potential_mobility",
    "corners",
    "corner_closeness",
    "edges",
    "stable_edges",
    "frontier_safety",
    "positional_value",
    "forced_pass",
)
GENOME_SIZE = len(FEATURE_NAMES) * len(PHASE_NAMES)

LEGACY_FEATURE_NAMES = (
    "disc_difference",
    "opponent_mobility",
    "corners",
    "edges",
    "frontier_safety",
    "positional_value",
)
LEGACY_GENOME_SIZE = len(LEGACY_FEATURE_NAMES) * 2

CORNER_COORDINATES = ((0, 0), (7, 0), (0, 7), (7, 7))
CORNER_NEIGHBORS = {
    (0, 0): ((1, 0), (0, 1), (1, 1)),
    (7, 0): ((6, 0), (7, 1), (6, 1)),
    (0, 7): ((1, 7), (0, 6), (1, 6)),
    (7, 7): ((6, 7), (7, 6), (6, 6)),
}
CORNER_EDGE_DIRECTIONS = {
    (0, 0): ((1, 0), (0, 1)),
    (7, 0): ((-1, 0), (0, 1)),
    (0, 7): ((1, 0), (0, -1)),
    (7, 7): ((-1, 0), (0, -1)),
}
EDGE_COORDINATES = tuple(
    (x, y)
    for y in range(BOARD_SIZE)
    for x in range(BOARD_SIZE)
    if (x in (0, 7) or y in (0, 7)) and (x, y) not in CORNER_COORDINATES
)
POSITION_SCALE = float(sum(abs(value) for row in POSITION_WEIGHTS for value in row))

# An informed seed gives evolution a useful starting point while random
# immigrants continue exploring unrelated strategies.
DEFAULT_SEED_GENOME = (
    # Opening
    -0.25,
    1.30,
    0.60,
    2.50,
    1.40,
    0.15,
    1.00,
    0.90,
    1.20,
    0.80,
    # Middlegame
    0.20,
    1.20,
    0.40,
    2.70,
    1.00,
    0.35,
    1.40,
    0.70,
    1.10,
    1.00,
    # Endgame
    2.00,
    0.40,
    0.10,
    2.00,
    0.40,
    0.70,
    2.00,
    0.25,
    0.70,
    1.50,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _upgrade_legacy_genome(genome: Sequence[float]) -> list[float]:
    """Map a version-1 two-phase genome onto the richer version-2 layout."""
    if len(genome) == GENOME_SIZE:
        return [float(value) for value in genome]
    if len(genome) != LEGACY_GENOME_SIZE:
        raise ValueError(
            f"Genetic genome requires {GENOME_SIZE} values; received {len(genome)}"
        )

    old_opening = [float(value) for value in genome[: len(LEGACY_FEATURE_NAMES)]]
    old_endgame = [float(value) for value in genome[len(LEGACY_FEATURE_NAMES) :]]
    old_middle = [
        (opening + endgame) / 2.0
        for opening, endgame in zip(old_opening, old_endgame)
    ]
    old_phases = (old_opening, old_middle, old_endgame)
    upgraded = list(DEFAULT_SEED_GENOME)
    feature_mapping = {
        "disc_difference": "disc_difference",
        "opponent_mobility": "mobility_difference",
        "corners": "corners",
        "edges": "edges",
        "frontier_safety": "frontier_safety",
        "positional_value": "positional_value",
    }

    for phase_index, old_phase in enumerate(old_phases):
        phase_offset = phase_index * len(FEATURE_NAMES)
        for old_index, old_name in enumerate(LEGACY_FEATURE_NAMES):
            new_index = FEATURE_NAMES.index(feature_mapping[old_name])
            upgraded[phase_offset + new_index] = old_phase[old_index]
    return upgraded


def _stable_edge_counts(game: HeadlessOthello, color: int) -> tuple[int, int]:
    """Count edge discs connected to an occupied corner.

    This is a conservative stability approximation: every counted disc is
    anchored to a corner, although it does not attempt full interior stability.
    """
    stable: dict[int, set[tuple[int, int]]] = {BLACK: set(), WHITE: set()}
    for corner, directions in CORNER_EDGE_DIRECTIONS.items():
        corner_x, corner_y = corner
        corner_color = game.board[corner_y][corner_x]
        if corner_color == EMPTY:
            continue
        for dx, dy in directions:
            x, y = corner_x, corner_y
            while (
                0 <= x < BOARD_SIZE
                and 0 <= y < BOARD_SIZE
                and game.board[y][x] == corner_color
            ):
                stable[corner_color].add((x, y))
                x += dx
                y += dy
    return len(stable[color]), len(stable[opponent(color)])


def extract_features(
    game: HeadlessOthello,
    color: int,
    opponent_move_count: int | None = None,
    own_move_count: int | None = None,
) -> tuple[float, ...]:
    """Return ten normalized features from ``color``'s perspective."""
    if own_move_count is None:
        own_move_count = len(game.legal_moves(color))
    if opponent_move_count is None:
        opponent_move_count = len(game.legal_moves(opponent(color)))
    features, _, _ = _analyze_board(
        game,
        color,
        own_move_count,
        opponent_move_count,
    )
    return features


def _analyze_board(
    game: HeadlessOthello,
    color: int,
    own_move_count: int,
    opponent_move_count: int,
) -> tuple[tuple[float, ...], float, int]:
    """Extract all board features in one pass.

    The previous evaluator scanned the board five times and generated the same
    legal moves repeatedly at each search leaf. Returning progress and disc
    difference with the features avoids those duplicate operations.
    """
    other = opponent(color)
    board = game.board
    own_discs = 0
    other_discs = 0
    empty_count = 0
    own_corners = 0
    other_corners = 0
    own_edges = 0
    other_edges = 0
    own_frontier = 0
    other_frontier = 0
    own_potential = 0
    other_potential = 0
    positional = 0

    for y, row in enumerate(board):
        for x, square in enumerate(row):
            is_corner = x in (0, 7) and y in (0, 7)
            is_edge = (x in (0, 7) or y in (0, 7)) and not is_corner
            if square == EMPTY:
                empty_count += 1
                touches_own = False
                touches_other = False
                for dx, dy in DIRECTIONS:
                    neighbor_x = x + dx
                    neighbor_y = y + dy
                    if not (
                        0 <= neighbor_x < BOARD_SIZE
                        and 0 <= neighbor_y < BOARD_SIZE
                    ):
                        continue
                    neighbor = board[neighbor_y][neighbor_x]
                    touches_own = touches_own or neighbor == color
                    touches_other = touches_other or neighbor == other
                if touches_other:
                    own_potential += 1
                if touches_own:
                    other_potential += 1
                continue

            touches_empty = False
            for dx, dy in DIRECTIONS:
                neighbor_x = x + dx
                neighbor_y = y + dy
                if (
                    0 <= neighbor_x < BOARD_SIZE
                    and 0 <= neighbor_y < BOARD_SIZE
                    and board[neighbor_y][neighbor_x] == EMPTY
                ):
                    touches_empty = True
                    break

            if square == color:
                own_discs += 1
                positional += POSITION_WEIGHTS[y][x]
                own_corners += int(is_corner)
                own_edges += int(is_edge)
                own_frontier += int(touches_empty)
            elif square == other:
                other_discs += 1
                positional -= POSITION_WEIGHTS[y][x]
                other_corners += int(is_corner)
                other_edges += int(is_edge)
                other_frontier += int(touches_empty)

    total_frontier = own_frontier + other_frontier
    own_stable, other_stable = _stable_edge_counts(game, color)

    own_corner_neighbors = 0
    other_corner_neighbors = 0
    for corner, neighbors in CORNER_NEIGHBORS.items():
        corner_x, corner_y = corner
        if board[corner_y][corner_x] != EMPTY:
            continue
        own_corner_neighbors += sum(board[y][x] == color for x, y in neighbors)
        other_corner_neighbors += sum(board[y][x] == other for x, y in neighbors)

    mobility_total = own_move_count + opponent_move_count
    forced_pass = 0.0
    if opponent_move_count == 0 and own_move_count > 0:
        forced_pass = 1.0
    elif own_move_count == 0 and opponent_move_count > 0:
        forced_pass = -1.0

    features = (
        (own_discs - other_discs) / (BOARD_SIZE * BOARD_SIZE),
        (own_move_count - opponent_move_count) / max(1, mobility_total),
        (own_potential - other_potential) / max(1, empty_count),
        (own_corners - other_corners) / len(CORNER_COORDINATES),
        (other_corner_neighbors - own_corner_neighbors) / 12.0,
        (own_edges - other_edges) / len(EDGE_COORDINATES),
        (own_stable - other_stable) / (len(EDGE_COORDINATES) + 4),
        (other_frontier - own_frontier) / max(1, total_frontier),
        _clamp(positional / POSITION_SCALE, -1.0, 1.0),
        forced_pass,
    )
    progress = (own_discs + other_discs) / (BOARD_SIZE * BOARD_SIZE)
    return features, progress, own_discs - other_discs


class GeneticPlayer:
    """Alpha-beta player whose phase-aware evaluator is genetically evolved."""

    WIN_SCORE = 10_000.0

    def __init__(
        self,
        genome: Sequence[float],
        name: str = "Genetic",
        search_depth: int = DEFAULT_SEARCH_DEPTH,
    ) -> None:
        if search_depth < 1:
            raise ValueError("Genetic search depth must be at least 1")
        self.genome = tuple(_upgrade_legacy_genome(genome))
        self.search_depth = search_depth
        self.name = name

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> GeneticPlayer:
        checkpoint_path = Path(path).expanduser().resolve()
        payload = load_checkpoint(checkpoint_path)
        champion = payload.get("champion") or payload.get("generation_best")
        if not champion:
            champion = payload["best_ever"]
        generation = int(payload["generation"])
        search_depth = int(
            payload.get("config", {}).get("search_depth", DEFAULT_SEARCH_DEPTH)
        )
        validation_score = champion.get("validation_score")
        if validation_score is None:
            quality = f"fitness {float(champion.get('fitness', 0.0)):.3f}"
        else:
            quality = f"validation {float(validation_score):.3f}"
        return cls(
            champion["genome"],
            name=(
                f"Genetic (generation {generation}, {quality}, "
                f"depth {search_depth}, {checkpoint_path.name})"
            ),
            search_depth=search_depth,
        )

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

    def _weights_for_progress(self, progress: float) -> tuple[float, ...]:
        feature_count = len(FEATURE_NAMES)
        opening = self.genome[:feature_count]
        middle = self.genome[feature_count : feature_count * 2]
        endgame = self.genome[feature_count * 2 :]
        if progress <= 0.5:
            amount = progress * 2.0
            first, second = opening, middle
        else:
            amount = (progress - 0.5) * 2.0
            first, second = middle, endgame
        return tuple(
            (1.0 - amount) * first[index] + amount * second[index]
            for index in range(feature_count)
        )

    def evaluate(
        self,
        game: HeadlessOthello,
        color: int,
        legal_move_counts: dict[int, int] | None = None,
    ) -> float:
        other = opponent(color)
        if legal_move_counts is None:
            own_move_count = len(game.legal_moves(color))
            opponent_move_count = len(game.legal_moves(other))
        else:
            own_move_count = legal_move_counts[color]
            opponent_move_count = legal_move_counts[other]
        features, progress, difference = _analyze_board(
            game,
            color,
            own_move_count,
            opponent_move_count,
        )
        if opponent_move_count == 0 and own_move_count == 0:
            if difference > 0:
                return self.WIN_SCORE + difference
            if difference < 0:
                return -self.WIN_SCORE + difference
            return 0.0

        weights = self._weights_for_progress(progress)
        return sum(weight * feature for weight, feature in zip(weights, features))

    def _alpha_beta(
        self,
        game: HeadlessOthello,
        color: int,
        depth: int,
        alpha: float,
        beta: float,
        root_color: int,
    ) -> float:
        legal_moves = game.legal_moves(color)
        other_color = opponent(color)
        if not legal_moves:
            other_moves = game.legal_moves(other_color)
            move_counts = {color: 0, other_color: len(other_moves)}
            if not other_moves:
                return self.evaluate(game, root_color, move_counts)
            if depth <= 0:
                return self.evaluate(game, root_color, move_counts)
            return self._alpha_beta(
                game,
                other_color,
                depth,
                alpha,
                beta,
                root_color,
            )

        if depth <= 0:
            other_move_count = len(game.legal_moves(other_color))
            return self.evaluate(
                game,
                root_color,
                {color: len(legal_moves), other_color: other_move_count},
            )

        if color == root_color:
            value = float("-inf")
            for move in self._ordered_moves(legal_moves):
                game.play(color, move)
                try:
                    child = self._alpha_beta(
                        game,
                        other_color,
                        depth - 1,
                        alpha,
                        beta,
                        root_color,
                    )
                finally:
                    game.undo(color, move)
                value = max(value, child)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = float("inf")
        for move in self._ordered_moves(legal_moves):
            game.play(color, move)
            try:
                child = self._alpha_beta(
                    game,
                    other_color,
                    depth - 1,
                    alpha,
                    beta,
                    root_color,
                )
            finally:
                game.undo(color, move)
            value = min(value, child)
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
        best_value = float("-inf")
        best_moves: list[LegalMove] = []
        alpha = float("-inf")

        for move in self._ordered_moves(legal_moves):
            game.play(color, move)
            try:
                value = self._alpha_beta(
                    game,
                    opponent(color),
                    self.search_depth - 1,
                    alpha,
                    float("inf"),
                    color,
                )
            finally:
                game.undo(color, move)

            if value > best_value + 1e-12:
                best_value = value
                best_moves = [move]
            elif abs(value - best_value) <= 1e-12:
                best_moves.append(move)
            alpha = max(alpha, best_value)

        selected = rng.choice(best_moves)
        return selected.x, selected.y


@dataclass
class Individual:
    genome: list[float]
    fitness: float = 0.0
    games: int = 0
    validation_score: float | None = None
    validation_games: int = 0
    validation_breakdown: dict[str, Any] = field(default_factory=dict)
    origin_generation: int | None = None

    def copy(self) -> Individual:
        return Individual(
            self.genome.copy(),
            self.fitness,
            self.games,
            self.validation_score,
            self.validation_games,
            dict(self.validation_breakdown),
            self.origin_generation,
        )


@dataclass
class TrainingConfig:
    generations: int = 50
    population_size: int = 30
    games_per_pair: int = 1
    coevolution_opponents: int = 6
    baseline_games: int = 1
    minimax_games: int = 2
    minimax_depth: int = 2
    minimax_weight: float = 3.0
    training_search_depth: int = 1
    search_depth: int = DEFAULT_SEARCH_DEPTH
    opening_plies: int = 10
    validation_candidates: int = 4
    validation_openings: int = 2
    validation_seed: int = 10_000
    validation_every: int = 2
    elite_count: int = 3
    tournament_size: int = 4
    crossover_rate: float = 0.90
    mutation_rate: float = 0.25
    mutation_sigma: float = 0.25
    gene_limit: float = 4.0
    margin_weight: float = 0.25
    random_immigrants: int = 2
    stagnation_generations: int = 6
    mutation_boost: float = 2.0
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
        if self.coevolution_opponents < 1:
            raise ValueError("co-evolution opponents must be at least 1")
        if self.baseline_games < 0 or self.minimax_games < 0:
            raise ValueError("baseline game counts cannot be negative")
        if (
            self.minimax_depth < 1
            or self.training_search_depth < 1
            or self.search_depth < 1
        ):
            raise ValueError("search depths must be at least 1")
        if self.minimax_weight <= 0.0:
            raise ValueError("minimax fitness weight must be positive")
        if self.opening_plies < 0:
            raise ValueError("opening plies cannot be negative")
        if not 1 <= self.validation_candidates <= self.population_size:
            raise ValueError(
                "validation candidates must be between 1 and population size"
            )
        if self.validation_openings < 1:
            raise ValueError("validation openings must be at least 1")
        if self.validation_every < 1:
            raise ValueError("validation interval must be at least 1")
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
        if self.margin_weight < 0.0:
            raise ValueError("margin weight cannot be negative")
        if self.random_immigrants < 0:
            raise ValueError("random immigrants cannot be negative")
        if self.elite_count + self.random_immigrants >= self.population_size:
            raise ValueError("elites and immigrants must leave room for offspring")
        if self.stagnation_generations < 1:
            raise ValueError("stagnation generations must be at least 1")
        if self.mutation_boost < 1.0:
            raise ValueError("mutation boost must be at least 1")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint interval must be at least 1")


@dataclass(frozen=True)
class MatchScenario:
    game: HeadlessOthello
    current_color: int
    black_seed: int
    white_seed: int


def _random_opening(
    rng: random.Random,
    maximum_plies: int,
) -> tuple[HeadlessOthello, int]:
    game = HeadlessOthello()
    color = BLACK
    target_plies = rng.randint(0, maximum_plies) if maximum_plies else 0
    moves_played = 0
    consecutive_passes = 0

    while moves_played < target_plies and consecutive_passes < 2:
        legal_moves = game.legal_moves(color)
        if not legal_moves:
            consecutive_passes += 1
            color = opponent(color)
            continue
        consecutive_passes = 0
        game.play(color, rng.choice(legal_moves))
        moves_played += 1
        color = opponent(color)
    return game, color


def _make_scenario(rng: random.Random, opening_plies: int) -> MatchScenario:
    game, current_color = _random_opening(rng, opening_plies)
    return MatchScenario(
        game=game,
        current_color=current_color,
        black_seed=rng.randrange(0, 2**63),
        white_seed=rng.randrange(0, 2**63),
    )


def _play_from_scenario(
    black_player: Player,
    white_player: Player,
    scenario: MatchScenario,
) -> GameOutcome:
    game = scenario.game.clone()
    players = {BLACK: black_player, WHITE: white_player}
    player_rngs = {
        BLACK: random.Random(scenario.black_seed),
        WHITE: random.Random(scenario.white_seed),
    }
    color = scenario.current_color
    consecutive_passes = 0
    moves_played = 0

    while consecutive_passes < 2:
        legal_moves = game.legal_moves(color)
        if not legal_moves:
            consecutive_passes += 1
            color = opponent(color)
            continue

        consecutive_passes = 0
        coordinate = players[color].choose_move(
            game,
            color,
            legal_moves,
            player_rngs[color],
        )
        legal_by_coordinate = {(move.x, move.y): move for move in legal_moves}
        if coordinate not in legal_by_coordinate:
            raise RuntimeError(
                f"{players[color].name} selected illegal move {coordinate}"
            )
        game.play(color, legal_by_coordinate[coordinate])
        moves_played += 1
        color = opponent(color)

    scores = game.score()
    if scores[BLACK] == scores[WHITE]:
        winner = None
    else:
        winner = BLACK if scores[BLACK] > scores[WHITE] else WHITE
    return GameOutcome(
        black_score=scores[BLACK],
        white_score=scores[WHITE],
        winner=winner,
        moves=moves_played,
    )


def _play_color_pair(
    first: Player,
    second: Player,
    scenario: MatchScenario,
) -> tuple[GameOutcome, GameOutcome]:
    """Play the same position twice, swapping the two players' colors."""
    return (
        _play_from_scenario(first, second, scenario),
        _play_from_scenario(second, first, scenario),
    )


def _result_points(outcome: GameOutcome, color: int, margin_weight: float) -> float:
    if outcome.winner is None:
        outcome_points = 0.5
    else:
        outcome_points = 1.0 if outcome.winner == color else 0.0
    own_score = outcome.black_score if color == BLACK else outcome.white_score
    other_score = outcome.white_score if color == BLACK else outcome.black_score
    return outcome_points + margin_weight * (own_score - other_score) / 64.0


def _round_robin_pairs(
    population_size: int,
    opponent_count: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Return sampled round-robin rounds with equal opponent counts."""
    participants: list[int | None] = list(range(population_size))
    rng.shuffle(participants)
    if len(participants) % 2:
        participants.append(None)
    rounds = min(opponent_count, population_size - 1)
    pairs: list[tuple[int, int]] = []

    for _ in range(rounds):
        half = len(participants) // 2
        for index in range(half):
            first = participants[index]
            second = participants[-1 - index]
            if first is not None and second is not None:
                pairs.append((first, second))
        participants = [
            participants[0],
            participants[-1],
            *participants[1:-1],
        ]
    return pairs


def evaluate_population(
    population: Sequence[Individual],
    config: TrainingConfig,
    rng: random.Random,
) -> None:
    """Score genomes with sampled co-evolution and weighted fixed opponents."""
    players = [
        GeneticPlayer(
            individual.genome,
            f"Genome {index}",
            search_depth=config.training_search_depth,
        )
        for index, individual in enumerate(population)
    ]
    points = [0.0 for _ in population]
    weights = [0.0 for _ in population]
    games = [0 for _ in population]

    def add_result(
        index: int,
        outcome: GameOutcome,
        color: int,
        weight: float,
    ) -> None:
        points[index] += weight * _result_points(
            outcome,
            color,
            config.margin_weight,
        )
        weights[index] += weight
        games[index] += 1

    pairs = _round_robin_pairs(
        len(population),
        config.coevolution_opponents,
        rng,
    )
    for first, second in pairs:
        for _ in range(config.games_per_pair):
            scenario = _make_scenario(rng, config.opening_plies)
            first_black, second_black = _play_color_pair(
                players[first],
                players[second],
                scenario,
            )
            add_result(first, first_black, BLACK, 1.0)
            add_result(second, first_black, WHITE, 1.0)
            add_result(first, second_black, WHITE, 1.0)
            add_result(second, second_black, BLACK, 1.0)

    baselines: list[tuple[Player, int, float]] = [
        (RandomBaseline(), config.baseline_games, 1.0),
        (GreedyBaseline(), config.baseline_games, 1.0),
    ]
    if config.minimax_games:
        baselines.append(
            (
                MinimaxPlayer(config.minimax_depth),
                config.minimax_games,
                config.minimax_weight,
            )
        )

    scenario_sets = [
        [_make_scenario(rng, config.opening_plies) for _ in range(opening_count)]
        for _, opening_count, _ in baselines
    ]
    for index, player in enumerate(players):
        for (baseline, _, weight), scenarios in zip(baselines, scenario_sets):
            for scenario in scenarios:
                player_black, baseline_black = _play_color_pair(
                    player,
                    baseline,
                    scenario,
                )
                add_result(index, player_black, BLACK, weight)
                add_result(index, baseline_black, WHITE, weight)

    for index, individual in enumerate(population):
        individual.games = games[index]
        individual.fitness = points[index] / max(1.0, weights[index])
        individual.validation_score = None
        individual.validation_games = 0
        individual.validation_breakdown = {}


def _validation_opponents(config: TrainingConfig) -> list[tuple[str, Player, float]]:
    opponents: list[tuple[str, Player, float]] = [
        ("random", RandomBaseline(), 0.5),
        ("greedy", GreedyBaseline(), 1.0),
        ("minimax_depth_1", MinimaxPlayer(1), 1.5),
    ]
    target_name = f"minimax_depth_{config.minimax_depth}"
    if config.minimax_depth == 1:
        opponents[-1] = (target_name, MinimaxPlayer(1), config.minimax_weight)
    else:
        opponents.append(
            (
                target_name,
                MinimaxPlayer(config.minimax_depth),
                config.minimax_weight,
            )
        )
    return opponents


def validate_candidates(
    candidates: Sequence[Individual],
    config: TrainingConfig,
    generation: int,
) -> Individual:
    """Choose a champion on a fixed suite comparable across generations."""
    validation_rng = random.Random(config.validation_seed)
    scenarios = [
        _make_scenario(validation_rng, config.opening_plies)
        for _ in range(config.validation_openings)
    ]
    opponents = _validation_opponents(config)
    validated: list[Individual] = []

    for candidate in candidates[: config.validation_candidates]:
        player = GeneticPlayer(
            candidate.genome,
            "Validation candidate",
            search_depth=config.search_depth,
        )
        weighted_points = 0.0
        total_weight = 0.0
        total_games = 0
        breakdown: dict[str, Any] = {}

        for name, baseline, weight in opponents:
            opponent_points = 0.0
            wins = 0
            draws = 0
            opponent_games = 0
            for scenario in scenarios:
                player_black, baseline_black = _play_color_pair(
                    player,
                    baseline,
                    scenario,
                )
                for outcome, color in (
                    (player_black, BLACK),
                    (baseline_black, WHITE),
                ):
                    score = _result_points(outcome, color, config.margin_weight)
                    opponent_points += score
                    weighted_points += weight * score
                    total_weight += weight
                    total_games += 1
                    opponent_games += 1
                    if outcome.winner is None:
                        draws += 1
                    elif outcome.winner == color:
                        wins += 1
            breakdown[name] = {
                "score": opponent_points / max(1, opponent_games),
                "win_rate": wins / max(1, opponent_games),
                "draws": draws,
                "games": opponent_games,
                "weight": weight,
            }

        evaluated = candidate.copy()
        evaluated.validation_score = weighted_points / max(1.0, total_weight)
        evaluated.validation_games = total_games
        evaluated.validation_breakdown = breakdown
        evaluated.origin_generation = generation
        validated.append(evaluated)

    return max(
        validated,
        key=lambda individual: (
            individual.validation_score
            if individual.validation_score is not None
            else float("-inf"),
            individual.fitness,
        ),
    )


def _tournament_select(
    population: Sequence[Individual],
    tournament_size: int,
    rng: random.Random,
) -> Individual:
    contenders = rng.sample(
        list(population),
        min(tournament_size, len(population)),
    )
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


def _mutate(
    genome: list[float],
    config: TrainingConfig,
    rng: random.Random,
    sigma_multiplier: float,
) -> None:
    for index in range(len(genome)):
        if rng.random() < config.mutation_rate:
            genome[index] = _clamp(
                genome[index]
                + rng.gauss(0.0, config.mutation_sigma * sigma_multiplier),
                -config.gene_limit,
                config.gene_limit,
            )


def _random_individual(config: TrainingConfig, rng: random.Random) -> Individual:
    return Individual(
        [
            rng.uniform(-config.gene_limit / 2, config.gene_limit / 2)
            for _ in range(GENOME_SIZE)
        ]
    )


def reproduce(
    population: Sequence[Individual],
    config: TrainingConfig,
    rng: random.Random,
    champion: Individual | None = None,
    sigma_multiplier: float = 1.0,
) -> list[Individual]:
    ranked = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    offspring_target = config.population_size - config.random_immigrants
    next_population = [
        Individual(ranked[index].genome.copy())
        for index in range(min(config.elite_count, len(ranked)))
    ]

    if champion is not None and len(next_population) < offspring_target:
        champion_genome = tuple(champion.genome)
        if all(tuple(individual.genome) != champion_genome for individual in next_population):
            next_population.append(Individual(champion.genome.copy()))

    while len(next_population) < offspring_target:
        first = _tournament_select(ranked, config.tournament_size, rng)
        second = _tournament_select(ranked, config.tournament_size, rng)
        if rng.random() < config.crossover_rate:
            genome = _crossover(first.genome, second.genome, rng, config.gene_limit)
        else:
            genome = first.genome.copy()
        _mutate(genome, config, rng, sigma_multiplier)
        next_population.append(Individual(genome))

    while len(next_population) < config.population_size:
        next_population.append(_random_individual(config, rng))
    return next_population


def create_population(config: TrainingConfig, rng: random.Random) -> list[Individual]:
    population = [Individual(list(DEFAULT_SEED_GENOME))]
    while len(population) < config.population_size:
        population.append(_random_individual(config, rng))
    return population


def _genome_diversity(population: Sequence[Individual]) -> float:
    if len(population) < 2:
        return 0.0
    deviations = []
    for gene_index in range(GENOME_SIZE):
        values = [individual.genome[gene_index] for individual in population]
        deviations.append(statistics.pstdev(values))
    return statistics.mean(deviations)


def _individual_payload(individual: Individual) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "genome": individual.genome,
        "fitness": individual.fitness,
        "games": individual.games,
    }
    if individual.validation_score is not None:
        payload.update(
            {
                "validation_score": individual.validation_score,
                "validation_games": individual.validation_games,
                "validation_breakdown": individual.validation_breakdown,
                "origin_generation": individual.origin_generation,
            }
        )
    return payload


def _individual_from_payload(payload: dict[str, Any]) -> Individual:
    genome = _upgrade_legacy_genome(payload.get("genome", []))
    validation_score = payload.get("validation_score")
    return Individual(
        genome=genome,
        fitness=float(payload.get("fitness", 0.0)),
        games=int(payload.get("games", 0)),
        validation_score=(
            None if validation_score is None else float(validation_score)
        ),
        validation_games=int(payload.get("validation_games", 0)),
        validation_breakdown=dict(payload.get("validation_breakdown", {})),
        origin_generation=payload.get("origin_generation"),
    )


def _config_payload(config: TrainingConfig) -> dict[str, Any]:
    return {
        "generations": config.generations,
        "population_size": config.population_size,
        "games_per_pair": config.games_per_pair,
        "coevolution_opponents": config.coevolution_opponents,
        "baseline_games": config.baseline_games,
        "minimax_games": config.minimax_games,
        "minimax_depth": config.minimax_depth,
        "minimax_weight": config.minimax_weight,
        "training_search_depth": config.training_search_depth,
        "search_depth": config.search_depth,
        "opening_plies": config.opening_plies,
        "validation_candidates": config.validation_candidates,
        "validation_openings": config.validation_openings,
        "validation_seed": config.validation_seed,
        "validation_every": config.validation_every,
        "elite_count": config.elite_count,
        "tournament_size": config.tournament_size,
        "crossover_rate": config.crossover_rate,
        "mutation_rate": config.mutation_rate,
        "mutation_sigma": config.mutation_sigma,
        "gene_limit": config.gene_limit,
        "margin_weight": config.margin_weight,
        "random_immigrants": config.random_immigrants,
        "stagnation_generations": config.stagnation_generations,
        "mutation_boost": config.mutation_boost,
        "checkpoint_every": config.checkpoint_every,
        "seed": config.seed,
    }


def save_checkpoint(
    path: Path,
    generation: int,
    population: Sequence[Individual],
    generation_best: Individual,
    champion: Individual,
    config: TrainingConfig,
    stagnation_count: int,
) -> None:
    champion_payload = _individual_payload(champion)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "generation": generation,
        "feature_names": FEATURE_NAMES,
        "phase_names": PHASE_NAMES,
        "genome_layout": "opening, middlegame, then endgame feature weights",
        "generation_best": _individual_payload(generation_best),
        "champion": champion_payload,
        # Compatibility alias for older consumers. Selection now uses champion.
        "best_ever": champion_payload,
        "population": [_individual_payload(individual) for individual in population],
        "config": _config_payload(config),
        "training_state": {"validation_stagnation": stagnation_count},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _upgrade_version_one_payload(payload: dict[str, Any]) -> dict[str, Any]:
    generation = int(payload.get("generation", 0))
    for key in ("generation_best", "best_ever"):
        if key in payload:
            payload[key] = _individual_payload(_individual_from_payload(payload[key]))
    payload["population"] = [
        _individual_payload(_individual_from_payload(stored))
        for stored in payload.get("population", [])
    ]

    # Raw co-evolution fitness from different generations is not comparable.
    # For a legacy file, prefer its current generation winner over stale
    # best_ever data until a version-2 validation champion is produced.
    champion = dict(payload.get("generation_best", payload.get("best_ever", {})))
    champion["origin_generation"] = generation
    payload["champion"] = champion
    payload["source_version"] = 1
    payload["version"] = CHECKPOINT_VERSION
    payload["feature_names"] = FEATURE_NAMES
    payload["phase_names"] = PHASE_NAMES
    payload["genome_layout"] = "opening, middlegame, then endgame feature weights"

    defaults = _config_payload(TrainingConfig())
    defaults.update(payload.get("config", {}))
    defaults["search_depth"] = DEFAULT_SEARCH_DEPTH
    payload["config"] = defaults
    payload.setdefault("training_state", {"validation_stagnation": 0})
    return payload


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read genetic checkpoint {checkpoint_path}: {exc}") from exc

    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Not a COSMOS genetic checkpoint: {checkpoint_path}")
    version = payload.get("version")
    if version not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(f"Unsupported genetic checkpoint version {version!r}")
    if version == 1:
        payload = _upgrade_version_one_payload(payload)

    for key in ("generation_best", "champion", "best_ever"):
        stored = payload.get(key)
        if stored and len(stored.get("genome", [])) != GENOME_SIZE:
            raise ValueError(f"Checkpoint contains an invalid genome: {checkpoint_path}")
    for stored in payload.get("population", []):
        if len(stored.get("genome", [])) != GENOME_SIZE:
            raise ValueError(
                f"Checkpoint population contains an invalid genome: {checkpoint_path}"
            )
    if not payload.get("champion"):
        payload["champion"] = payload.get("generation_best") or payload.get("best_ever")
    return payload


def _restore_population(payload: dict[str, Any]) -> list[Individual]:
    population = [
        _individual_from_payload(stored)
        for stored in payload.get("population", [])
    ]
    if len(population) < 2:
        raise ValueError("Checkpoint does not contain a usable population")
    return population


def train(config: TrainingConfig) -> Path:
    """Run evolution and return the final checkpoint path."""
    config.validate()
    rng = random.Random(config.seed)
    config.output_directory = config.output_directory.expanduser().resolve()
    config.output_directory.mkdir(parents=True, exist_ok=True)

    champion: Individual | None = None
    stagnation_count = 0
    if config.resume is not None:
        resume_payload = load_checkpoint(config.resume)
        evaluated_population = _restore_population(resume_payload)
        stored_champion = resume_payload.get("champion")
        if stored_champion:
            champion = _individual_from_payload(stored_champion)
        stagnation_count = int(
            resume_payload.get("training_state", {}).get(
                "validation_stagnation",
                0,
            )
        )
        start_generation = int(resume_payload["generation"]) + 1
        if start_generation >= config.generations:
            return Path(config.resume).expanduser().resolve()
        sigma_multiplier = (
            config.mutation_boost
            if stagnation_count >= config.stagnation_generations
            else 1.0
        )
        population = reproduce(
            evaluated_population,
            config,
            rng,
            champion=champion,
            sigma_multiplier=sigma_multiplier,
        )
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
        screening_started = time.perf_counter()
        evaluate_population(population, config, rng)
        screening_elapsed = time.perf_counter() - screening_started
        ranked = sorted(population, key=lambda individual: individual.fitness, reverse=True)
        generation_best = ranked[0].copy()
        is_final_generation = generation == config.generations - 1
        should_validate = (
            champion is None
            or champion.validation_score is None
            or generation % config.validation_every == 0
            or is_final_generation
        )
        validation_leader: Individual | None = None
        validation_elapsed = 0.0
        if should_validate:
            validation_started = time.perf_counter()
            validation_leader = validate_candidates(ranked, config, generation)
            validation_elapsed = time.perf_counter() - validation_started
            champion_improved = (
                champion is None
                or champion.validation_score is None
                or (
                    validation_leader.validation_score is not None
                    and validation_leader.validation_score
                    > champion.validation_score + 1e-12
                )
            )
            if champion_improved:
                champion = validation_leader.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
        assert champion is not None

        fitnesses = [individual.fitness for individual in population]
        target_name = f"minimax_depth_{config.minimax_depth}"
        reported_candidate = validation_leader or champion
        target_result = reported_candidate.validation_breakdown.get(target_name, {})
        target_win_rate = float(target_result.get("win_rate", 0.0))
        if validation_leader is None:
            validation_text = "skipped"
        else:
            validation_text = f"{validation_leader.validation_score:.4f}"
        print(
            f"Generation {generation + 1}/{config.generations}: "
            f"fitness best={generation_best.fitness:.4f}, "
            f"mean={statistics.mean(fitnesses):.4f}, "
            f"validation_score={validation_text}, "
            f"target wins={target_win_rate:.1%}, "
            f"diversity={_genome_diversity(population):.3f}, "
            f"games/genome={generation_best.games}, "
            f"screen={screening_elapsed:.2f}s, "
            f"validation={validation_elapsed:.2f}s, "
            f"total={time.perf_counter() - generation_started:.2f}s"
        )

        should_save = (
            generation == start_generation
            or (generation + 1) % config.checkpoint_every == 0
            or is_final_generation
        )
        if should_save:
            checkpoint = config.output_directory / f"genetic_gen_{generation:04d}.json"
            save_checkpoint(
                checkpoint,
                generation,
                population,
                generation_best,
                champion,
                config,
                stagnation_count,
            )
            save_checkpoint(
                config.output_directory / "latest.json",
                generation,
                population,
                generation_best,
                champion,
                config,
                stagnation_count,
            )
            final_checkpoint = checkpoint
            print(f"  Saved checkpoint: {checkpoint}")

        if generation < config.generations - 1:
            sigma_multiplier = (
                config.mutation_boost
                if stagnation_count >= config.stagnation_generations
                else 1.0
            )
            population = reproduce(
                population,
                config,
                rng,
                champion=champion,
                sigma_multiplier=sigma_multiplier,
            )

    assert final_checkpoint is not None
    print(
        f"Training complete in {time.perf_counter() - training_started:.2f}s. "
        f"Champion validation score: {champion.validation_score:.4f}."
    )
    return final_checkpoint


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Evolve a three-phase Othello evaluator with alpha-beta search, "
            "paired randomized openings, and fixed champion validation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--generations", type=int, default=defaults.generations)
    parser.add_argument("--population", type=int, default=defaults.population_size)
    parser.add_argument(
        "--games-per-pair",
        type=int,
        default=defaults.games_per_pair,
        help="paired openings per sampled genome opponent (two games each)",
    )
    parser.add_argument(
        "--coevolution-opponents",
        type=int,
        default=defaults.coevolution_opponents,
        help="different population opponents per genome",
    )
    parser.add_argument(
        "--baseline-games",
        type=int,
        default=defaults.baseline_games,
        help="paired openings against each random/greedy baseline",
    )
    parser.add_argument(
        "--minimax-games",
        type=int,
        default=defaults.minimax_games,
        help="paired openings against the target minimax",
    )
    parser.add_argument("--minimax-depth", type=int, default=defaults.minimax_depth)
    parser.add_argument("--minimax-weight", type=float, default=defaults.minimax_weight)
    parser.add_argument(
        "--training-search-depth",
        type=int,
        default=defaults.training_search_depth,
        help="search depth for broad population fitness screening",
    )
    parser.add_argument("--search-depth", type=int, default=defaults.search_depth)
    parser.add_argument(
        "--opening-plies",
        type=int,
        default=defaults.opening_plies,
        help="maximum random legal moves used to create training positions",
    )
    parser.add_argument(
        "--validation-candidates",
        type=int,
        default=defaults.validation_candidates,
    )
    parser.add_argument(
        "--validation-openings",
        type=int,
        default=defaults.validation_openings,
    )
    parser.add_argument("--validation-seed", type=int, default=defaults.validation_seed)
    parser.add_argument(
        "--validation-every",
        type=int,
        default=defaults.validation_every,
        help="run full-depth champion validation every N generations",
    )
    parser.add_argument("--elite-count", type=int, default=defaults.elite_count)
    parser.add_argument("--tournament-size", type=int, default=defaults.tournament_size)
    parser.add_argument("--crossover-rate", type=float, default=defaults.crossover_rate)
    parser.add_argument("--mutation-rate", type=float, default=defaults.mutation_rate)
    parser.add_argument("--mutation-sigma", type=float, default=defaults.mutation_sigma)
    parser.add_argument("--gene-limit", type=float, default=defaults.gene_limit)
    parser.add_argument("--margin-weight", type=float, default=defaults.margin_weight)
    parser.add_argument(
        "--random-immigrants",
        type=int,
        default=defaults.random_immigrants,
    )
    parser.add_argument(
        "--stagnation-generations",
        type=int,
        default=defaults.stagnation_generations,
    )
    parser.add_argument("--mutation-boost", type=float, default=defaults.mutation_boost)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=defaults.checkpoint_every,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "Resume from a version-1 or version-2 checkpoint. --generations "
            "is the total target generation count, not an additional count."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = TrainingConfig(
        generations=args.generations,
        population_size=args.population,
        games_per_pair=args.games_per_pair,
        coevolution_opponents=args.coevolution_opponents,
        baseline_games=args.baseline_games,
        minimax_games=args.minimax_games,
        minimax_depth=args.minimax_depth,
        minimax_weight=args.minimax_weight,
        training_search_depth=args.training_search_depth,
        search_depth=args.search_depth,
        opening_plies=args.opening_plies,
        validation_candidates=args.validation_candidates,
        validation_openings=args.validation_openings,
        validation_seed=args.validation_seed,
        validation_every=args.validation_every,
        elite_count=args.elite_count,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        mutation_sigma=args.mutation_sigma,
        gene_limit=args.gene_limit,
        margin_weight=args.margin_weight,
        random_immigrants=args.random_immigrants,
        stagnation_generations=args.stagnation_generations,
        mutation_boost=args.mutation_boost,
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
