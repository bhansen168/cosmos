#!/usr/bin/env python3
"""Train and load an evolutionary Othello evaluation model.

The current genome contains opening, middlegame, and endgame weights for ten
normalized board features. Training screens the full population with a fast
search, then uses a style-diverse league, rotating validation folds, protected
historical references, and paired champion challenges to choose the checkpoint
champion. Games combine sampled co-evolution with paired randomized openings
against seed, heuristic-search, and accepted historical genetic baselines.
Bard and DQN models are deliberately absent from the opponent pool.

Version-1 checkpoints containing the original twelve-gene, one-ply evaluator
retain their exact evaluator when loaded for play and are projected onto the
current genome only when resumed for training. Existing checkpoint files are
never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
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
CHECKPOINT_VERSION = 3
SUPPORTED_CHECKPOINT_VERSIONS = (1, 2, CHECKPOINT_VERSION)
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "models" / "genetic"
DEFAULT_SEARCH_DEPTH = 2
DEFAULT_ENDGAME_EXACT_EMPTIES = 8
DEFAULT_CHECKPOINT_SUFFIX = "v3"
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

# The original generation-24 file was overwritten during later experiments,
# but its exact version-1 best-ever genome remains in Git commit 99679db0.
# Keeping it here makes the old player a permanent training reference instead
# of relying on a mutable checkpoint filename.
HISTORICAL_V1_GENOME = (
    0.6731700979014472,
    0.1961182771691752,
    0.8820439810285096,
    0.5363647418404579,
    0.47985659043237194,
    1.4490437737873854,
    0.1664352311386429,
    -0.02344160927197031,
    0.31557339178164895,
    0.311750338134197,
    0.029406616705753882,
    -0.4596828003537443,
)

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

# Two different 30-gene checkpoints occupied the unsuffixed generation-24 path
# before v2 was created. Both are retained as immutable opponents because the
# filename alone no longer tells us which one produced the user's benchmark.
PRE_V2_GEN24_EARLY_GENOME = (
    # Git blob fd9241ba:models/genetic/genetic_gen_0024.json
    1.044667145662147,
    1.259250737522687,
    0.5033969435359518,
    -0.20153021074191887,
    -0.46822526108085344,
    -0.1387118723636216,
    0.846074858081777,
    0.8953886792251774,
    0.8324296441554258,
    -0.01704400746808951,
    0.5656037384810517,
    -0.11843771153484445,
    -0.11849253725351251,
    2.2127724348053928,
    1.3830878001135105,
    -0.29128318327605424,
    1.4499229060667724,
    1.2467192351448997,
    1.648627339580486,
    1.2258904677754645,
    0.4157398744935763,
    1.041597766940174,
    -0.06683234313969656,
    1.800822791767185,
    0.06914524856433875,
    0.4314016224815917,
    1.3309995033082522,
    -0.24277471987332433,
    -0.137014670586779,
    0.5668651208217476,
)

PRE_V2_GEN24_LATE_GENOME = (
    # Git blob 89a67496:models/genetic/genetic_gen_0024.json
    0.02027633314604145,
    1.3498621250304823,
    0.2760986382286438,
    0.5156342983781111,
    0.8458713319147233,
    -0.24974582151992591,
    1.292761261792027,
    -0.026720876918888192,
    0.8154265239323888,
    0.4298376943776322,
    -0.057210946485275214,
    0.05526525026883779,
    -0.05644252265626469,
    0.796380214444296,
    1.570400265563758,
    -0.3843456068957236,
    1.2817881020140536,
    1.0450340745061835,
    1.980380955154725,
    1.0581192242906867,
    1.7033055514950481,
    1.0757480380825402,
    -0.0956368299962005,
    1.990847296910807,
    0.7169936113433069,
    0.7498824500228944,
    0.014910097392817984,
    0.4411518612694238,
    0.485173801867249,
    1.2601043172686408,
)

# In controlled, like-for-like tests this v2 generation-24 champion was the
# strongest saved genome against minimax depth 3. Later v2 training regressed,
# so this is the primary non-regression anchor for version 3.
V2_GEN24_REFERENCE_GENOME = (
    -0.3196216055204014,
    0.6563326742065625,
    0.24580239020932207,
    0.5860259697820984,
    0.8451783412773045,
    0.05147072439312589,
    0.8282164103065013,
    1.299682856186614,
    0.9837052313409542,
    1.3156045673560794,
    -0.4833760493846777,
    0.141623756038732,
    0.15431293248584427,
    2.3666036312725547,
    0.9305188145914395,
    -0.08994846462979506,
    1.0059260590067822,
    0.3349756000662764,
    1.7350031456969444,
    0.47541522402174463,
    1.4625003217792825,
    0.2285179786715565,
    0.059460076366382016,
    1.708381751421385,
    0.4255704845483733,
    0.6081614820648584,
    2.075887587166177,
    1.3498812770589985,
    0.602656284950107,
    -0.033339848507672284,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _upgrade_legacy_genome(genome: Sequence[float]) -> list[float]:
    """Project a version-1 two-phase genome onto the current layout.

    The old evaluator blended its phases with ``progress ** 2``. At 50 percent
    progress that is 75 percent opening and 25 percent endgame, so using that
    mixture for the new middle phase is a closer projection than the previous
    50/50 blend. Features that did not exist in version 1 start at zero; silently
    filling them with the default seed used to change a legacy model's policy.
    """
    if len(genome) == GENOME_SIZE:
        return [float(value) for value in genome]
    if len(genome) != LEGACY_GENOME_SIZE:
        raise ValueError(
            f"Genetic genome requires {GENOME_SIZE} values; received {len(genome)}"
        )

    old_opening = [float(value) for value in genome[: len(LEGACY_FEATURE_NAMES)]]
    old_endgame = [float(value) for value in genome[len(LEGACY_FEATURE_NAMES) :]]
    old_middle = [
        0.75 * opening + 0.25 * endgame
        for opening, endgame in zip(old_opening, old_endgame)
    ]
    old_phases = (old_opening, old_middle, old_endgame)
    upgraded = [0.0] * GENOME_SIZE
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


class LegacyGeneticPlayer:
    """Exact one-ply evaluator used by version-1 checkpoints."""

    WIN_SCORE = 1_000.0

    def __init__(
        self,
        genome: Sequence[float],
        name: str = "Genetic v1",
    ) -> None:
        if len(genome) != LEGACY_GENOME_SIZE:
            raise ValueError(
                "Legacy genetic genome requires "
                f"{LEGACY_GENOME_SIZE} values; received {len(genome)}"
            )
        self.genome = tuple(float(value) for value in genome)
        self.name = name

    def evaluate(self, game: HeadlessOthello, color: int) -> float:
        other = opponent(color)
        own_moves = game.legal_moves(color)
        other_moves = game.legal_moves(other)
        if not own_moves and not other_moves:
            scores = game.score()
            difference = scores[color] - scores[other]
            if difference > 0:
                return self.WIN_SCORE + difference
            if difference < 0:
                return -self.WIN_SCORE + difference
            return 0.0

        current, progress, _ = _analyze_board(
            game,
            color,
            len(own_moves),
            len(other_moves),
        )
        features = (
            current[FEATURE_NAMES.index("disc_difference")],
            -min(len(other_moves), 20) / 20.0,
            current[FEATURE_NAMES.index("corners")],
            current[FEATURE_NAMES.index("edges")],
            current[FEATURE_NAMES.index("frontier_safety")],
            current[FEATURE_NAMES.index("positional_value")],
        )
        phase = progress * progress
        opening = self.genome[: len(LEGACY_FEATURE_NAMES)]
        endgame = self.genome[len(LEGACY_FEATURE_NAMES) :]
        weights = (
            (1.0 - phase) * opening[index] + phase * endgame[index]
            for index in range(len(LEGACY_FEATURE_NAMES))
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


class GeneticPlayer:
    """Alpha-beta player whose phase-aware evaluator is genetically evolved."""

    WIN_SCORE = 10_000.0

    def __init__(
        self,
        genome: Sequence[float],
        name: str = "Genetic",
        search_depth: int = DEFAULT_SEARCH_DEPTH,
        endgame_exact_empties: int = DEFAULT_ENDGAME_EXACT_EMPTIES,
    ) -> None:
        if search_depth < 1:
            raise ValueError("Genetic search depth must be at least 1")
        if endgame_exact_empties < 0:
            raise ValueError("Exact endgame threshold cannot be negative")
        self.genome = tuple(_upgrade_legacy_genome(genome))
        self.search_depth = search_depth
        self.endgame_exact_empties = endgame_exact_empties
        self.name = name
        #<<<<<<< Updated upstream
        #    def from_checkpoint(cls, path: str | Path, search_depth: int | None = None) -> GeneticPlayer:
        #=======


    @classmethod
    def from_checkpoint(cls, path: str | Path) -> Player:
        #>>>>>>> Stashed changes
        checkpoint_path = Path(path).expanduser().resolve()
        payload = load_checkpoint(checkpoint_path)
        legacy_champion = payload.get("legacy_champion")
        if payload.get("source_version") == 1 and legacy_champion:
            generation = int(payload["generation"])
            fitness = float(legacy_champion.get("fitness", 0.0))
            return LegacyGeneticPlayer(
                legacy_champion["genome"],
                name=(
                    f"Genetic v1 (best-ever through checkpoint generation "
                    f"{generation}, fitness {fitness:.3f}, "
                    f"{checkpoint_path.name})"
                ),
            )

        champion = payload.get("champion") or payload.get("generation_best")
        if not champion:
            champion = payload["best_ever"]
        checkpoint_generation = int(payload["generation"])
        stored_origin = champion.get("origin_generation")
        origin_generation = (
            checkpoint_generation
            if stored_origin is None
            else int(stored_origin)
        )
        search_depth = int(
            payload.get("config", {}).get("search_depth", DEFAULT_SEARCH_DEPTH)
        )
        endgame_exact_empties = int(
            payload.get("config", {}).get(
                "endgame_exact_empties",
                DEFAULT_ENDGAME_EXACT_EMPTIES,
            )
        )
        validation_score = champion.get("validation_score")
        if validation_score is None:
            quality = f"fitness {float(champion.get('fitness', 0.0)):.3f}"
        else:
            quality = f"validation {float(validation_score):.3f}"
        return cls(
            champion["genome"],
            name=(
                f"Genetic (champion generation {origin_generation}, "
                f"checkpoint generation {checkpoint_generation}, {quality}, "
                f"depth {search_depth}, {checkpoint_path.name})"
            ),
            search_depth=search_depth,
            endgame_exact_empties=endgame_exact_empties,
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
        transposition: dict[
            tuple[tuple[tuple[int, ...], ...], int, int, int],
            tuple[float, str],
        ]
        | None = None,
    ) -> float:
        original_alpha = alpha
        original_beta = beta
        cache_key: tuple[tuple[tuple[int, ...], ...], int, int, int] | None = None
        if transposition is not None:
            cache_key = (
                tuple(tuple(row) for row in game.board),
                color,
                depth,
                root_color,
            )
            cached = transposition.get(cache_key)
            if cached is not None:
                cached_value, bound = cached
                if bound == "exact":
                    return cached_value
                if bound == "lower":
                    alpha = max(alpha, cached_value)
                else:
                    beta = min(beta, cached_value)
                if alpha >= beta:
                    return cached_value

        def store(value: float) -> float:
            if transposition is not None and cache_key is not None:
                if value <= original_alpha:
                    bound = "upper"
                elif value >= original_beta:
                    bound = "lower"
                else:
                    bound = "exact"
                transposition[cache_key] = (value, bound)
            return value

        legal_moves = game.legal_moves(color)
        other_color = opponent(color)
        if not legal_moves:
            other_moves = game.legal_moves(other_color)
            move_counts = {color: 0, other_color: len(other_moves)}
            if not other_moves:
                return store(self.evaluate(game, root_color, move_counts))
            if depth <= 0:
                return store(self.evaluate(game, root_color, move_counts))
            return store(
                self._alpha_beta(
                    game,
                    other_color,
                    depth,
                    alpha,
                    beta,
                    root_color,
                    transposition,
                )
            )

        if depth <= 0:
            other_move_count = len(game.legal_moves(other_color))
            return store(
                self.evaluate(
                    game,
                    root_color,
                    {color: len(legal_moves), other_color: other_move_count},
                )
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
                        transposition,
                    )
                finally:
                    game.undo(color, move)
                value = max(value, child)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return store(value)

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
                    transposition,
                )
            finally:
                game.undo(color, move)
            value = min(value, child)
            beta = min(beta, value)
            if alpha >= beta:
                break
        return store(value)

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
        empty_count = sum(
            square == EMPTY for row in game.board for square in row
        )
        remaining_depth = self.search_depth - 1
        exact_endgame = empty_count <= self.endgame_exact_empties
        if exact_endgame:
            remaining_depth = max(remaining_depth, empty_count - 1)
        transposition = {} if exact_endgame else None

        game = game.copy()
        for move in self._ordered_moves(legal_moves):
            game.play(color, move)
            try:
                value = self._alpha_beta(
                    game,
                    opponent(color),
                    remaining_depth,
                    alpha,
                    float("inf"),
                    color,
                    transposition,
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


def _same_genome(first: Individual, second: Individual) -> bool:
    return tuple(first.genome) == tuple(second.genome)


def _add_to_hall_of_fame(
    hall_of_fame: list[Individual],
    individual: Individual,
    maximum_size: int,
) -> None:
    if maximum_size <= 0:
        return
    if any(_same_genome(stored, individual) for stored in hall_of_fame):
        return
    hall_of_fame.append(individual.copy())
    if len(hall_of_fame) > maximum_size:
        del hall_of_fame[: len(hall_of_fame) - maximum_size]


@dataclass
class TrainingConfig:
    generations: int = 50
    population_size: int = 30
    games_per_pair: int = 1
    coevolution_opponents: int = 6
    baseline_games: int = 1
    minimax_games: int = 1
    # Depth 4 remains available through the CLI, but is not a practical
    # default training opponent in the current pure-Python engine.
    minimax_depth: int = 3
    # Weight of the deepest target; shallower anchors together add half more.
    minimax_weight: float = 3.0
    training_search_depth: int = 1
    search_depth: int = DEFAULT_SEARCH_DEPTH
    endgame_exact_empties: int = DEFAULT_ENDGAME_EXACT_EMPTIES
    opening_plies: int = 14
    validation_candidates: int = 4
    validation_openings: int = 3
    validation_seed: int = 10_000
    validation_every: int = 2
    validation_folds: int = 12
    validation_min_improvement: float = 0.01
    validation_hall_of_fame_opponents: int = 1
    promotion_validation_tolerance: float = 0.0
    promotion_regression_tolerance: float = 0.17
    reference_min_score: float = 0.45
    reference_weight: float = 2.0
    validation_parent_weight: float = 0.35
    challenge_openings: int = 12
    challenge_score: float = 0.55
    hall_of_fame_size: int = 12
    hall_of_fame_opponents: int = 1
    hall_of_fame_weight: float = 2.0
    elite_count: int = 3
    tournament_size: int = 4
    crossover_rate: float = 0.75
    mutation_rate: float = 0.20
    mutation_sigma: float = 0.18
    gene_limit: float = 4.0
    margin_weight: float = 0.05
    normalize_genomes: bool = True
    champion_mutants: int = 4
    warm_start_mutants: int = 8
    random_immigrants: int = 2
    stagnation_generations: int = 6
    mutation_boost: float = 1.75
    stagnation_immigrants: int = 2
    checkpoint_every: int = 5
    checkpoint_suffix: str = DEFAULT_CHECKPOINT_SUFFIX
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
        if self.endgame_exact_empties < 0:
            raise ValueError("exact endgame threshold cannot be negative")
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
        if self.validation_folds < 1:
            raise ValueError("validation folds must be at least 1")
        if self.validation_min_improvement < 0.0:
            raise ValueError("validation minimum improvement cannot be negative")
        if self.validation_hall_of_fame_opponents < 0:
            raise ValueError("validation hall-of-fame opponents cannot be negative")
        if self.promotion_validation_tolerance < 0.0:
            raise ValueError("promotion validation tolerance cannot be negative")
        if not 0.0 <= self.promotion_regression_tolerance <= 1.0:
            raise ValueError("promotion regression tolerance must be between 0 and 1")
        if not 0.0 <= self.reference_min_score <= 1.0:
            raise ValueError("reference minimum score must be between 0 and 1")
        if self.reference_weight < 0.0:
            raise ValueError("reference fitness weight cannot be negative")
        if not 0.0 <= self.validation_parent_weight <= 1.0:
            raise ValueError("validation parent weight must be between 0 and 1")
        if self.challenge_openings < 1:
            raise ValueError("challenge openings must be at least 1")
        if not 0.5 <= self.challenge_score <= 1.0:
            raise ValueError("challenge score must be between 0.5 and 1")
        if self.hall_of_fame_size < 0 or self.hall_of_fame_opponents < 0:
            raise ValueError("hall-of-fame sizes cannot be negative")
        if self.hall_of_fame_weight <= 0.0:
            raise ValueError("hall-of-fame weight must be positive")
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
        if self.champion_mutants < 0 or self.warm_start_mutants < 0:
            raise ValueError("champion and warm-start mutant counts cannot be negative")
        if self.random_immigrants < 0:
            raise ValueError("random immigrants cannot be negative")
        if self.elite_count + self.random_immigrants >= self.population_size:
            raise ValueError("elites and immigrants must leave room for offspring")
        if self.stagnation_generations < 1:
            raise ValueError("stagnation generations must be at least 1")
        if self.mutation_boost < 1.0:
            raise ValueError("mutation boost must be at least 1")
        if self.stagnation_immigrants < 0:
            raise ValueError("stagnation immigrants cannot be negative")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint interval must be at least 1")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.checkpoint_suffix):
            raise ValueError(
                "checkpoint suffix must contain only letters, numbers, underscores, "
                "or hyphens"
            )


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
    target_plies = rng.randint(0, maximum_plies) if maximum_plies else 0
    return _opening_at_plies(rng, target_plies)


def _opening_at_plies(
    rng: random.Random,
    target_plies: int,
) -> tuple[HeadlessOthello, int]:
    """Create an opening with an exact number of legal moves when possible."""
    game = HeadlessOthello()
    color = BLACK
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


def _make_scenario(
    rng: random.Random,
    opening_plies: int,
    *,
    exact: bool = False,
) -> MatchScenario:
    if exact:
        game, current_color = _opening_at_plies(rng, opening_plies)
    else:
        game, current_color = _random_opening(rng, opening_plies)
    return MatchScenario(
        game=game,
        current_color=current_color,
        black_seed=rng.randrange(0, 2**63),
        white_seed=rng.randrange(0, 2**63),
    )


def _stratified_scenarios(
    rng: random.Random,
    count: int,
    maximum_plies: int,
) -> list[MatchScenario]:
    """Cover the standard start, benchmark opening, and later openings.

    Remaining scenarios stay random so repeated challenges do not train on a
    tiny fixed set of opening depths.
    """
    protected_depths = (0, min(4, maximum_plies), maximum_plies)
    depths = list(protected_depths[:count])
    while len(depths) < count:
        depths.append(rng.randint(0, maximum_plies) if maximum_plies else 0)
    return [
        _make_scenario(rng, opening_depth, exact=True)
        for opening_depth in depths
    ]


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


def _minimax_anchor_depths(
    config: TrainingConfig,
    maximum_depth: int | None = None,
) -> tuple[int, ...]:
    """Return a spectrum of heuristic search strengths used as anchors."""
    if maximum_depth is None:
        maximum_depth = config.minimax_depth
    return tuple(range(1, maximum_depth + 1))


def _minimax_anchor_weights(
    config: TrainingConfig,
    maximum_depth: int | None = None,
) -> tuple[tuple[int, float], ...]:
    """Keep the deepest target strong while retaining shallower styles."""
    depths = _minimax_anchor_depths(config, maximum_depth)
    target_depth = depths[-1]
    if len(depths) == 1:
        return ((depths[0], config.minimax_weight),)
    auxiliary_weight = config.minimax_weight * 0.5 / (len(depths) - 1)
    return tuple(
        (
            depth,
            config.minimax_weight
            if depth == target_depth
            else auxiliary_weight,
        )
        for depth in depths
    )


def _trusted_reference_opponents(
    config: TrainingConfig,
    *,
    search_depth: int | None = None,
    endgame_exact_empties: int | None = None,
    historical_index: int | None = None,
) -> list[tuple[str, Player, float]]:
    """Return immutable players used to detect and prevent regressions."""
    if config.reference_weight <= 0.0:
        return []
    if search_depth is None:
        search_depth = config.search_depth
    if endgame_exact_empties is None:
        endgame_exact_empties = config.endgame_exact_empties
    primary_weight = config.reference_weight
    if historical_index is not None:
        primary_weight *= 0.75
    opponents: list[tuple[str, Player, float]] = [
        (
            "reference_v2_gen24",
            GeneticPlayer(
                V2_GEN24_REFERENCE_GENOME,
                "Strong v2 generation-24 reference",
                search_depth=search_depth,
                endgame_exact_empties=endgame_exact_empties,
            ),
            primary_weight,
        ),
    ]
    if historical_index is None:
        return opponents

    historical_weight = config.reference_weight * 0.25
    historical_choice = historical_index % 3
    if historical_choice == 0:
        opponents.append(
            (
                "reference_original_v1",
                LegacyGeneticPlayer(
                    HISTORICAL_V1_GENOME,
                    "Original generation-24 v1 reference",
                ),
                historical_weight,
            )
        )
    elif historical_choice == 1:
        opponents.append(
            (
                "reference_pre_v2_gen24_early",
                GeneticPlayer(
                    PRE_V2_GEN24_EARLY_GENOME,
                    "Early pre-v2 generation-24 reference",
                    search_depth=search_depth,
                    endgame_exact_empties=endgame_exact_empties,
                ),
                historical_weight,
            )
        )
    else:
        opponents.append(
            (
                "reference_pre_v2_gen24_late",
                GeneticPlayer(
                    PRE_V2_GEN24_LATE_GENOME,
                    "Late pre-v2 generation-24 reference",
                    search_depth=search_depth,
                    endgame_exact_empties=endgame_exact_empties,
                ),
                historical_weight,
            )
        )
    return opponents


def evaluate_population(
    population: Sequence[Individual],
    config: TrainingConfig,
    rng: random.Random,
    hall_of_fame: Sequence[Individual] = (),
) -> None:
    """Score genomes with sampled co-evolution and weighted fixed opponents."""
    players = [
        GeneticPlayer(
            individual.genome,
            f"Genome {index}",
            search_depth=config.training_search_depth,
            endgame_exact_empties=0,
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
        (
            GeneticPlayer(
                DEFAULT_SEED_GENOME,
                "Seed evaluator",
                search_depth=config.training_search_depth,
                endgame_exact_empties=0,
            ),
            config.baseline_games,
            1.0,
        ),
    ]
    if config.minimax_games:
        baselines.extend(
            (
                MinimaxPlayer(depth),
                config.minimax_games,
                weight,
            )
            for depth, weight in _minimax_anchor_weights(config)
        )
    baselines.extend(
        (player, 1, weight)
        for _, player, weight in _trusted_reference_opponents(
            config,
            search_depth=config.training_search_depth,
            endgame_exact_empties=0,
        )
    )

    archive_count = min(config.hall_of_fame_opponents, len(hall_of_fame))
    if archive_count:
        archive = rng.sample(list(hall_of_fame), archive_count)
        for archive_index, individual in enumerate(archive):
            baselines.append(
                (
                    GeneticPlayer(
                        individual.genome,
                        f"Hall of fame {archive_index + 1}",
                        search_depth=config.training_search_depth,
                        endgame_exact_empties=0,
                    ),
                    1,
                    config.hall_of_fame_weight,
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


def _validation_opponents(
    config: TrainingConfig,
    generation: int,
    hall_of_fame: Sequence[Individual] = (),
) -> list[tuple[str, Player, float]]:
    """Build a deterministic, style-diverse league for champion selection."""
    fold = (generation // config.validation_every) % config.validation_folds
    opponents: list[tuple[str, Player, float]] = [
        ("random", RandomBaseline(), 0.5),
        ("greedy", GreedyBaseline(), 1.0),
        (
            "seed_genetic",
            GeneticPlayer(
                DEFAULT_SEED_GENOME,
                "Seed evaluator",
                search_depth=config.search_depth,
                endgame_exact_empties=config.endgame_exact_empties,
            ),
            1.0,
        ),
    ]
    opponents.extend(
        (f"minimax_depth_{depth}", MinimaxPlayer(depth), weight)
        for depth, weight in _minimax_anchor_weights(config)
    )
    opponents.extend(
        _trusted_reference_opponents(config, historical_index=fold)
    )

    archive_count = min(
        config.validation_hall_of_fame_opponents,
        len(hall_of_fame),
    )
    if archive_count:
        archive_rng = random.Random(
            config.validation_seed + 20_000_033 + fold * 1_000_037
        )
        archive = archive_rng.sample(list(hall_of_fame), archive_count)
        archive_weight = config.hall_of_fame_weight / archive_count
        opponents.extend(
            (
                f"historical_genetic_{index + 1}",
                GeneticPlayer(
                    individual.genome,
                    f"Historical genetic {index + 1}",
                    search_depth=config.search_depth,
                    endgame_exact_empties=config.endgame_exact_empties,
                ),
                archive_weight,
            )
            for index, individual in enumerate(archive)
        )
    return opponents


def _validation_shortlist(
    ranked: Sequence[Individual],
    champion: Individual | None,
    config: TrainingConfig,
) -> list[Individual]:
    """Mix shallow leaders, a champion neighbor, and trusted starting points."""
    limit = config.validation_candidates
    references = (
        _trusted_reference_individuals(config)
        if champion is None
        else []
    )
    reserved_neighbors = int(champion is not None and limit > len(references) + 1)
    shallow_slots = max(1, limit - len(references) - reserved_neighbors)
    selected = [individual for individual in ranked[:shallow_slots]]

    def add_unique(individual: Individual) -> None:
        if len(selected) >= limit:
            return
        if not any(_same_genome(individual, stored) for stored in selected):
            selected.append(individual)

    if reserved_neighbors and champion is not None:
        alternatives = [
            individual
            for individual in ranked
            if not _same_genome(individual, champion)
            and not any(_same_genome(individual, stored) for stored in selected)
        ]
        if alternatives:
            nearest = min(
                alternatives,
                key=lambda individual: sum(
                    (first - second) ** 2
                    for first, second in zip(individual.genome, champion.genome)
                ),
            )
            add_unique(nearest)

    for reference in references:
        matching = next(
            (
                individual
                for individual in ranked
                if _same_genome(individual, reference)
            ),
            reference,
        )
        add_unique(matching)
    for individual in ranked:
        add_unique(individual)
    return selected


def validate_candidates(
    candidates: Sequence[Individual],
    config: TrainingConfig,
    generation: int,
    hall_of_fame: Sequence[Individual] = (),
) -> Individual:
    """Choose the best candidate on one rotating validation fold."""
    fold = (generation // config.validation_every) % config.validation_folds
    validation_rng = random.Random(
        config.validation_seed + fold * 1_000_003
    )
    scenarios = _stratified_scenarios(
        validation_rng,
        config.validation_openings,
        config.opening_plies,
    )
    opponents = _validation_opponents(config, generation, hall_of_fame)
    validated: list[Individual] = []

    for candidate in candidates[: config.validation_candidates]:
        player = GeneticPlayer(
            candidate.genome,
            "Validation candidate",
            search_depth=config.search_depth,
            endgame_exact_empties=config.endgame_exact_empties,
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
        evaluated.origin_generation = (
            candidate.origin_generation
            if candidate.origin_generation is not None
            else generation
        )
        # Feed the deployed-depth result back into parent selection on
        # validation generations; broad screening remains the cheaper signal.
        candidate.validation_score = evaluated.validation_score
        candidate.validation_games = evaluated.validation_games
        candidate.validation_breakdown = dict(evaluated.validation_breakdown)
        candidate.origin_generation = evaluated.origin_generation
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


def challenge_champion(
    challenger: Individual,
    champion: Individual,
    config: TrainingConfig,
    generation: int,
) -> dict[str, Any]:
    """Play a paired, changing head-to-head promotion match."""
    challenger_player = GeneticPlayer(
        challenger.genome,
        "Champion challenger",
        search_depth=config.search_depth,
        endgame_exact_empties=config.endgame_exact_empties,
    )
    champion_player = GeneticPlayer(
        champion.genome,
        "Incumbent champion",
        search_depth=config.search_depth,
        endgame_exact_empties=config.endgame_exact_empties,
    )
    rng = random.Random(
        config.validation_seed + 10_000_019 + generation * 1_000_033
    )
    wins = 0
    losses = 0
    draws = 0
    disc_margin = 0

    scenarios = _stratified_scenarios(
        rng,
        config.challenge_openings,
        config.opening_plies,
    )
    for scenario in scenarios:
        challenger_black, champion_black = _play_color_pair(
            challenger_player,
            champion_player,
            scenario,
        )
        for outcome, challenger_color in (
            (challenger_black, BLACK),
            (champion_black, WHITE),
        ):
            if outcome.winner is None:
                draws += 1
            elif outcome.winner == challenger_color:
                wins += 1
            else:
                losses += 1
            color_margin = outcome.black_score - outcome.white_score
            disc_margin += (
                color_margin if challenger_color == BLACK else -color_margin
            )

    games = wins + losses + draws
    match_score = (wins + 0.5 * draws) / games
    average_disc_margin = disc_margin / games
    passed = _challenge_passed(
        match_score,
        average_disc_margin,
        config.challenge_score,
    )
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": games,
        "score": match_score,
        "average_disc_margin": average_disc_margin,
        "passed": passed,
    }


def _challenge_passed(
    match_score: float,
    average_disc_margin: float,
    required_score: float,
) -> bool:
    """Require match-point superiority; margin is reporting-only."""
    del average_disc_margin
    return match_score + 1e-12 >= required_score


def _eligible_for_challenge(
    validation_advantage: float,
    config: TrainingConfig,
) -> bool:
    """Let statistically near-equal league leaders settle the result directly."""
    return validation_advantage >= (
        config.validation_min_improvement
        - config.promotion_validation_tolerance
    )


def _promotion_guardrails(
    challenger: Individual,
    incumbent: Individual,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Reject candidates that trade away established general strength."""
    protected_names = {
        "random",
        "greedy",
        "seed_genetic",
        f"minimax_depth_{config.minimax_depth}",
    }
    protected_names.update(
        name
        for name in (
            set(challenger.validation_breakdown)
            & set(incumbent.validation_breakdown)
        )
        if name.startswith("reference_")
    )
    regressions: dict[str, float] = {}
    reference_failures: dict[str, float] = {}

    for name in sorted(protected_names):
        challenger_result = challenger.validation_breakdown.get(name)
        incumbent_result = incumbent.validation_breakdown.get(name)
        if not challenger_result or not incumbent_result:
            regressions[name] = -1.0
            continue
        challenger_score = float(challenger_result["score"])
        incumbent_score = float(incumbent_result["score"])
        difference = challenger_score - incumbent_score
        if difference < -config.promotion_regression_tolerance - 1e-12:
            regressions[name] = difference
        if name.startswith("reference_"):
            if incumbent_score + 1e-12 >= config.reference_min_score:
                if challenger_score + 1e-12 < config.reference_min_score:
                    reference_failures[name] = challenger_score
            elif challenger_score + 1e-12 < incumbent_score:
                reference_failures[name] = challenger_score

    return {
        "passed": not regressions and not reference_failures,
        "regressions": regressions,
        "reference_failures": reference_failures,
    }


def _selection_score(
    individual: Individual,
    validation_parent_weight: float,
) -> float:
    """Blend cheap screening with deployed-depth evidence when available."""
    if individual.validation_score is None:
        return individual.fitness
    return (
        (1.0 - validation_parent_weight) * individual.fitness
        + validation_parent_weight * individual.validation_score
    )


def _tournament_select(
    population: Sequence[Individual],
    tournament_size: int,
    rng: random.Random,
    validation_parent_weight: float = 0.0,
) -> Individual:
    contenders = rng.sample(
        list(population),
        min(tournament_size, len(population)),
    )
    return max(
        contenders,
        key=lambda individual: _selection_score(
            individual,
            validation_parent_weight,
        ),
    )


def _crossover(
    first: Sequence[float],
    second: Sequence[float],
    rng: random.Random,
    gene_limit: float,
    normalize_parents: bool = True,
) -> list[float]:
    first_parent = [float(value) for value in first]
    second_parent = [float(value) for value in second]
    if normalize_parents:
        _normalize_genome_scale(first_parent, gene_limit)
        _normalize_genome_scale(second_parent, gene_limit)
    child = [0.0] * min(len(first_parent), len(second_parent))

    # One blend per feature preserves its opening-to-endgame trajectory.
    # Interpolation stays between the parents; mutation handles extrapolation.
    feature_count = len(FEATURE_NAMES)
    for feature_index in range(feature_count):
        blend = rng.random()
        for phase_index in range(len(PHASE_NAMES)):
            index = phase_index * feature_count + feature_index
            value = (
                blend * first_parent[index]
                + (1.0 - blend) * second_parent[index]
            )
            child[index] = _clamp(value, -gene_limit, gene_limit)
    return child


def _mutate(
    genome: list[float],
    config: TrainingConfig,
    rng: random.Random,
    sigma_multiplier: float,
) -> None:
    feature_count = len(FEATURE_NAMES)
    for feature_index in range(feature_count):
        if rng.random() < config.mutation_rate:
            sigma = config.mutation_sigma * sigma_multiplier
            shared_change = rng.gauss(0.0, sigma * 0.7)
            for phase_index in range(len(PHASE_NAMES)):
                index = phase_index * feature_count + feature_index
                phase_change = rng.gauss(0.0, sigma * 0.3)
                genome[index] = _clamp(
                    genome[index] + shared_change + phase_change,
                    -config.gene_limit,
                    config.gene_limit,
                )


def _normalize_genome_scale(genome: list[float], gene_limit: float) -> None:
    """Remove the evaluator's behaviorally redundant global scale.

    Multiplying every evaluator weight by the same positive constant does not
    change move ordering, but it otherwise gives evolution a neutral direction
    to wander through. Keeping offspring at unit RMS spends mutations on weight
    ratios instead of arbitrary magnitude.
    """
    if not genome:
        return
    rms = math.sqrt(sum(value * value for value in genome) / len(genome))
    if rms <= 1e-12:
        return
    scale = 1.0 / rms
    maximum = max(abs(value) for value in genome)
    if maximum > 0.0:
        scale = min(scale, gene_limit / maximum)
    for index, value in enumerate(genome):
        genome[index] = _clamp(value * scale, -gene_limit, gene_limit)


def _random_individual(config: TrainingConfig, rng: random.Random) -> Individual:
    genome = [
        rng.uniform(-config.gene_limit / 2, config.gene_limit / 2)
        for _ in range(GENOME_SIZE)
    ]
    if config.normalize_genomes:
        _normalize_genome_scale(genome, config.gene_limit)
    return Individual(genome)


def _prepared_genome(
    genome: Sequence[float],
    config: TrainingConfig,
) -> list[float]:
    prepared = _upgrade_legacy_genome(genome)
    if config.normalize_genomes:
        _normalize_genome_scale(prepared, config.gene_limit)
    return prepared


def _trusted_reference_individuals(
    config: TrainingConfig,
) -> list[Individual]:
    if config.reference_weight <= 0.0:
        return []
    return [
        Individual(
            _prepared_genome(V2_GEN24_REFERENCE_GENOME, config),
            origin_generation=12,
        ),
    ]


def _mutated_copy(
    genome: Sequence[float],
    config: TrainingConfig,
    rng: random.Random,
    sigma_multiplier: float,
) -> list[float]:
    mutated = [float(value) for value in genome]
    _mutate(mutated, config, rng, sigma_multiplier)
    if (
        mutated == list(genome)
        and config.mutation_rate > 0.0
        and config.mutation_sigma > 0.0
    ):
        feature_index = rng.randrange(len(FEATURE_NAMES))
        shared_change = rng.gauss(
            0.0,
            config.mutation_sigma * sigma_multiplier,
        )
        for phase_index in range(len(PHASE_NAMES)):
            index = phase_index * len(FEATURE_NAMES) + feature_index
            mutated[index] = _clamp(
                mutated[index] + shared_change,
                -config.gene_limit,
                config.gene_limit,
            )
    if config.normalize_genomes:
        _normalize_genome_scale(mutated, config.gene_limit)
    return mutated


def reproduce(
    population: Sequence[Individual],
    config: TrainingConfig,
    rng: random.Random,
    champion: Individual | None = None,
    sigma_multiplier: float = 1.0,
    immigrant_count: int | None = None,
) -> list[Individual]:
    ranked = sorted(
        population,
        key=lambda individual: _selection_score(
            individual,
            config.validation_parent_weight,
        ),
        reverse=True,
    )
    if immigrant_count is None:
        immigrant_count = config.random_immigrants
    maximum_immigrants = max(
        0,
        config.population_size - config.elite_count - 1,
    )
    immigrant_count = min(immigrant_count, maximum_immigrants)
    offspring_target = config.population_size - immigrant_count
    next_population = [
        Individual(
            ranked[index].genome.copy(),
            origin_generation=ranked[index].origin_generation,
        )
        for index in range(min(config.elite_count, len(ranked)))
    ]

    if champion is not None and len(next_population) < offspring_target:
        champion_genome = tuple(champion.genome)
        if all(tuple(individual.genome) != champion_genome for individual in next_population):
            next_population.append(
                Individual(
                    champion.genome.copy(),
                    origin_generation=champion.origin_generation,
                )
            )

    for reference in _trusted_reference_individuals(config):
        if len(next_population) >= offspring_target:
            break
        if all(not _same_genome(reference, stored) for stored in next_population):
            next_population.append(reference)

    mutation_parent = champion or (
        next_population[0] if next_population else ranked[0]
    )
    for _ in range(config.champion_mutants):
        if len(next_population) >= offspring_target:
            break
        next_population.append(
            Individual(
                _mutated_copy(
                    mutation_parent.genome,
                    config,
                    rng,
                    sigma_multiplier,
                )
            )
        )

    while len(next_population) < offspring_target:
        first = _tournament_select(
            ranked,
            config.tournament_size,
            rng,
            config.validation_parent_weight,
        )
        second = _tournament_select(
            ranked,
            config.tournament_size,
            rng,
            config.validation_parent_weight,
        )
        if rng.random() < config.crossover_rate:
            genome = _crossover(
                first.genome,
                second.genome,
                rng,
                config.gene_limit,
                normalize_parents=config.normalize_genomes,
            )
        else:
            genome = first.genome.copy()
        _mutate(genome, config, rng, sigma_multiplier)
        if config.normalize_genomes:
            _normalize_genome_scale(genome, config.gene_limit)
        next_population.append(Individual(genome))

    while len(next_population) < config.population_size:
        next_population.append(_random_individual(config, rng))
    return next_population


def _reproduction_settings(
    config: TrainingConfig,
    stagnation_count: int,
    last_recovery_stagnation: int = 0,
) -> tuple[float, int, bool]:
    recovery = (
        stagnation_count > 0
        and stagnation_count % config.stagnation_generations == 0
        and stagnation_count != last_recovery_stagnation
    )
    if not recovery:
        return 1.0, config.random_immigrants, False
    return (
        config.mutation_boost,
        config.random_immigrants + config.stagnation_immigrants,
        True,
    )


def create_population(config: TrainingConfig, rng: random.Random) -> list[Individual]:
    anchors = [
        (_prepared_genome(DEFAULT_SEED_GENOME, config), None),
    ]
    if config.reference_weight > 0.0:
        anchors.extend(
            (
                (_prepared_genome(PRE_V2_GEN24_EARLY_GENOME, config), 14),
                (_prepared_genome(PRE_V2_GEN24_LATE_GENOME, config), 12),
                (_prepared_genome(V2_GEN24_REFERENCE_GENOME, config), 12),
            )
        )
    population = [
        Individual(genome.copy(), origin_generation=origin_generation)
        for genome, origin_generation in anchors
    ]
    mutant_target = min(
        config.population_size,
        len(population) + config.warm_start_mutants,
    )
    while len(population) < mutant_target:
        base, _ = anchors[(len(population) - len(anchors)) % len(anchors)]
        population.append(
            Individual(_mutated_copy(base, config, rng, sigma_multiplier=1.0))
        )
    while len(population) < config.population_size:
        population.append(_random_individual(config, rng))
    return population[: config.population_size]


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
        "endgame_exact_empties": config.endgame_exact_empties,
        "opening_plies": config.opening_plies,
        "validation_candidates": config.validation_candidates,
        "validation_openings": config.validation_openings,
        "validation_seed": config.validation_seed,
        "validation_every": config.validation_every,
        "validation_folds": config.validation_folds,
        "validation_min_improvement": config.validation_min_improvement,
        "validation_hall_of_fame_opponents": (
            config.validation_hall_of_fame_opponents
        ),
        "promotion_validation_tolerance": config.promotion_validation_tolerance,
        "promotion_regression_tolerance": config.promotion_regression_tolerance,
        "reference_min_score": config.reference_min_score,
        "reference_weight": config.reference_weight,
        "validation_parent_weight": config.validation_parent_weight,
        "challenge_openings": config.challenge_openings,
        "challenge_score": config.challenge_score,
        "hall_of_fame_size": config.hall_of_fame_size,
        "hall_of_fame_opponents": config.hall_of_fame_opponents,
        "hall_of_fame_weight": config.hall_of_fame_weight,
        "elite_count": config.elite_count,
        "tournament_size": config.tournament_size,
        "crossover_rate": config.crossover_rate,
        "mutation_rate": config.mutation_rate,
        "mutation_sigma": config.mutation_sigma,
        "gene_limit": config.gene_limit,
        "margin_weight": config.margin_weight,
        "normalize_genomes": config.normalize_genomes,
        "champion_mutants": config.champion_mutants,
        "warm_start_mutants": config.warm_start_mutants,
        "random_immigrants": config.random_immigrants,
        "stagnation_generations": config.stagnation_generations,
        "mutation_boost": config.mutation_boost,
        "stagnation_immigrants": config.stagnation_immigrants,
        "checkpoint_every": config.checkpoint_every,
        "checkpoint_suffix": config.checkpoint_suffix,
        "seed": config.seed,
    }


def save_checkpoint(
    path: Path,
    generation: int,
    population: Sequence[Individual],
    generation_best: Individual,
    champion: Individual,
    hall_of_fame: Sequence[Individual],
    config: TrainingConfig,
    stagnation_count: int,
    rng: random.Random,
    last_challenge: dict[str, Any] | None = None,
    last_recovery_stagnation: int = 0,
    validation_leader: Individual | None = None,
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
        "validation_leader": (
            None
            if validation_leader is None
            else _individual_payload(validation_leader)
        ),
        "champion": champion_payload,
        # Compatibility alias for older consumers. Selection now uses champion.
        "best_ever": champion_payload,
        "hall_of_fame": [
            _individual_payload(individual) for individual in hall_of_fame
        ],
        "population": [_individual_payload(individual) for individual in population],
        "config": _config_payload(config),
        "training_state": {
            "validation_stagnation": stagnation_count,
            "rng_state": rng.getstate(),
            "last_challenge": last_challenge,
            "last_recovery_stagnation": last_recovery_stagnation,
            "hall_of_fame_policy": "accepted_champions_only",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _upgrade_version_one_payload(payload: dict[str, Any]) -> dict[str, Any]:
    generation = int(payload.get("generation", 0))
    legacy_champion = dict(
        payload.get("best_ever") or payload.get("generation_best") or {}
    )
    legacy_resume_champion = dict(
        payload.get("generation_best") or legacy_champion
    )
    for key in ("generation_best", "best_ever"):
        if key in payload:
            payload[key] = _individual_payload(_individual_from_payload(payload[key]))
    payload["population"] = [
        _individual_payload(_individual_from_payload(stored))
        for stored in payload.get("population", [])
    ]

    # Version 1 deliberately deployed best_ever, which exact playback retains.
    # Resume from the current generation winner because co-evolution fitness
    # values from different generations are not directly comparable.
    champion = _individual_payload(
        _individual_from_payload(legacy_resume_champion)
    )
    champion["origin_generation"] = generation
    payload["champion"] = champion
    payload["legacy_champion"] = legacy_champion
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
    else:
        payload.setdefault("source_version", version)

    for key in ("generation_best", "validation_leader", "champion", "best_ever"):
        stored = payload.get(key)
        if stored and len(stored.get("genome", [])) != GENOME_SIZE:
            raise ValueError(f"Checkpoint contains an invalid genome: {checkpoint_path}")
    for stored in payload.get("population", []):
        if len(stored.get("genome", [])) != GENOME_SIZE:
            raise ValueError(
                f"Checkpoint population contains an invalid genome: {checkpoint_path}"
            )
    for stored in payload.get("hall_of_fame", []):
        if len(stored.get("genome", [])) != GENOME_SIZE:
            raise ValueError(
                f"Checkpoint hall of fame contains an invalid genome: {checkpoint_path}"
            )
    if not payload.get("champion"):
        payload["champion"] = payload.get("generation_best") or payload.get("best_ever")
    return payload


def _restore_population(
    payload: dict[str, Any],
    config: TrainingConfig | None = None,
) -> list[Individual]:
    population = [
        _individual_from_payload(stored)
        for stored in payload.get("population", [])
    ]
    if len(population) < 2:
        raise ValueError("Checkpoint does not contain a usable population")
    if config is not None and config.normalize_genomes:
        for individual in population:
            _normalize_genome_scale(individual.genome, config.gene_limit)
    return population


def _nested_tuple(value: Any) -> Any:
    """Undo JSON's tuple-to-list conversion for random.Random.setstate."""
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _restore_hall_of_fame(
    payload: dict[str, Any],
    checkpoint_path: Path,
    maximum_size: int,
) -> list[Individual]:
    hall_of_fame: list[Individual] = []
    stored_hall = payload.get("hall_of_fame")
    archive_policy = payload.get("training_state", {}).get(
        "hall_of_fame_policy"
    )
    if stored_hall is not None and archive_policy == "accepted_champions_only":
        for stored in stored_hall:
            _add_to_hall_of_fame(
                hall_of_fame,
                _individual_from_payload(stored),
                maximum_size,
            )
    # Older archives admitted every validation leader, including rejected
    # challengers. Do not carry that contaminated FIFO state into version 3.
    del checkpoint_path

    stored_champion = payload.get("champion")
    if stored_champion:
        _add_to_hall_of_fame(
            hall_of_fame,
            _individual_from_payload(stored_champion),
            maximum_size,
        )
    return hall_of_fame


def _checkpoint_filename(generation: int, suffix: str) -> str:
    return f"genetic_gen_{generation:04d}_{suffix}.json"


def _latest_checkpoint_filename(suffix: str) -> str:
    return f"latest_{suffix}.json"


def train(config: TrainingConfig) -> Path:
    """Run evolution and return the final checkpoint path."""
    config.validate()
    rng = random.Random(config.seed)
    config.output_directory = config.output_directory.expanduser().resolve()
    config.output_directory.mkdir(parents=True, exist_ok=True)

    champion: Individual | None = None
    hall_of_fame: list[Individual] = []
    stagnation_count = 0
    last_recovery_stagnation = 0
    last_challenge: dict[str, Any] | None = None
    last_validation_leader: Individual | None = None
    migrating_checkpoint = False
    if config.resume is not None:
        resume_path = Path(config.resume).expanduser().resolve()
        resume_payload = load_checkpoint(resume_path)
        migrating_checkpoint = int(
            resume_payload.get("source_version", resume_payload["version"])
        ) < CHECKPOINT_VERSION
        evaluated_population = _restore_population(resume_payload, config)
        stored_champion = resume_payload.get("champion")
        if stored_champion:
            champion = _individual_from_payload(stored_champion)
            if config.normalize_genomes:
                _normalize_genome_scale(champion.genome, config.gene_limit)
        stored_validation_leader = resume_payload.get("validation_leader")
        if stored_validation_leader:
            last_validation_leader = _individual_from_payload(
                stored_validation_leader
            )
        hall_of_fame = _restore_hall_of_fame(
            resume_payload,
            resume_path,
            config.hall_of_fame_size,
        )
        if config.normalize_genomes:
            for historical in hall_of_fame:
                _normalize_genome_scale(historical.genome, config.gene_limit)
        training_state = resume_payload.get("training_state", {})
        stagnation_count = int(training_state.get("validation_stagnation", 0))
        last_recovery_stagnation = int(
            training_state.get("last_recovery_stagnation", 0)
        )
        stored_challenge = training_state.get("last_challenge")
        if isinstance(stored_challenge, dict):
            last_challenge = dict(stored_challenge)
        stored_rng_state = training_state.get("rng_state")
        if stored_rng_state is not None:
            try:
                rng.setstate(_nested_tuple(stored_rng_state))
            except (TypeError, ValueError) as exc:
                raise ValueError("Checkpoint contains an invalid RNG state") from exc
        start_generation = int(resume_payload["generation"]) + 1
        if start_generation >= config.generations:
            return resume_path
        if migrating_checkpoint:
            stagnation_count = 0
            last_recovery_stagnation = 0
            last_challenge = None
        sigma_multiplier, immigrant_count, recovery = _reproduction_settings(
            config,
            stagnation_count,
            last_recovery_stagnation,
        )
        population = reproduce(
            evaluated_population,
            config,
            rng,
            champion=champion,
            sigma_multiplier=sigma_multiplier,
            immigrant_count=immigrant_count,
        )
        if migrating_checkpoint:
            # Old validation scores and promotion decisions used a different
            # league. Keep the old champion in the population/archive, but let
            # the v3 league select a fresh incumbent that includes the protected
            # v2 generation-24 anchor.
            champion = None
        if recovery:
            last_recovery_stagnation = stagnation_count
            print(
                "Applying stagnation recovery: "
                f"mutation sigma x{sigma_multiplier:g}, "
                f"{immigrant_count} immigrants."
            )
        print(
            f"Resuming at generation {start_generation + 1} from {resume_path} "
            f"with {len(hall_of_fame)} hall-of-fame genomes"
            f"{'; revalidating the champion under v3' if migrating_checkpoint else ''}."
        )
    else:
        population = create_population(config, rng)
        start_generation = 0

    final_checkpoint: Path | None = None
    training_started = time.perf_counter()

    for generation in range(start_generation, config.generations):
        generation_started = time.perf_counter()
        screening_started = time.perf_counter()
        evaluate_population(population, config, rng, hall_of_fame)
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
        challenge_result: dict[str, Any] | None = None
        promotion_guardrails: dict[str, Any] | None = None
        validation_advantage: float | None = None
        promoted = False
        validation_elapsed = 0.0
        if should_validate:
            validation_started = time.perf_counter()
            validation_leader = validate_candidates(
                _validation_shortlist(ranked, champion, config),
                config,
                generation,
                hall_of_fame,
            )
            last_validation_leader = validation_leader.copy()
            if champion is None or champion.validation_score is None:
                champion = validation_leader.copy()
                _add_to_hall_of_fame(
                    hall_of_fame,
                    champion,
                    config.hall_of_fame_size,
                )
                stagnation_count = 0
                promoted = True
            else:
                incumbent = validate_candidates(
                    [champion],
                    config,
                    generation,
                    hall_of_fame,
                )
                assert validation_leader.validation_score is not None
                assert incumbent.validation_score is not None
                validation_advantage = (
                    validation_leader.validation_score
                    - incumbent.validation_score
                )
                promotion_guardrails = _promotion_guardrails(
                    validation_leader,
                    incumbent,
                    config,
                )
                if _same_genome(validation_leader, champion):
                    champion = incumbent
                    stagnation_count += 1
                elif (
                    _eligible_for_challenge(validation_advantage, config)
                    and promotion_guardrails["passed"]
                ):
                    challenge_result = challenge_champion(
                        validation_leader,
                        incumbent,
                        config,
                        generation,
                    )
                    last_challenge = {
                        **challenge_result,
                        "generation": generation,
                        "validation_advantage": validation_advantage,
                    }
                    if challenge_result["passed"]:
                        _add_to_hall_of_fame(
                            hall_of_fame,
                            incumbent,
                            config.hall_of_fame_size,
                        )
                        champion = validation_leader.copy()
                        _add_to_hall_of_fame(
                            hall_of_fame,
                            champion,
                            config.hall_of_fame_size,
                        )
                        stagnation_count = 0
                        last_recovery_stagnation = 0
                        promoted = True
                    else:
                        champion = incumbent
                        stagnation_count += 1
                else:
                    champion = incumbent
                    stagnation_count += 1
            validation_elapsed = time.perf_counter() - validation_started
        assert champion is not None

        fitnesses = [individual.fitness for individual in population]
        deepest_anchor_name = f"minimax_depth_{config.minimax_depth}"
        reported_candidate = validation_leader or champion
        anchor_result = reported_candidate.validation_breakdown.get(
            deepest_anchor_name,
            {},
        )
        anchor_win_rate = float(anchor_result.get("win_rate", 0.0))
        if validation_leader is None:
            validation_text = "skipped"
        else:
            validation_text = f"{validation_leader.validation_score:.4f}"
        if challenge_result is not None:
            champion_state = "promoted" if promoted else "held"
            promotion_text = (
                f", challenge={challenge_result['score']:.1%} "
                f"({challenge_result['wins']}-{challenge_result['losses']}-"
                f"{challenge_result['draws']}), champion={champion_state}"
            )
        elif promoted:
            promotion_text = ", champion=promoted"
        elif promotion_guardrails is not None and not promotion_guardrails["passed"]:
            failed_names = sorted(
                {
                    *promotion_guardrails["regressions"],
                    *promotion_guardrails["reference_failures"],
                }
            )
            promotion_text = (
                f", validation_delta={validation_advantage:+.4f}, "
                f"guardrail=held ({', '.join(failed_names)})"
            )
        elif validation_advantage is not None:
            promotion_text = (
                f", validation_delta={validation_advantage:+.4f}, champion=held"
            )
        else:
            promotion_text = ""
        print(
            f"Generation {generation + 1}/{config.generations}: "
            f"fitness best={generation_best.fitness:.4f}, "
            f"mean={statistics.mean(fitnesses):.4f}, "
            f"validation_score={validation_text}, "
            f"deepest anchor wins={anchor_win_rate:.1%}, "
            f"diversity={_genome_diversity(population):.3f}, "
            f"games/genome={generation_best.games}, "
            f"screen={screening_elapsed:.2f}s, "
            f"validation={validation_elapsed:.2f}s, "
            f"total={time.perf_counter() - generation_started:.2f}s"
            f"{promotion_text}"
        )

        should_save = (
            generation == start_generation
            or (generation + 1) % config.checkpoint_every == 0
            or is_final_generation
        )
        if should_save:
            checkpoint = config.output_directory / _checkpoint_filename(
                generation,
                config.checkpoint_suffix,
            )
            save_checkpoint(
                checkpoint,
                generation,
                population,
                generation_best,
                champion,
                hall_of_fame,
                config,
                stagnation_count,
                rng,
                last_challenge,
                last_recovery_stagnation,
                last_validation_leader,
            )
            save_checkpoint(
                config.output_directory
                / _latest_checkpoint_filename(config.checkpoint_suffix),
                generation,
                population,
                generation_best,
                champion,
                hall_of_fame,
                config,
                stagnation_count,
                rng,
                last_challenge,
                last_recovery_stagnation,
                last_validation_leader,
            )
            final_checkpoint = checkpoint
            print(f"  Saved checkpoint: {checkpoint}")

        if generation < config.generations - 1:
            sigma_multiplier, immigrant_count, recovery = _reproduction_settings(
                config,
                stagnation_count,
                last_recovery_stagnation,
            )
            population = reproduce(
                population,
                config,
                rng,
                champion=champion,
                sigma_multiplier=sigma_multiplier,
                immigrant_count=immigrant_count,
            )
            if recovery:
                last_recovery_stagnation = stagnation_count
                print(
                    "  Applying stagnation recovery for the next generation: "
                    f"mutation sigma x{sigma_multiplier:g}, "
                    f"{immigrant_count} immigrants."
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
            "paired randomized openings, rotating validation, and robust "
            "champion promotion."
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
        help="paired openings against each minimax anchor depth",
    )
    parser.add_argument(
        "--minimax-depth",
        type=int,
        default=defaults.minimax_depth,
        help="maximum anchor depth; every depth from 1 through this value is used",
    )
    parser.add_argument(
        "--minimax-weight",
        type=float,
        default=defaults.minimax_weight,
        help=(
            "fitness weight for the deepest minimax target; shallower depths "
            "together receive an additional half of this weight"
        ),
    )
    parser.add_argument(
        "--training-search-depth",
        type=int,
        default=defaults.training_search_depth,
        help="search depth for broad population fitness screening",
    )
    parser.add_argument("--search-depth", type=int, default=defaults.search_depth)
    parser.add_argument(
        "--endgame-exact-empties",
        type=int,
        default=defaults.endgame_exact_empties,
        help=(
            "solve positions exactly at or below this many empty squares for "
            "validation and deployed checkpoint players (0 disables)"
        ),
    )
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
    parser.add_argument(
        "--validation-folds",
        type=int,
        default=defaults.validation_folds,
        help="number of deterministic opening suites rotated across validations",
    )
    parser.add_argument(
        "--validation-min-improvement",
        type=float,
        default=defaults.validation_min_improvement,
        help="minimum same-fold score advantage required to challenge the champion",
    )
    parser.add_argument(
        "--validation-hall-of-fame-opponents",
        type=int,
        default=defaults.validation_hall_of_fame_opponents,
        help="historical genetic opponents in the champion-selection league",
    )
    parser.add_argument(
        "--promotion-validation-tolerance",
        type=float,
        default=defaults.promotion_validation_tolerance,
        help=(
            "how far below the incumbent a league leader may score and still "
            "earn a head-to-head challenge"
        ),
    )
    parser.add_argument(
        "--promotion-regression-tolerance",
        type=float,
        default=defaults.promotion_regression_tolerance,
        help="largest allowed validation-score drop against a protected opponent",
    )
    parser.add_argument(
        "--reference-min-score",
        type=float,
        default=defaults.reference_min_score,
        help="minimum match-point score required against each trusted reference",
    )
    parser.add_argument(
        "--reference-weight",
        type=float,
        default=defaults.reference_weight,
        help=(
            "total fitness weight for trusted reference players; zero disables "
            "reference opponents and warm starts"
        ),
    )
    parser.add_argument(
        "--validation-parent-weight",
        type=float,
        default=defaults.validation_parent_weight,
        help="deployed-depth validation contribution to parent selection",
    )
    parser.add_argument(
        "--challenge-openings",
        type=int,
        default=defaults.challenge_openings,
        help="fresh paired openings in a champion promotion match",
    )
    parser.add_argument(
        "--challenge-score",
        type=float,
        default=defaults.challenge_score,
        help="match-point share required to replace the incumbent champion",
    )
    parser.add_argument(
        "--hall-of-fame-size",
        type=int,
        default=defaults.hall_of_fame_size,
        help="maximum number of unique historical champions retained",
    )
    parser.add_argument(
        "--hall-of-fame-opponents",
        type=int,
        default=defaults.hall_of_fame_opponents,
        help="historical champions sampled per genome during screening",
    )
    parser.add_argument(
        "--hall-of-fame-weight",
        type=float,
        default=defaults.hall_of_fame_weight,
        help="fitness weight for games against historical champions",
    )
    parser.add_argument("--elite-count", type=int, default=defaults.elite_count)
    parser.add_argument("--tournament-size", type=int, default=defaults.tournament_size)
    parser.add_argument("--crossover-rate", type=float, default=defaults.crossover_rate)
    parser.add_argument("--mutation-rate", type=float, default=defaults.mutation_rate)
    parser.add_argument("--mutation-sigma", type=float, default=defaults.mutation_sigma)
    parser.add_argument("--gene-limit", type=float, default=defaults.gene_limit)
    parser.add_argument("--margin-weight", type=float, default=defaults.margin_weight)
    parser.add_argument(
        "--no-normalize-genomes",
        dest="normalize_genomes",
        action="store_false",
        default=defaults.normalize_genomes,
        help="disable behavior-preserving unit-RMS normalization of new genomes",
    )
    parser.add_argument(
        "--champion-mutants",
        type=int,
        default=defaults.champion_mutants,
        help="local mutations of the protected champion retained each generation",
    )
    parser.add_argument(
        "--warm-start-mutants",
        type=int,
        default=defaults.warm_start_mutants,
        help="mutated seed/reference genomes used to initialize a new run",
    )
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
        "--stagnation-immigrants",
        type=int,
        default=defaults.stagnation_immigrants,
        help="extra random genomes injected on each stagnation recovery generation",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=defaults.checkpoint_every,
    )
    parser.add_argument(
        "--checkpoint-suffix",
        default=defaults.checkpoint_suffix,
        help="suffix for periodic and latest checkpoint names",
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
            "Resume from a version-1, version-2, or version-3 checkpoint. "
            "--generations is the total target generation count, not an "
            "additional count."
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
        endgame_exact_empties=args.endgame_exact_empties,
        opening_plies=args.opening_plies,
        validation_candidates=args.validation_candidates,
        validation_openings=args.validation_openings,
        validation_seed=args.validation_seed,
        validation_every=args.validation_every,
        validation_folds=args.validation_folds,
        validation_min_improvement=args.validation_min_improvement,
        validation_hall_of_fame_opponents=(
            args.validation_hall_of_fame_opponents
        ),
        promotion_validation_tolerance=args.promotion_validation_tolerance,
        promotion_regression_tolerance=args.promotion_regression_tolerance,
        reference_min_score=args.reference_min_score,
        reference_weight=args.reference_weight,
        validation_parent_weight=args.validation_parent_weight,
        challenge_openings=args.challenge_openings,
        challenge_score=args.challenge_score,
        hall_of_fame_size=args.hall_of_fame_size,
        hall_of_fame_opponents=args.hall_of_fame_opponents,
        hall_of_fame_weight=args.hall_of_fame_weight,
        elite_count=args.elite_count,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        mutation_sigma=args.mutation_sigma,
        gene_limit=args.gene_limit,
        margin_weight=args.margin_weight,
        normalize_genomes=args.normalize_genomes,
        champion_mutants=args.champion_mutants,
        warm_start_mutants=args.warm_start_mutants,
        random_immigrants=args.random_immigrants,
        stagnation_generations=args.stagnation_generations,
        mutation_boost=args.mutation_boost,
        stagnation_immigrants=args.stagnation_immigrants,
        checkpoint_every=args.checkpoint_every,
        checkpoint_suffix=args.checkpoint_suffix,
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
