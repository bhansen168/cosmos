"""Train the COSMOS Othello actor-critic with PPO and self-play.

The trainer deliberately does not use WTHOR or any other human-game corpus.
Every PPO position comes from a current-policy rollout.  Optional supervised
targets are generated online by the repository's minimax search on those same
positions.

Examples:

    python train_ppo.py --iterations 800
    python train_ppo.py --resume models/ppo/best.ppo --iterations 800
    python train_ppo.py --iterations 1 --rollout-steps 128 --channels 16 \
        --blocks 1 --ppo-epochs 1 --validation-every 0 --champion-every 0

Iteration targets are cumulative when resuming.  Scripted and historical
opponent actions are never included in PPO likelihood-ratio updates.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from computer import Computer as GreedyPlayer
from computer import RandomComputer as RandomPlayer
from game import Game, LegalMove
from minimax_model import MinimaxPlayer
from ppo_model import (
    ModelConfig,
    PPOActorCritic,
    PPOPlayer,
    SearchConfig,
    action_to_coord,
    choose_search_move,
    coord_to_action,
    encode_state,
    inverse_transform_action,
    legal_moves_mask,
    load_checkpoint,
    opponent,
    resolve_device,
    save_checkpoint,
    transform_action,
    transform_mask,
    transform_planes,
)


CHAMPION_SCORE_VERSION = 2


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 800
    rollout_steps: int = 8192
    parallel_games: int = 32
    ppo_epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 2e-4
    min_learning_rate_fraction: float = 0.10
    gamma: float = 1.0
    gae_lambda: float = 0.98
    clip_range: float = 0.15
    value_coefficient: float = 0.5
    score_target_weight: float = 0.10
    teacher_fraction: float = 0.02
    teacher_depth: int = 3
    teacher_coefficient: float = 0.10
    entropy_start: float = 0.01
    entropy_end: float = 0.004
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    self_play_fraction: float = 0.45
    league_fraction: float = 0.25
    baseline_fraction: float = 0.30
    checkpoint_every: int = 10
    snapshot_every: int = 10
    validation_every: int = 10
    validation_pairs: int = 8
    champion_every: int = 50
    champion_pairs: int = 32
    validation_opening_plies: int = 4
    validation_search_depth: int = 0
    validation_endgame_exact_empties: int = 0
    champion_search_depth: int = 2
    champion_endgame_exact_empties: int = 8
    champion_max_minimax_depth: int = 4
    champion_margin: float = 0.0
    champion_head_to_head: float = 0.0
    max_league_size: int = 12
    max_hall_of_fame: int = 8
    symmetry: bool = True
    channels: int = 64
    residual_blocks: int = 4
    seed: int = 42
    device: str = "auto"
    output_directory: Path = Path("models/ppo")
    genetic_checkpoint: Path = Path("models/genetic/latest_v2.json")
    resume: Path | None = None

    def validate(self) -> None:
        if self.iterations < 1 or self.rollout_steps < 1:
            raise ValueError("iterations and rollout_steps must be positive")
        if self.parallel_games < 1:
            raise ValueError("parallel_games must be at least 1")
        if self.ppo_epochs < 1 or self.minibatch_size < 1:
            raise ValueError("ppo_epochs and minibatch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.min_learning_rate_fraction <= 1:
            raise ValueError("min_learning_rate_fraction must be in [0, 1]")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in (0, 1] and [0, 1]")
        if self.clip_range <= 0 or self.max_grad_norm <= 0:
            raise ValueError("clip_range and max_grad_norm must be positive")
        if not 0 <= self.score_target_weight <= 0.25:
            raise ValueError("score_target_weight must be in [0, 0.25]")
        if not 0 <= self.teacher_fraction <= 1:
            raise ValueError("teacher_fraction must be in [0, 1]")
        if self.teacher_depth < 1 or self.teacher_coefficient < 0:
            raise ValueError(
                "teacher_depth must be positive and coefficient nonnegative"
            )
        fractions = (
            self.self_play_fraction,
            self.league_fraction,
            self.baseline_fraction,
        )
        if any(fraction < 0 for fraction in fractions):
            raise ValueError("opponent fractions cannot be negative")
        if not math.isclose(sum(fractions), 1.0, abs_tol=1e-8):
            raise ValueError("self-play, league, and baseline fractions must sum to 1")
        for name, interval in (
            ("checkpoint_every", self.checkpoint_every),
            ("snapshot_every", self.snapshot_every),
            ("validation_every", self.validation_every),
            ("champion_every", self.champion_every),
        ):
            if interval < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.validation_pairs < 1 or self.champion_pairs < 1:
            raise ValueError("validation pair counts must be positive")
        if self.validation_opening_plies < 0:
            raise ValueError("validation_opening_plies cannot be negative")
        for name, depth in (
            ("validation_search_depth", self.validation_search_depth),
            ("validation_endgame_exact_empties", self.validation_endgame_exact_empties),
            ("champion_search_depth", self.champion_search_depth),
            ("champion_endgame_exact_empties", self.champion_endgame_exact_empties),
        ):
            if depth < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 2 <= self.champion_max_minimax_depth <= 4:
            raise ValueError("champion_max_minimax_depth must be between 2 and 4")
        if not -1 <= self.champion_head_to_head <= 1:
            raise ValueError("champion_head_to_head must be in [-1, 1]")
        if self.max_league_size < 1 or self.max_hall_of_fame < 1:
            raise ValueError("league and hall-of-fame sizes must be positive")
        ModelConfig(self.channels, self.residual_blocks).validate()


@dataclass
class Experience:
    state: np.ndarray
    legal_mask: np.ndarray
    action: int
    old_log_probability: float
    old_value: float
    teacher_action: int = -1
    reward: float = 0.0
    advantage: float = 0.0
    return_value: float = 0.0


@dataclass(frozen=True)
class RolloutBatch:
    states: Tensor
    legal_masks: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    teacher_actions: Tensor
    advantages: Tensor
    returns: Tensor
    games: int
    black_wins: int
    white_wins: int
    draws: int
    opponent_results: Mapping[str, tuple[int, float]]

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class UpdateMetrics:
    policy_loss: float
    value_loss: float
    teacher_loss: float
    teacher_sample_fraction: float
    entropy: float
    normalized_entropy: float
    approximate_kl: float
    clip_fraction: float
    explained_variance: float
    epochs_completed: int


@dataclass
class OpponentStats:
    games: int = 0
    points: float = 0.0

    @property
    def learner_score(self) -> float:
        return self.points / self.games if self.games else 0.5

    def record(self, outcome: float) -> None:
        self.games += 1
        self.points += (outcome + 1.0) / 2.0

    def to_payload(self) -> dict[str, float | int]:
        return {"games": self.games, "points": self.points}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OpponentStats":
        return cls(
            games=int(payload.get("games", 0)),
            points=float(payload.get("points", 0.0)),
        )


class ModelPlayer:
    """A deterministic Player wrapper around an in-memory actor-critic."""

    def __init__(
        self,
        model: PPOActorCritic,
        device: torch.device,
        name: str = "Current PPO",
        search: SearchConfig | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.name = name
        self.search = search or SearchConfig(depth=0, endgame_exact_empties=0)

    def choose_move(
        self,
        game: Game,
        color: int,
        legal_moves: Sequence[LegalMove],
        rng: random.Random,
    ) -> tuple[int, int]:
        del rng
        return choose_search_move(
            self.model,
            self.device,
            game,
            color,
            legal_moves,
            self.search,
        )


def _find_legal_move(
    legal_moves: Sequence[LegalMove],
    coordinate: tuple[int, int],
) -> LegalMove:
    by_coordinate = {(move.x, move.y): move for move in legal_moves}
    try:
        return by_coordinate[coordinate]
    except KeyError as exc:
        raise RuntimeError(f"Player selected illegal move {coordinate}") from exc


def _game_targets(
    game: Game,
) -> tuple[dict[int, float], dict[int, float], int | None]:
    scores = game.get_score()
    difference = scores[Game.BLACK] - scores[Game.WHITE]
    margins = {
        Game.BLACK: difference / 64.0,
        Game.WHITE: -difference / 64.0,
    }
    if difference == 0:
        return {Game.BLACK: 0.0, Game.WHITE: 0.0}, margins, None
    winner = Game.BLACK if difference > 0 else Game.WHITE
    return {winner: 1.0, opponent(winner): -1.0}, margins, winner


def _game_outcomes(game: Game) -> tuple[dict[int, float], int | None]:
    outcomes, _, winner = _game_targets(game)
    return outcomes, winner


def _finish_trajectory(
    trajectory: list[Experience],
    outcome: float,
    score_margin: float,
    config: TrainingConfig,
) -> None:
    """Compute GAE for policy learning and Monte Carlo value targets."""

    if not trajectory:
        return
    trajectory[-1].reward = outcome
    next_value = 0.0
    next_advantage = 0.0
    final_target = (
        1.0 - config.score_target_weight
    ) * outcome + config.score_target_weight * score_margin
    for index in range(len(trajectory) - 1, -1, -1):
        transition = trajectory[index]
        nonterminal = 0.0 if index == len(trajectory) - 1 else 1.0
        delta = (
            transition.reward
            + config.gamma * next_value * nonterminal
            - transition.old_value
        )
        advantage = (
            delta + config.gamma * config.gae_lambda * next_advantage * nonterminal
        )
        transition.advantage = advantage
        distance = len(trajectory) - 1 - index
        transition.return_value = (config.gamma**distance) * final_target
        next_value = transition.old_value
        next_advantage = advantage


class TrainingSession:
    """One live game used by the synchronous batched rollout collector."""

    def __init__(
        self,
        rng: random.Random,
        fixed_opponent: Any | None,
        opponent_name: str | None,
    ) -> None:
        self.game = Game()
        self.current_color = Game.BLACK
        self.fixed_opponent = fixed_opponent
        self.opponent_name = opponent_name
        if fixed_opponent is None:
            self.trainable_colors = {Game.BLACK, Game.WHITE}
            self.learner_color: int | None = None
        else:
            learner_color = rng.choice((Game.BLACK, Game.WHITE))
            self.trainable_colors = {learner_color}
            self.learner_color = learner_color
        self.trajectories: dict[int, list[Experience]] = {
            Game.BLACK: [],
            Game.WHITE: [],
        }
        self.finished = False

    def advance_to_learner(
        self,
        rng: random.Random,
    ) -> Sequence[LegalMove] | None:
        while not self.finished:
            legal_moves = self.game.legal_moves(self.current_color)
            if not legal_moves:
                other = opponent(self.current_color)
                if not self.game.legal_moves(other):
                    self.finished = True
                    return None
                self.current_color = other
                continue
            if self.current_color in self.trainable_colors:
                return legal_moves
            coordinate = self.fixed_opponent.choose_move(
                self.game,
                self.current_color,
                legal_moves,
                rng,
            )
            self.game.play(
                self.current_color,
                _find_legal_move(legal_moves, coordinate),
            )
            self.current_color = opponent(self.current_color)
        return None

    def play_learner_move(
        self,
        legal_moves: Sequence[LegalMove],
        coordinate: tuple[int, int],
        experience: Experience,
    ) -> None:
        self.trajectories[self.current_color].append(experience)
        self.game.play(
            self.current_color,
            _find_legal_move(legal_moves, coordinate),
        )
        self.current_color = opponent(self.current_color)

    def finish(
        self,
        config: TrainingConfig,
    ) -> tuple[list[Experience], int | None, float | None]:
        outcomes, margins, winner = _game_targets(self.game)
        collected: list[Experience] = []
        for color in sorted(self.trainable_colors):
            trajectory = self.trajectories[color]
            _finish_trajectory(
                trajectory,
                outcomes[color],
                margins[color],
                config,
            )
            collected.extend(trajectory)
        learner_outcome = (
            None if self.learner_color is None else outcomes[self.learner_color]
        )
        return collected, winner, learner_outcome


class AdaptiveOpponentPool:
    """Difficulty-aware mix of historical and fixed search opponents."""

    def __init__(
        self,
        config: TrainingConfig,
        device: torch.device,
        rng: random.Random,
        league: Sequence[Path],
        hall_of_fame: Sequence[Path],
        best_checkpoint: Path | None,
        stats: dict[str, OpponentStats] | None = None,
        cache: dict[Path, PPOPlayer] | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.rng = rng
        self.league = list(league)
        self.hall_of_fame = list(hall_of_fame)
        self.best_checkpoint = best_checkpoint
        self.stats = stats if stats is not None else {}
        self.cache = cache if cache is not None else {}
        self._genetic_player: Any | None = None

    def update_paths(
        self,
        league: Sequence[Path],
        hall_of_fame: Sequence[Path],
        best_checkpoint: Path | None,
    ) -> None:
        self.league = list(league)
        self.hall_of_fame = list(hall_of_fame)
        self.best_checkpoint = best_checkpoint

    def _ppo_player(self, path: Path) -> PPOPlayer:
        resolved = path.resolve()
        if resolved not in self.cache:
            self.cache[resolved] = PPOPlayer(
                resolved,
                self.device,
                search_depth=0,
                endgame_exact_empties=0,
            )
        return self.cache[resolved]

    def _historical_candidates(self) -> list[tuple[str, Any]]:
        paths: list[Path] = []
        for path in [*self.hall_of_fame, *self.league]:
            resolved = path.resolve()
            if resolved.is_file() and resolved not in paths:
                paths.append(resolved)
        if self.best_checkpoint is not None and self.best_checkpoint.is_file():
            resolved_best = self.best_checkpoint.resolve()
            if resolved_best not in paths:
                paths.insert(0, resolved_best)
        return [(f"ppo:{path.name}", self._ppo_player(path)) for path in paths]

    def _fixed_candidates(self, iteration: int) -> list[tuple[str, Any]]:
        candidates: list[tuple[str, Any]] = [
            ("random", RandomPlayer()),
            ("greedy", GreedyPlayer()),
            ("minimax_1", MinimaxPlayer(depth=1)),
        ]
        if iteration >= 25:
            candidates.append(("minimax_2", MinimaxPlayer(depth=2)))
        if iteration >= 100:
            candidates.append(("minimax_3", MinimaxPlayer(depth=3)))
            genetic_path = _available_genetic_checkpoint(self.config)
            if genetic_path is not None:
                if self._genetic_player is None:
                    from genetic_model import GeneticPlayer

                    self._genetic_player = GeneticPlayer.from_checkpoint(genetic_path)
                candidates.append(("genetic_v2", self._genetic_player))
        if iteration >= 200:
            candidates.append(("minimax_4", MinimaxPlayer(depth=4)))
        return candidates

    def _adaptive_choice(
        self, candidates: Sequence[tuple[str, Any]]
    ) -> tuple[str, Any]:
        weights: list[float] = []
        for name, _ in candidates:
            record = self.stats.setdefault(name, OpponentStats())
            if record.games < 8:
                weight = 1.0
            else:
                distance = abs(record.learner_score - 0.5)
                weight = max(0.05, math.exp(-0.5 * (distance / 0.22) ** 2))
            weights.append(weight)
        return self.rng.choices(list(candidates), weights=weights, k=1)[0]

    def choose(self, iteration: int) -> tuple[Any | None, str | None]:
        roll = self.rng.random()
        if roll < self.config.self_play_fraction:
            return None, None
        if roll < self.config.self_play_fraction + self.config.league_fraction:
            historical = self._historical_candidates()
            if historical:
                name, player = self._adaptive_choice(historical)
                return player, name
        name, player = self._adaptive_choice(self._fixed_candidates(iteration))
        return player, name

    def record(self, name: str | None, learner_outcome: float | None) -> None:
        if name is None or learner_outcome is None:
            return
        self.stats.setdefault(name, OpponentStats()).record(learner_outcome)

    def payload(self) -> dict[str, dict[str, float | int]]:
        return {name: record.to_payload() for name, record in self.stats.items()}


def _prepared_state(
    session: TrainingSession,
    legal_moves: Sequence[LegalMove],
    rng: random.Random,
    use_symmetry: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    state = encode_state(session.game, session.current_color, legal_moves)
    mask = legal_moves_mask(legal_moves)
    symmetry = rng.randrange(8) if use_symmetry else 0
    return (
        transform_planes(state, symmetry),
        transform_mask(mask, symmetry),
        symmetry,
    )


def _sample_policy_batch(
    model: PPOActorCritic,
    device: torch.device,
    ready: Sequence[tuple[TrainingSession, Sequence[LegalMove]]],
    rng: random.Random,
    config: TrainingConfig,
    teacher: MinimaxPlayer | None,
) -> list[tuple[Experience, tuple[int, int]]]:
    prepared = [
        _prepared_state(session, legal_moves, rng, config.symmetry)
        for session, legal_moves in ready
    ]
    states = torch.from_numpy(np.stack([item[0] for item in prepared])).to(device)
    masks = torch.from_numpy(np.stack([item[1] for item in prepared])).to(device)
    with torch.inference_mode():
        distribution, values = model.distribution_and_value(states, masks)
        transformed_actions = distribution.sample()
        log_probabilities = distribution.log_prob(transformed_actions)

    decisions: list[tuple[Experience, tuple[int, int]]] = []
    for index, ((session, legal_moves), (state, mask, symmetry)) in enumerate(
        zip(ready, prepared, strict=True)
    ):
        stored_action = int(transformed_actions[index].item())
        board_action = inverse_transform_action(stored_action, symmetry)
        teacher_action = -1
        if teacher is not None and rng.random() < config.teacher_fraction:
            teacher_coordinate = teacher.choose_move(
                session.game,
                session.current_color,
                legal_moves,
                rng,
            )
            teacher_board_action = coord_to_action(*teacher_coordinate)
            teacher_action = transform_action(teacher_board_action, symmetry)
        experience = Experience(
            state=state,
            legal_mask=mask,
            action=stored_action,
            old_log_probability=float(log_probabilities[index].item()),
            old_value=float(values[index].item()),
            teacher_action=teacher_action,
        )
        decisions.append((experience, action_to_coord(board_action)))
    return decisions


def play_training_game(
    model: PPOActorCritic,
    device: torch.device,
    rng: random.Random,
    config: TrainingConfig,
    fixed_opponent: Any | None = None,
) -> tuple[list[Experience], int | None]:
    """Compatibility helper that plays one complete on-policy game."""

    model.eval()
    teacher = (
        MinimaxPlayer(depth=config.teacher_depth)
        if config.teacher_fraction > 0
        else None
    )
    session = TrainingSession(rng, fixed_opponent, None)
    while not session.finished:
        legal_moves = session.advance_to_learner(rng)
        if legal_moves is None:
            break
        decision = _sample_policy_batch(
            model,
            device,
            [(session, legal_moves)],
            rng,
            config,
            teacher,
        )[0]
        session.play_learner_move(legal_moves, decision[1], decision[0])
    experiences, winner, _ = session.finish(config)
    return experiences, winner


def collect_rollout(
    model: PPOActorCritic,
    device: torch.device,
    rng: random.Random,
    config: TrainingConfig,
    league: Sequence[Path],
    league_cache: dict[Path, PPOPlayer],
    hall_of_fame: Sequence[Path] = (),
    opponent_stats: dict[str, OpponentStats] | None = None,
    iteration: int = 0,
    best_checkpoint: Path | None = None,
) -> RolloutBatch:
    """Collect complete games while batching current-policy network calls."""

    model.eval()
    pool = AdaptiveOpponentPool(
        config,
        device,
        rng,
        league,
        hall_of_fame,
        best_checkpoint,
        stats=opponent_stats,
        cache=league_cache,
    )
    teacher = (
        MinimaxPlayer(depth=config.teacher_depth)
        if config.teacher_fraction > 0
        else None
    )
    experiences: list[Experience] = []
    games = black_wins = white_wins = draws = 0
    opponent_results: dict[str, list[float]] = {}

    def new_session() -> TrainingSession:
        fixed_opponent, opponent_name = pool.choose(iteration)
        return TrainingSession(rng, fixed_opponent, opponent_name)

    # A learner controls both colours in pure self-play and one colour against a
    # fixed opponent, so 45 is a useful middle-ground estimate.  It also keeps us
    # from launching a full replacement wave just before the target is reached.
    estimated_decisions_per_game = 45
    active_count = min(
        config.parallel_games,
        max(1, math.ceil(config.rollout_steps / estimated_decisions_per_game)),
    )
    active = [new_session() for _ in range(active_count)]
    while active:
        ready: list[tuple[TrainingSession, Sequence[LegalMove]]] = []
        survivors: list[TrainingSession] = []
        for session in active:
            legal_moves = session.advance_to_learner(rng)
            if legal_moves is not None:
                ready.append((session, legal_moves))
                survivors.append(session)
                continue

            game_experiences, winner, learner_outcome = session.finish(config)
            experiences.extend(game_experiences)
            games += 1
            if winner == Game.BLACK:
                black_wins += 1
            elif winner == Game.WHITE:
                white_wins += 1
            else:
                draws += 1
            pool.record(session.opponent_name, learner_outcome)
            if session.opponent_name is not None and learner_outcome is not None:
                aggregate = opponent_results.setdefault(
                    session.opponent_name, [0.0, 0.0]
                )
                aggregate[0] += 1.0
                aggregate[1] += learner_outcome
        estimated_pending = len(survivors) * estimated_decisions_per_game
        uncovered = config.rollout_steps - len(experiences) - estimated_pending
        replacement_count = min(
            config.parallel_games - len(survivors),
            max(0, math.ceil(uncovered / estimated_decisions_per_game)),
        )
        active = [
            *survivors,
            *(new_session() for _ in range(replacement_count)),
        ]
        if ready:
            decisions = _sample_policy_batch(
                model,
                device,
                ready,
                rng,
                config,
                teacher,
            )
            for (session, legal_moves), (experience, coordinate) in zip(
                ready,
                decisions,
                strict=True,
            ):
                session.play_learner_move(legal_moves, coordinate, experience)

    advantages = np.asarray(
        [experience.advantage for experience in experiences],
        dtype=np.float32,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    summarized_results = {
        name: (int(values[0]), values[1] / values[0])
        for name, values in opponent_results.items()
        if values[0]
    }
    return RolloutBatch(
        states=torch.from_numpy(np.stack([item.state for item in experiences])),
        legal_masks=torch.from_numpy(
            np.stack([item.legal_mask for item in experiences])
        ),
        actions=torch.tensor([item.action for item in experiences], dtype=torch.long),
        old_log_probabilities=torch.tensor(
            [item.old_log_probability for item in experiences],
            dtype=torch.float32,
        ),
        old_values=torch.tensor(
            [item.old_value for item in experiences],
            dtype=torch.float32,
        ),
        teacher_actions=torch.tensor(
            [item.teacher_action for item in experiences],
            dtype=torch.long,
        ),
        advantages=torch.from_numpy(advantages),
        returns=torch.tensor(
            [item.return_value for item in experiences],
            dtype=torch.float32,
        ),
        games=games,
        black_wins=black_wins,
        white_wins=white_wins,
        draws=draws,
        opponent_results=summarized_results,
    )


def entropy_coefficient(config: TrainingConfig, iteration: int) -> float:
    progress = min(max(iteration / max(config.iterations - 1, 1), 0.0), 1.0)
    return config.entropy_start + progress * (config.entropy_end - config.entropy_start)


def learning_rate_factor(config: TrainingConfig, step: int) -> float:
    progress = min(max(step / max(config.iterations, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = config.min_learning_rate_fraction
    return floor + (1.0 - floor) * cosine


def _explained_variance(returns: Tensor, predictions: Tensor) -> float:
    target_variance = torch.var(returns, unbiased=False)
    if float(target_variance) < 1e-12:
        return 0.0
    residual_variance = torch.var(returns - predictions, unbiased=False)
    return float((1.0 - residual_variance / target_variance).item())


def update_policy(
    model: PPOActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    device: torch.device,
    config: TrainingConfig,
    iteration: int,
) -> UpdateMetrics:
    model.train()
    entropy_weight = entropy_coefficient(config, iteration)
    totals = {
        "policy": 0.0,
        "value": 0.0,
        "teacher": 0.0,
        "entropy": 0.0,
        "normalized_entropy": 0.0,
        "kl": 0.0,
        "clip": 0.0,
    }
    updates = 0
    epochs_completed = 0

    for epoch in range(config.ppo_epochs):
        permutation = torch.randperm(batch.size)
        epoch_kl = 0.0
        epoch_updates = 0
        for start in range(0, batch.size, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            states = batch.states[indices].to(device)
            legal_masks = batch.legal_masks[indices].to(device)
            actions = batch.actions[indices].to(device)
            old_log_probabilities = batch.old_log_probabilities[indices].to(device)
            old_values = batch.old_values[indices].to(device)
            teacher_actions = batch.teacher_actions[indices].to(device)
            advantages = batch.advantages[indices].to(device)
            returns = batch.returns[indices].to(device)

            distribution, values = model.distribution_and_value(states, legal_masks)
            log_probabilities = distribution.log_prob(actions)
            log_ratio = log_probabilities - old_log_probabilities
            ratio = log_ratio.exp()
            unclipped_policy = -advantages * ratio
            clipped_policy = -advantages * torch.clamp(
                ratio,
                1.0 - config.clip_range,
                1.0 + config.clip_range,
            )
            policy_loss = torch.maximum(unclipped_policy, clipped_policy).mean()

            clipped_values = old_values + torch.clamp(
                values - old_values,
                -config.clip_range,
                config.clip_range,
            )
            value_loss = (
                0.5
                * torch.maximum(
                    (values - returns).square(),
                    (clipped_values - returns).square(),
                ).mean()
            )

            entropy_per_state = distribution.entropy()
            legal_counts = legal_masks.sum(dim=-1)
            maximum_entropy = torch.log(legal_counts.float()).clamp_min(1e-8)
            normalized_per_state = torch.where(
                legal_counts > 1,
                entropy_per_state / maximum_entropy,
                torch.zeros_like(entropy_per_state),
            )
            normalized_entropy = normalized_per_state.mean()
            entropy = entropy_per_state.mean()

            teacher_rows = teacher_actions >= 0
            if bool(teacher_rows.any()):
                teacher_loss = F.nll_loss(
                    distribution.logits[teacher_rows],
                    teacher_actions[teacher_rows],
                )
            else:
                teacher_loss = torch.zeros((), device=device)
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                + config.teacher_coefficient * teacher_loss
                - entropy_weight * normalized_entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > config.clip_range).float().mean()
                )
            totals["policy"] += float(policy_loss.item())
            totals["value"] += float(value_loss.item())
            totals["teacher"] += float(teacher_loss.item())
            totals["entropy"] += float(entropy.item())
            totals["normalized_entropy"] += float(normalized_entropy.item())
            totals["kl"] += float(approximate_kl.item())
            totals["clip"] += float(clip_fraction.item())
            epoch_kl += float(approximate_kl.item())
            epoch_updates += 1
            updates += 1

        epochs_completed = epoch + 1
        if epoch_updates and epoch_kl / epoch_updates > config.target_kl:
            break

    model.eval()
    predicted_values: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, batch.size, config.minibatch_size):
            states = batch.states[start : start + config.minibatch_size].to(device)
            _, values = model(states)
            predicted_values.append(values.cpu())
    predictions = torch.cat(predicted_values)
    divisor = max(updates, 1)
    teacher_fraction = float((batch.teacher_actions >= 0).float().mean().item())
    return UpdateMetrics(
        policy_loss=totals["policy"] / divisor,
        value_loss=totals["value"] / divisor,
        teacher_loss=totals["teacher"] / divisor,
        teacher_sample_fraction=teacher_fraction,
        entropy=totals["entropy"] / divisor,
        normalized_entropy=totals["normalized_entropy"] / divisor,
        approximate_kl=totals["kl"] / divisor,
        clip_fraction=totals["clip"] / divisor,
        explained_variance=_explained_variance(batch.returns, predictions),
        epochs_completed=epochs_completed,
    )


def _randomized_opening(seed: int, plies: int) -> tuple[Game, int]:
    rng = random.Random(seed)
    game = Game()
    current_color = Game.BLACK
    for _ in range(plies):
        legal_moves = game.legal_moves(current_color)
        if not legal_moves:
            current_color = opponent(current_color)
            legal_moves = game.legal_moves(current_color)
            if not legal_moves:
                break
        game.play(current_color, rng.choice(legal_moves))
        current_color = opponent(current_color)
    return game, current_color


def _play_from_position(
    initial_game: Game,
    current_color: int,
    black_player: Any,
    white_player: Any,
    rng: random.Random,
) -> int | None:
    game = initial_game.clone()
    players = {Game.BLACK: black_player, Game.WHITE: white_player}
    color = current_color
    while True:
        legal_moves = game.legal_moves(color)
        if not legal_moves:
            other = opponent(color)
            if not game.legal_moves(other):
                break
            color = other
            continue
        coordinate = players[color].choose_move(game, color, legal_moves, rng)
        game.play(color, _find_legal_move(legal_moves, coordinate))
        color = opponent(color)
    _, winner = _game_outcomes(game)
    return winner


def _score_result(winner: int | None, learner_color: int) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == learner_color else -1.0


def _available_genetic_checkpoint(config: TrainingConfig) -> Path | None:
    requested = config.genetic_checkpoint
    candidates = [requested]
    if requested.parent.name.casefold() == "genetic":
        candidates.append(requested.parent.parent / requested.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _genetic_validation_player(config: TrainingConfig) -> Any | None:
    checkpoint = _available_genetic_checkpoint(config)
    if checkpoint is None:
        return None
    from genetic_model import GeneticPlayer

    return GeneticPlayer.from_checkpoint(checkpoint)


def evaluate_model(
    model: PPOActorCritic,
    device: torch.device,
    config: TrainingConfig,
    best_checkpoint: Path | None = None,
    *,
    strong: bool = False,
) -> dict[str, float]:
    """Run a reproducible, color-paired fast or champion validation panel."""

    model.eval()
    if strong:
        prefix = "champion"
        pairs = config.champion_pairs
        search = SearchConfig(
            config.champion_search_depth,
            config.champion_endgame_exact_empties,
        )
        opponents = [
            (
                f"minimax_{depth}",
                MinimaxPlayer(depth=depth),
                1.0 + 0.5 * (depth - 2),
            )
            for depth in range(2, config.champion_max_minimax_depth + 1)
        ]
        genetic = _genetic_validation_player(config)
        if genetic is not None:
            opponents.append(("genetic_v2", genetic, 2.0))
    else:
        prefix = "validation"
        pairs = config.validation_pairs
        search = SearchConfig(
            config.validation_search_depth,
            config.validation_endgame_exact_empties,
        )
        opponents = [
            ("random", RandomPlayer(), 0.25),
            ("greedy", GreedyPlayer(), 0.50),
            ("minimax_1", MinimaxPlayer(depth=1), 1.0),
            ("minimax_2", MinimaxPlayer(depth=2), 1.5),
        ]

    learner = ModelPlayer(model, device, search=search)
    if best_checkpoint is not None and best_checkpoint.is_file():
        opponents.append(
            (
                "previous_best",
                PPOPlayer(
                    best_checkpoint,
                    device,
                    search_depth=search.depth,
                    endgame_exact_empties=search.endgame_exact_empties,
                ),
                0.0,
            )
        )

    metrics: dict[str, float] = {}
    weighted_score = 0.0
    total_weight = 0.0
    seed_offset = 700_001 if strong else 0
    for opponent_name, fixed_opponent, weight in opponents:
        scores: list[float] = []
        for pair in range(pairs):
            opening_seed = config.seed + seed_offset + 100_003 * (pair + 1)
            opening, color = _randomized_opening(
                opening_seed,
                config.validation_opening_plies,
            )
            first_winner = _play_from_position(
                opening,
                color,
                learner,
                fixed_opponent,
                random.Random(opening_seed + 1),
            )
            second_winner = _play_from_position(
                opening,
                color,
                fixed_opponent,
                learner,
                random.Random(opening_seed + 1),
            )
            scores.append(_score_result(first_winner, Game.BLACK))
            scores.append(_score_result(second_winner, Game.WHITE))
        mean_score = float(np.mean(scores))
        metrics[f"{prefix}_{opponent_name}"] = mean_score
        if weight > 0:
            weighted_score += weight * mean_score
            total_weight += weight
    metrics[f"{prefix}_composite"] = weighted_score / total_weight
    metrics[f"{prefix}_games"] = float(2 * pairs * len(opponents))
    return metrics


def _serialized_config(config: TrainingConfig) -> dict[str, Any]:
    serialized = asdict(config)
    serialized["output_directory"] = str(config.output_directory)
    serialized["genetic_checkpoint"] = str(config.genetic_checkpoint)
    serialized["resume"] = None if config.resume is None else str(config.resume)
    return serialized


def _relative_paths(paths: Sequence[Path], output: Path) -> list[str]:
    serialized: list[str] = []
    for path in paths:
        try:
            serialized.append(str(path.resolve().relative_to(output.resolve())))
        except ValueError:
            serialized.append(str(path.resolve()))
    return serialized


def _training_state(
    iteration: int,
    total_steps: int,
    best_score: float,
    rng: random.Random,
    opponent_stats: Mapping[str, OpponentStats],
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "iteration": iteration,
        "total_steps": total_steps,
        "best_score": best_score,
        "champion_score_version": CHAMPION_SCORE_VERSION,
        "opponent_stats": {
            name: record.to_payload() for name, record in opponent_stats.items()
        },
        "python_rng_state": rng.getstate(),
        "torch_rng_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(training_state: Mapping[str, Any], rng: random.Random) -> None:
    if "python_rng_state" in training_state:
        rng.setstate(training_state["python_rng_state"])
    if "torch_rng_state" in training_state:
        torch.random.set_rng_state(training_state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_states" in training_state:
        torch.cuda.set_rng_state_all(training_state["cuda_rng_states"])


def _restore_opponent_stats(
    training_state: Mapping[str, Any],
) -> dict[str, OpponentStats]:
    raw_stats = training_state.get("opponent_stats", {})
    if not isinstance(raw_stats, Mapping):
        return {}
    return {
        str(name): OpponentStats.from_payload(payload)
        for name, payload in raw_stats.items()
        if isinstance(payload, Mapping)
    }


def _resolve_saved_paths(
    raw_paths: Sequence[Any],
    output_directory: Path,
) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in raw_paths:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = output_directory / path
        if path.is_file():
            resolved.append(path.resolve())
    return resolved


def _save_training_checkpoint(
    path: Path,
    model: PPOActorCritic,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: TrainingConfig,
    iteration: int,
    total_steps: int,
    best_score: float,
    rng: random.Random,
    metrics: Mapping[str, Any],
    league: Sequence[Path],
    hall_of_fame: Sequence[Path],
    opponent_stats: Mapping[str, OpponentStats],
) -> Path:
    return save_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=_serialized_config(config),
        training_state=_training_state(
            iteration,
            total_steps,
            best_score,
            rng,
            opponent_stats,
        ),
        metrics=metrics,
        league=_relative_paths(league, config.output_directory),
        hall_of_fame=_relative_paths(hall_of_fame, config.output_directory),
    )


def _should_promote(
    validation: Mapping[str, float],
    best_score: float,
    config: TrainingConfig,
    has_previous_best: bool,
) -> bool:
    score = float(validation["champion_composite"])
    if not has_previous_best:
        return True
    head_to_head = float(validation.get("champion_previous_best", -1.0))
    return (
        score > best_score + config.champion_margin
        and head_to_head >= config.champion_head_to_head
    )


def train(config: TrainingConfig) -> Path:
    config.validate()
    device = resolve_device(config.device)
    output = config.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    league_directory = output / "league"
    champion_directory = output / "champions"
    latest_checkpoint = output / "latest.ppo"
    best_checkpoint = output / "best.ppo"
    incumbent_checkpoint: Path | None = (
        best_checkpoint if best_checkpoint.is_file() else None
    )
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    start_iteration = 0
    total_steps = 0
    best_score = -float("inf")
    league: list[Path] = []
    hall_of_fame: list[Path] = []
    opponent_stats: dict[str, OpponentStats] = {}
    resume_payload: Mapping[str, Any] | None = None
    if config.resume is not None:
        model, resume_payload = load_checkpoint(config.resume, device)
        legacy_best = config.resume.resolve().with_name("best.ppo")
        if incumbent_checkpoint is None and legacy_best.is_file():
            incumbent_checkpoint = legacy_best
        saved_state = resume_payload.get("training_state", {})
        if isinstance(saved_state, Mapping):
            start_iteration = int(saved_state.get("iteration", 0))
            total_steps = int(saved_state.get("total_steps", 0))
            if saved_state.get("champion_score_version") == CHAMPION_SCORE_VERSION:
                best_score = float(saved_state.get("best_score", -float("inf")))
            opponent_stats = _restore_opponent_stats(saved_state)
            _restore_rng(saved_state, rng)
        raw_league = resume_payload.get("league", ())
        if isinstance(raw_league, Sequence) and not isinstance(raw_league, str):
            league = _resolve_saved_paths(raw_league, output)
        raw_hall = resume_payload.get("hall_of_fame", ())
        if isinstance(raw_hall, Sequence) and not isinstance(raw_hall, str):
            hall_of_fame = _resolve_saved_paths(raw_hall, output)
    else:
        model = PPOActorCritic(ModelConfig(config.channels, config.residual_blocks)).to(
            device
        )

    if start_iteration >= config.iterations:
        assert config.resume is not None
        completed = config.resume.resolve()
        print(
            f"Checkpoint is already at iteration {start_iteration}; "
            f"the requested target is {config.iterations}."
        )
        return completed

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        eps=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_factor(config, step),
    )
    if resume_payload is not None:
        if "optimizer_state" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if "scheduler_state" in resume_payload:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        scheduler.base_lrs = [config.learning_rate for _ in optimizer.param_groups]
        resumed_factor = learning_rate_factor(config, start_iteration)
        resumed_rates = []
        for base_rate, parameter_group in zip(
            scheduler.base_lrs,
            optimizer.param_groups,
            strict=True,
        ):
            rate = base_rate * resumed_factor
            parameter_group["lr"] = rate
            resumed_rates.append(rate)
        scheduler.last_epoch = start_iteration
        scheduler._last_lr = resumed_rates

    league_cache: dict[Path, PPOPlayer] = {}
    print(
        f"Training PPO on {device} from iteration {start_iteration} "
        f"to {config.iterations}; rollout target {config.rollout_steps} decisions, "
        f"{config.parallel_games} parallel games"
    )
    started = time.monotonic()
    latest_metrics: dict[str, Any] = {}

    try:
        for iteration_index in range(start_iteration, config.iterations):
            iteration = iteration_index + 1
            rollout_started = time.monotonic()
            batch = collect_rollout(
                model,
                device,
                rng,
                config,
                league,
                league_cache,
                hall_of_fame,
                opponent_stats,
                iteration,
                incumbent_checkpoint,
            )
            update_learning_rate = optimizer.param_groups[0]["lr"]
            update_metrics = update_policy(
                model,
                optimizer,
                batch,
                device,
                config,
                iteration_index,
            )
            scheduler.step()
            total_steps += batch.size
            elapsed = max(time.monotonic() - rollout_started, 1e-9)
            latest_metrics = {
                **asdict(update_metrics),
                "iteration": iteration,
                "total_steps": total_steps,
                "rollout_games": batch.games,
                "rollout_steps": batch.size,
                "rollout_black_wins": batch.black_wins,
                "rollout_white_wins": batch.white_wins,
                "rollout_draws": batch.draws,
                "rollout_opponents": dict(batch.opponent_results),
                "steps_per_second": batch.size / elapsed,
                "learning_rate": update_learning_rate,
                "entropy_coefficient": entropy_coefficient(config, iteration_index),
            }

            if config.validation_every > 0 and iteration % config.validation_every == 0:
                latest_metrics.update(
                    evaluate_model(
                        model,
                        device,
                        config,
                        incumbent_checkpoint,
                    )
                )

            promoted = False
            if config.champion_every > 0 and iteration % config.champion_every == 0:
                champion_validation = evaluate_model(
                    model,
                    device,
                    config,
                    incumbent_checkpoint,
                    strong=True,
                )
                latest_metrics.update(champion_validation)
                if _should_promote(
                    champion_validation,
                    best_score,
                    config,
                    incumbent_checkpoint is not None,
                ):
                    promoted = True
                    best_score = champion_validation["champion_composite"]
                    champion_directory.mkdir(parents=True, exist_ok=True)
                    champion_snapshot = save_checkpoint(
                        champion_directory / f"ppo_champion_{iteration:05d}.ppo",
                        model,
                        training_state={"iteration": iteration},
                        metrics=latest_metrics,
                    ).resolve()
                    hall_of_fame.append(champion_snapshot)
                    hall_of_fame = hall_of_fame[-config.max_hall_of_fame :]
                    _save_training_checkpoint(
                        best_checkpoint,
                        model,
                        optimizer,
                        scheduler,
                        config,
                        iteration,
                        total_steps,
                        best_score,
                        rng,
                        latest_metrics,
                        league,
                        hall_of_fame,
                        opponent_stats,
                    )
                    incumbent_checkpoint = best_checkpoint

            if config.snapshot_every > 0 and iteration % config.snapshot_every == 0:
                league_directory.mkdir(parents=True, exist_ok=True)
                snapshot = save_checkpoint(
                    league_directory / f"ppo_iter_{iteration:05d}.ppo",
                    model,
                    training_state={"iteration": iteration},
                    metrics=latest_metrics,
                ).resolve()
                league.append(snapshot)
                while len(league) > config.max_league_size:
                    expired = league.pop(0)
                    league_cache.pop(expired, None)
                    if (
                        expired.parent == league_directory.resolve()
                        and expired.is_file()
                    ):
                        expired.unlink()

            should_checkpoint = (
                config.checkpoint_every > 0 and iteration % config.checkpoint_every == 0
            )
            if should_checkpoint or iteration == config.iterations:
                _save_training_checkpoint(
                    latest_checkpoint,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    iteration,
                    total_steps,
                    best_score,
                    rng,
                    latest_metrics,
                    league,
                    hall_of_fame,
                    opponent_stats,
                )
                if should_checkpoint:
                    _save_training_checkpoint(
                        output / f"ppo_iter_{iteration:05d}.ppo",
                        model,
                        optimizer,
                        scheduler,
                        config,
                        iteration,
                        total_steps,
                        best_score,
                        rng,
                        latest_metrics,
                        league,
                        hall_of_fame,
                        opponent_stats,
                    )

            validation_text = (
                f", validation={latest_metrics['validation_composite']:+.3f}"
                if "validation_composite" in latest_metrics
                else ""
            )
            champion_text = (
                f", champion={latest_metrics['champion_composite']:+.3f}"
                f"{' promoted' if promoted else ''}"
                if "champion_composite" in latest_metrics
                else ""
            )
            print(
                f"Iteration {iteration}/{config.iterations}: "
                f"{batch.size} decisions in {batch.games} games, "
                f"policy={update_metrics.policy_loss:+.4f}, "
                f"value={update_metrics.value_loss:.4f}, "
                f"teacher={update_metrics.teacher_loss:.4f}, "
                f"entropy={update_metrics.normalized_entropy:.3f}, "
                f"KL={update_metrics.approximate_kl:.5f}"
                f"{validation_text}{champion_text}"
            )
    except KeyboardInterrupt:
        interrupted = output / "interrupted.ppo"
        _save_training_checkpoint(
            interrupted,
            model,
            optimizer,
            scheduler,
            config,
            max(start_iteration, int(latest_metrics.get("iteration", start_iteration))),
            total_steps,
            best_score,
            rng,
            latest_metrics,
            league,
            hall_of_fame,
            opponent_stats,
        )
        print(f"Training interrupted; resumable checkpoint saved to {interrupted}")
        raise

    duration = time.monotonic() - started
    print(
        f"Finished PPO training in {duration:.1f}s; latest checkpoint: {latest_checkpoint}"
    )
    return latest_checkpoint


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a batched residual PPO actor-critic through Othello self-play."
    )
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--rollout-steps", type=int, default=8192)
    parser.add_argument("--parallel-games", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate-fraction", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--score-target-weight", type=float, default=0.10)
    parser.add_argument("--teacher-fraction", type=float, default=0.02)
    parser.add_argument("--teacher-depth", type=int, default=3)
    parser.add_argument("--teacher-coefficient", type=float, default=0.10)
    parser.add_argument("--entropy-start", type=float, default=0.01)
    parser.add_argument("--entropy-end", type=float, default=0.004)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--self-play-fraction", type=float, default=0.45)
    parser.add_argument("--league-fraction", type=float, default=0.25)
    parser.add_argument("--baseline-fraction", type=float, default=0.30)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--validation-pairs", type=int, default=8)
    parser.add_argument("--champion-every", type=int, default=50)
    parser.add_argument("--champion-pairs", type=int, default=32)
    parser.add_argument("--validation-opening-plies", type=int, default=4)
    parser.add_argument("--validation-search-depth", type=int, default=0)
    parser.add_argument("--validation-endgame-exact-empties", type=int, default=0)
    parser.add_argument("--champion-search-depth", type=int, default=2)
    parser.add_argument("--champion-endgame-exact-empties", type=int, default=8)
    parser.add_argument("--champion-max-minimax-depth", type=int, default=4)
    parser.add_argument("--champion-margin", type=float, default=0.0)
    parser.add_argument("--champion-head-to-head", type=float, default=0.0)
    parser.add_argument("--max-league-size", type=int, default=12)
    parser.add_argument("--max-hall-of-fame", type=int, default=8)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4, dest="residual_blocks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output-directory", type=Path, default=Path("models/ppo"))
    parser.add_argument(
        "--genetic-checkpoint",
        type=Path,
        default=Path("models/genetic/latest_v2.json"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--no-symmetry",
        action="store_false",
        dest="symmetry",
        help="disable random board rotations/reflections during rollout",
    )
    parser.set_defaults(symmetry=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    train(TrainingConfig(**vars(arguments)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AdaptiveOpponentPool",
    "Experience",
    "ModelPlayer",
    "OpponentStats",
    "RolloutBatch",
    "TrainingConfig",
    "TrainingSession",
    "UpdateMetrics",
    "collect_rollout",
    "entropy_coefficient",
    "evaluate_model",
    "learning_rate_factor",
    "main",
    "play_training_game",
    "train",
    "update_policy",
]
