"""Train the COSMOS Othello actor-critic with proximal policy optimization.

Examples:

    python train_ppo.py --iterations 500
    python train_ppo.py --resume models/ppo/latest.ppo --iterations 1000
    python train_ppo.py --iterations 1 --rollout-steps 128 --channels 16 \
        --blocks 1 --ppo-epochs 1 --validation-every 0

An iteration target is cumulative when resuming.  On-policy samples are drawn
only from the frozen current network; scripted and historical opponent actions
never enter the PPO update.
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
from torch import Tensor

from computer import Computer as GreedyPlayer
from computer import RandomComputer as RandomPlayer
from game import Game, LegalMove
from minimax_model import MinimaxPlayer
from ppo_model import (
    ModelConfig,
    PPOActorCritic,
    PPOPlayer,
    action_to_coord,
    encode_state,
    inverse_transform_action,
    legal_moves_mask,
    load_checkpoint,
    opponent,
    resolve_device,
    save_checkpoint,
    transform_mask,
    transform_planes,
)


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 500
    rollout_steps: int = 8192
    ppo_epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_start: float = 0.01
    entropy_end: float = 0.002
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    self_play_fraction: float = 0.60
    league_fraction: float = 0.25
    baseline_fraction: float = 0.15
    checkpoint_every: int = 10
    snapshot_every: int = 10
    validation_every: int = 10
    validation_pairs: int = 8
    validation_opening_plies: int = 4
    max_league_size: int = 12
    symmetry: bool = True
    channels: int = 64
    residual_blocks: int = 4
    seed: int = 42
    device: str = "auto"
    output_directory: Path = Path("models/ppo")
    resume: Path | None = None

    def validate(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.rollout_steps < 1:
            raise ValueError("rollout_steps must be at least 1")
        if self.ppo_epochs < 1 or self.minibatch_size < 1:
            raise ValueError("ppo_epochs and minibatch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.gae_lambda <= 1 or not 0 < self.gamma <= 1:
            raise ValueError("gamma and gae_lambda must be in (0, 1] and [0, 1]")
        if self.clip_range <= 0 or self.max_grad_norm <= 0:
            raise ValueError("clip_range and max_grad_norm must be positive")
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
        ):
            if interval < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.validation_pairs < 1:
            raise ValueError("validation_pairs must be at least 1")
        if self.validation_opening_plies < 0:
            raise ValueError("validation_opening_plies cannot be negative")
        if self.max_league_size < 1:
            raise ValueError("max_league_size must be at least 1")
        ModelConfig(self.channels, self.residual_blocks).validate()


@dataclass
class Experience:
    state: np.ndarray
    legal_mask: np.ndarray
    action: int
    old_log_probability: float
    old_value: float
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
    advantages: Tensor
    returns: Tensor
    games: int
    black_wins: int
    white_wins: int
    draws: int

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class UpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    explained_variance: float
    epochs_completed: int


class ModelPlayer:
    """A deterministic Player wrapper around an in-memory actor-critic."""

    def __init__(
        self,
        model: PPOActorCritic,
        device: torch.device,
        name: str = "Current PPO",
    ) -> None:
        self.model = model
        self.device = device
        self.name = name

    def choose_move(
        self,
        game: Game,
        color: int,
        legal_moves: Sequence[LegalMove],
        rng: random.Random,
    ) -> tuple[int, int]:
        del rng
        encoded = encode_state(game, color, legal_moves)
        mask = legal_moves_mask(legal_moves)
        states = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        masks = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            distribution, _ = self.model.distribution_and_value(states, masks)
            action = int(distribution.logits.argmax(dim=-1).item())
        return action_to_coord(action)


def _find_legal_move(
    legal_moves: Sequence[LegalMove],
    coordinate: tuple[int, int],
) -> LegalMove:
    by_coordinate = {(move.x, move.y): move for move in legal_moves}
    try:
        return by_coordinate[coordinate]
    except KeyError as exc:
        raise RuntimeError(f"Player selected illegal move {coordinate}") from exc


def _sample_current_policy(
    model: PPOActorCritic,
    device: torch.device,
    game: Game,
    color: int,
    legal_moves: Sequence[LegalMove],
    rng: random.Random,
    use_symmetry: bool,
) -> tuple[Experience, tuple[int, int]]:
    state = encode_state(game, color, legal_moves)
    mask = legal_moves_mask(legal_moves)
    symmetry = rng.randrange(8) if use_symmetry else 0
    transformed_state = transform_planes(state, symmetry)
    transformed_mask = transform_mask(mask, symmetry)
    states = torch.from_numpy(transformed_state).unsqueeze(0).to(device)
    masks = torch.from_numpy(transformed_mask).unsqueeze(0).to(device)
    with torch.inference_mode():
        distribution, value = model.distribution_and_value(states, masks)
        transformed_action = distribution.sample()
        log_probability = distribution.log_prob(transformed_action)
    stored_action = int(transformed_action.item())
    board_action = inverse_transform_action(stored_action, symmetry)
    experience = Experience(
        state=transformed_state,
        legal_mask=transformed_mask,
        action=stored_action,
        old_log_probability=float(log_probability.item()),
        old_value=float(value.item()),
    )
    return experience, action_to_coord(board_action)


def _finish_trajectory(
    trajectory: list[Experience],
    outcome: float,
    gamma: float,
    gae_lambda: float,
) -> None:
    if not trajectory:
        return
    trajectory[-1].reward = outcome
    next_value = 0.0
    next_advantage = 0.0
    for index in range(len(trajectory) - 1, -1, -1):
        transition = trajectory[index]
        nonterminal = 0.0 if index == len(trajectory) - 1 else 1.0
        delta = (
            transition.reward + gamma * next_value * nonterminal - transition.old_value
        )
        advantage = delta + gamma * gae_lambda * next_advantage * nonterminal
        transition.advantage = advantage
        transition.return_value = advantage + transition.old_value
        next_value = transition.old_value
        next_advantage = advantage


def _game_outcomes(game: Game) -> tuple[dict[int, float], int | None]:
    scores = game.get_score()
    if scores[Game.BLACK] == scores[Game.WHITE]:
        return {Game.BLACK: 0.0, Game.WHITE: 0.0}, None
    winner = Game.BLACK if scores[Game.BLACK] > scores[Game.WHITE] else Game.WHITE
    return {winner: 1.0, opponent(winner): -1.0}, winner


def play_training_game(
    model: PPOActorCritic,
    device: torch.device,
    rng: random.Random,
    config: TrainingConfig,
    fixed_opponent: Any | None = None,
) -> tuple[list[Experience], int | None]:
    """Play one game and return only on-policy current-network decisions."""

    game = Game()
    current_color = Game.BLACK
    if fixed_opponent is None:
        trainable_colors = {Game.BLACK, Game.WHITE}
    else:
        trainable_colors = {rng.choice((Game.BLACK, Game.WHITE))}
    trajectories = {Game.BLACK: [], Game.WHITE: []}

    while True:
        legal_moves = game.legal_moves(current_color)
        if not legal_moves:
            other = opponent(current_color)
            if not game.legal_moves(other):
                break
            current_color = other
            continue

        if current_color in trainable_colors:
            experience, coordinate = _sample_current_policy(
                model,
                device,
                game,
                current_color,
                legal_moves,
                rng,
                config.symmetry,
            )
            trajectories[current_color].append(experience)
        else:
            coordinate = fixed_opponent.choose_move(
                game,
                current_color,
                legal_moves,
                rng,
            )
        game.play(current_color, _find_legal_move(legal_moves, coordinate))
        current_color = opponent(current_color)

    outcomes, winner = _game_outcomes(game)
    collected: list[Experience] = []
    for color in sorted(trainable_colors):
        trajectory = trajectories[color]
        _finish_trajectory(
            trajectory,
            outcomes[color],
            config.gamma,
            config.gae_lambda,
        )
        collected.extend(trajectory)
    return collected, winner


def _choose_rollout_opponent(
    config: TrainingConfig,
    rng: random.Random,
    league: Sequence[Path],
    league_cache: dict[Path, PPOPlayer],
    baselines: Sequence[Any],
    device: torch.device,
) -> Any | None:
    roll = rng.random()
    if roll < config.self_play_fraction:
        return None
    if roll < config.self_play_fraction + config.league_fraction and league:
        checkpoint = rng.choice(league)
        if checkpoint not in league_cache:
            league_cache[checkpoint] = PPOPlayer(checkpoint, device)
        return league_cache[checkpoint]
    return rng.choice(baselines)


def collect_rollout(
    model: PPOActorCritic,
    device: torch.device,
    rng: random.Random,
    config: TrainingConfig,
    league: Sequence[Path],
    league_cache: dict[Path, PPOPlayer],
) -> RolloutBatch:
    model.eval()
    baselines = (RandomPlayer(), GreedyPlayer(), MinimaxPlayer(depth=1))
    experiences: list[Experience] = []
    games = black_wins = white_wins = draws = 0
    while len(experiences) < config.rollout_steps:
        fixed_opponent = _choose_rollout_opponent(
            config,
            rng,
            league,
            league_cache,
            baselines,
            device,
        )
        game_experiences, winner = play_training_game(
            model,
            device,
            rng,
            config,
            fixed_opponent,
        )
        experiences.extend(game_experiences)
        games += 1
        if winner == Game.BLACK:
            black_wins += 1
        elif winner == Game.WHITE:
            white_wins += 1
        else:
            draws += 1

    advantages = np.asarray(
        [experience.advantage for experience in experiences], dtype=np.float32
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return RolloutBatch(
        states=torch.from_numpy(np.stack([item.state for item in experiences])),
        legal_masks=torch.from_numpy(
            np.stack([item.legal_mask for item in experiences])
        ),
        actions=torch.tensor([item.action for item in experiences], dtype=torch.long),
        old_log_probabilities=torch.tensor(
            [item.old_log_probability for item in experiences], dtype=torch.float32
        ),
        old_values=torch.tensor(
            [item.old_value for item in experiences], dtype=torch.float32
        ),
        advantages=torch.from_numpy(advantages),
        returns=torch.tensor(
            [item.return_value for item in experiences], dtype=torch.float32
        ),
        games=games,
        black_wins=black_wins,
        white_wins=white_wins,
        draws=draws,
    )


def entropy_coefficient(config: TrainingConfig, iteration: int) -> float:
    progress = min(max(iteration / max(config.iterations - 1, 1), 0.0), 1.0)
    return config.entropy_start + progress * (config.entropy_end - config.entropy_start)


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
        "entropy": 0.0,
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
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - entropy_weight * entropy
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
            totals["entropy"] += float(entropy.item())
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
    return UpdateMetrics(
        policy_loss=totals["policy"] / divisor,
        value_loss=totals["value"] / divisor,
        entropy=totals["entropy"] / divisor,
        approximate_kl=totals["kl"] / divisor,
        clip_fraction=totals["clip"] / divisor,
        explained_variance=_explained_variance(batch.returns, predictions),
        epochs_completed=epochs_completed,
    )


def _randomized_opening(
    seed: int,
    plies: int,
) -> tuple[Game, int]:
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


def evaluate_model(
    model: PPOActorCritic,
    device: torch.device,
    config: TrainingConfig,
    best_checkpoint: Path | None = None,
) -> dict[str, float]:
    """Run a fixed, color-paired validation panel from randomized openings."""

    model.eval()
    learner = ModelPlayer(model, device)
    opponents: list[tuple[str, Any, bool]] = [
        ("random", RandomPlayer(), True),
        ("greedy", GreedyPlayer(), True),
        ("minimax_1", MinimaxPlayer(depth=1), True),
        ("minimax_2", MinimaxPlayer(depth=2), True),
    ]
    if best_checkpoint is not None and best_checkpoint.is_file():
        opponents.append(("previous_best", PPOPlayer(best_checkpoint, device), False))

    metrics: dict[str, float] = {}
    composite_scores: list[float] = []
    for opponent_name, fixed_opponent, include_in_composite in opponents:
        scores: list[float] = []
        for pair in range(config.validation_pairs):
            opening_seed = config.seed + 100_003 * (pair + 1)
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
        metrics[f"validation_{opponent_name}"] = mean_score
        if include_in_composite:
            composite_scores.append(mean_score)
    metrics["validation_composite"] = float(np.mean(composite_scores))
    return metrics


def _serialized_config(config: TrainingConfig) -> dict[str, Any]:
    serialized = asdict(config)
    serialized["output_directory"] = str(config.output_directory)
    serialized["resume"] = None if config.resume is None else str(config.resume)
    return serialized


def _relative_league_paths(paths: Sequence[Path], output: Path) -> list[str]:
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
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "iteration": iteration,
        "total_steps": total_steps,
        "best_score": best_score,
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


def _resolve_saved_league(
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
        ),
        metrics=metrics,
        league=_relative_league_paths(league, config.output_directory),
    )


def train(config: TrainingConfig) -> Path:
    config.validate()
    device = resolve_device(config.device)
    output = config.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    league_directory = output / "league"
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    start_iteration = 0
    total_steps = 0
    best_score = -float("inf")
    league: list[Path] = []
    resume_payload: Mapping[str, Any] | None = None
    if config.resume is not None:
        model, resume_payload = load_checkpoint(config.resume, device)
        saved_state = resume_payload.get("training_state", {})
        if isinstance(saved_state, Mapping):
            start_iteration = int(saved_state.get("iteration", 0))
            total_steps = int(saved_state.get("total_steps", 0))
            best_score = float(saved_state.get("best_score", -float("inf")))
            _restore_rng(saved_state, rng)
        raw_league = resume_payload.get("league", ())
        if isinstance(raw_league, Sequence) and not isinstance(raw_league, str):
            league = _resolve_saved_league(raw_league, output)
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
        lr_lambda=lambda step: max(0.0, 1.0 - step / config.iterations),
    )
    if resume_payload is not None:
        if "optimizer_state" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if "scheduler_state" in resume_payload:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        # The user may extend the cumulative iteration target while resuming.
        # Recompute the next update's learning rate against that new target
        # instead of retaining a rate decayed for the old target.
        resumed_factor = max(0.0, 1.0 - start_iteration / config.iterations)
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

    latest_checkpoint = output / "latest.ppo"
    best_checkpoint = output / "best.ppo"
    league_cache: dict[Path, PPOPlayer] = {}
    print(
        f"Training PPO on {device} from iteration {start_iteration} "
        f"to {config.iterations}; rollout target {config.rollout_steps} decisions"
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
                "steps_per_second": batch.size / elapsed,
                "learning_rate": update_learning_rate,
                "entropy_coefficient": entropy_coefficient(config, iteration_index),
            }

            should_validate = (
                config.validation_every > 0 and iteration % config.validation_every == 0
            )
            if should_validate:
                validation = evaluate_model(
                    model,
                    device,
                    config,
                    best_checkpoint if best_checkpoint.is_file() else None,
                )
                latest_metrics.update(validation)
                validation_score = validation["validation_composite"]
                if validation_score > best_score:
                    best_score = validation_score
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
                    )

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
                    )

            validation_text = (
                f", validation={latest_metrics['validation_composite']:+.3f}"
                if "validation_composite" in latest_metrics
                else ""
            )
            print(
                f"Iteration {iteration}/{config.iterations}: "
                f"{batch.size} decisions in {batch.games} games, "
                f"policy={update_metrics.policy_loss:+.4f}, "
                f"value={update_metrics.value_loss:.4f}, "
                f"entropy={update_metrics.entropy:.3f}, "
                f"KL={update_metrics.approximate_kl:.5f}"
                f"{validation_text}"
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
        description="Train a residual PPO actor-critic through Othello self-play."
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=8192)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-start", type=float, default=0.01)
    parser.add_argument("--entropy-end", type=float, default=0.002)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--self-play-fraction", type=float, default=0.60)
    parser.add_argument("--league-fraction", type=float, default=0.25)
    parser.add_argument("--baseline-fraction", type=float, default=0.15)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--validation-pairs", type=int, default=8)
    parser.add_argument("--validation-opening-plies", type=int, default=4)
    parser.add_argument("--max-league-size", type=int, default=12)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4, dest="residual_blocks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output-directory", type=Path, default=Path("models/ppo"))
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
    "Experience",
    "ModelPlayer",
    "RolloutBatch",
    "TrainingConfig",
    "UpdateMetrics",
    "collect_rollout",
    "entropy_coefficient",
    "evaluate_model",
    "main",
    "play_training_game",
    "train",
    "update_policy",
]
