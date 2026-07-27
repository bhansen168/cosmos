"""Train the independent COSMOS AlphaZero Othello model.

Training data is produced exclusively from current-network MCTS self-play and
the exact endgame solver.  WTHOR and other human-game corpora are deliberately
absent from this pipeline.

Examples:

    python train_alphazero.py --generations 500
    python train_alphazero.py --resume models/alphazero/latest.az \
        --generations 800
    python train_alphazero.py --generations 1 --games-per-generation 2 \
        --simulations 2 --channels 8 --blocks 1 --value-hidden 16 \
        --training-steps 1 --batch-size 32 --evaluation-every 0

GPU collection uses many simultaneous games to batch leaf inference through a
single model.  CPU-only collection may additionally use ``--self-play-workers``
to distribute independent game groups across spawned processes.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time
import warnings
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .board import (
    ACTION_DIM,
    BLACK,
    BOARD_SIZE,
    WHITE,
    BitBoard,
    iter_actions,
    opponent_color,
)
from .mcts import MCTSConfig, MCTSNode, NeuralMCTS, select_action
from .model import (
    DEFAULT_OUTPUT_DIRECTORY,
    AlphaZeroEvaluator,
    AlphaZeroModelConfig,
    AlphaZeroNetwork,
    load_checkpoint,
    prepare_model,
    resolve_device,
    save_checkpoint,
)
from .replay import ReplayBatch, ReplayBuffer, SelfPlayRecord


def recommended_cpu_workers() -> int:
    """Use one process per likely physical core on a hybrid laptop CPU."""

    logical_cpus = os.cpu_count() or 2
    return max(1, min(10, logical_cpus - 2))


@dataclass
class TrainingConfig:
    generations: int = 500
    games_per_generation: int = 64
    simulations: int = 128
    parallel_games: int = 64
    self_play_workers: int = field(default_factory=recommended_cpu_workers)
    worker_torch_threads: int = 1
    channels: int = 96
    residual_blocks: int = 6
    value_hidden: int = 128
    c_puct: float = 1.5
    fpu_reduction: float = 0.20
    leaf_batch_size: int = 8
    virtual_loss: float = 1.0
    dirichlet_alpha: float = 0.30
    dirichlet_fraction: float = 0.25
    exact_endgame_empties: int = 10
    temperature: float = 1.0
    temperature_moves: int = 16
    inference_batch_size: int = 512
    inference_cache_size: int = 100_000
    replay_capacity: int = 500_000
    minimum_replay_size: int = 2_048
    replay_recent_fraction: float = 0.50
    replay_prioritized_fraction: float = 0.20
    replay_recent_generations: int = 10
    replay_priority_alpha: float = 0.60
    reanalysis_positions: int = 256
    reanalysis_minimum_age: int = 5
    reanalysis_batch_size: int = 128
    batch_size: int = 256
    training_steps: int = 64
    learning_rate: float = 1e-3
    minimum_learning_rate_fraction: float = 0.05
    weight_decay: float = 1e-4
    margin_coefficient: float = 0.10
    ownership_coefficient: float = 0.15
    ema_decay: float = 0.995
    max_grad_norm: float = 5.0
    mixed_precision: bool = True
    symmetry_augmentation: bool = True
    evaluation_every: int = 10
    evaluation_pairs: int = 16
    evaluation_simulations: int = 192
    evaluation_opening_plies: int = 8
    promotion_score: float = 0.55
    checkpoint_every: int = 1
    snapshot_every: int = 25
    replay_save_every: int = 5
    compress_replay: bool = False
    seed: int = 0
    device: str = "auto"
    output_directory: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIRECTORY)
    resume: Path | None = None

    def validate(self) -> None:
        positive = (
            ("generations", self.generations),
            ("games_per_generation", self.games_per_generation),
            ("simulations", self.simulations),
            ("parallel_games", self.parallel_games),
            ("self_play_workers", self.self_play_workers),
            ("channels", self.channels),
            ("residual_blocks", self.residual_blocks),
            ("value_hidden", self.value_hidden),
            ("inference_batch_size", self.inference_batch_size),
            ("replay_capacity", self.replay_capacity),
            ("minimum_replay_size", self.minimum_replay_size),
            ("batch_size", self.batch_size),
            ("learning_rate", self.learning_rate),
        )
        for name, value in positive:
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.worker_torch_threads < 1:
            raise ValueError("worker_torch_threads must be positive")
        if self.training_steps < 0:
            raise ValueError("training_steps cannot be negative")
        if self.temperature < 0 or self.temperature_moves < 0:
            raise ValueError("Temperature settings cannot be negative")
        if self.fpu_reduction < 0:
            raise ValueError("fpu_reduction cannot be negative")
        if self.leaf_batch_size < 1:
            raise ValueError("leaf_batch_size must be positive")
        if self.virtual_loss < 0:
            raise ValueError("virtual_loss cannot be negative")
        if self.exact_endgame_empties < 0:
            raise ValueError("exact_endgame_empties cannot be negative")
        if self.inference_cache_size < 0:
            raise ValueError("inference_cache_size cannot be negative")
        if self.minimum_replay_size > self.replay_capacity:
            raise ValueError("minimum_replay_size exceeds replay capacity")
        if not 0 <= self.replay_recent_fraction <= 1:
            raise ValueError("replay_recent_fraction must be in [0, 1]")
        if not 0 <= self.replay_prioritized_fraction <= 1:
            raise ValueError("replay_prioritized_fraction must be in [0, 1]")
        if self.replay_recent_fraction + self.replay_prioritized_fraction > 1:
            raise ValueError("Replay sampling fractions cannot exceed one")
        if self.replay_recent_generations < 1:
            raise ValueError("replay_recent_generations must be positive")
        if self.replay_priority_alpha < 0:
            raise ValueError("replay_priority_alpha cannot be negative")
        if self.reanalysis_positions < 0:
            raise ValueError("reanalysis_positions cannot be negative")
        if self.reanalysis_minimum_age < 1:
            raise ValueError("reanalysis_minimum_age must be positive")
        if self.reanalysis_batch_size < 1:
            raise ValueError("reanalysis_batch_size must be positive")
        if not 0 < self.minimum_learning_rate_fraction <= 1:
            raise ValueError("minimum_learning_rate_fraction must be in (0, 1]")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.margin_coefficient < 0 or self.ownership_coefficient < 0:
            raise ValueError("Auxiliary loss coefficients cannot be negative")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        for name, value in (
            ("evaluation_every", self.evaluation_every),
            ("checkpoint_every", self.checkpoint_every),
            ("snapshot_every", self.snapshot_every),
            ("replay_save_every", self.replay_save_every),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.evaluation_pairs < 1:
            raise ValueError("evaluation_pairs must be positive")
        if self.evaluation_simulations < 1:
            raise ValueError("evaluation_simulations must be positive")
        if self.evaluation_opening_plies < 0:
            raise ValueError("evaluation_opening_plies cannot be negative")
        if not 0.5 <= self.promotion_score <= 1:
            raise ValueError("promotion_score must be between 0.5 and 1")

        AlphaZeroModelConfig(
            self.channels,
            self.residual_blocks,
            self.value_hidden,
        ).validate()
        MCTSConfig(
            simulations=self.simulations,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_fraction=self.dirichlet_fraction,
            exact_endgame_empties=self.exact_endgame_empties,
            fpu_reduction=self.fpu_reduction,
            leaf_batch_size=self.leaf_batch_size,
            virtual_loss=self.virtual_loss,
        ).validate()


@dataclass(frozen=True)
class SelfPlayMetrics:
    games: int
    positions: int
    black_wins: int
    white_wins: int
    draws: int
    exact_roots: int
    elapsed_seconds: float


@dataclass(frozen=True)
class UpdateMetrics:
    policy_loss: float
    wdl_loss: float
    margin_loss: float
    ownership_loss: float
    total_loss: float
    gradient_norm: float
    wdl_accuracy: float
    explained_variance: float
    updates: int


@dataclass(frozen=True)
class ReanalysisMetrics:
    positions: int
    exact_roots: int
    elapsed_seconds: float


@dataclass(slots=True)
class _PendingRecord:
    board: BitBoard
    color: int
    visits: np.ndarray


@dataclass(slots=True)
class _ActiveGame:
    board: BitBoard
    color: int
    root: MCTSNode
    records: list[_PendingRecord]
    moves: int = 0

    @classmethod
    def create(cls) -> _ActiveGame:
        board = BitBoard.initial()
        return cls(
            board=board,
            color=BLACK,
            root=MCTSNode(board),
            records=[],
        )


_WORKER_MODEL: AlphaZeroNetwork | None = None
_WORKER_MODEL_CONFIG: AlphaZeroModelConfig | None = None


def _absolute_bitboards(board: BitBoard, color: int) -> tuple[int, int]:
    if color == BLACK:
        return board.player, board.opponent
    return board.opponent, board.player


def _ownership_labels(own: int, other: int) -> np.ndarray:
    labels = np.ones(ACTION_DIM, dtype=np.int8)
    for action in iter_actions(other):
        labels[action] = 0
    for action in iter_actions(own):
        labels[action] = 2
    return labels


def _finish_game(
    active: _ActiveGame,
    generation: int,
) -> tuple[list[SelfPlayRecord], int | None]:
    black, white = _absolute_bitboards(active.board, active.color)
    black_score = black.bit_count()
    white_score = white.bit_count()
    if black_score == white_score:
        winner = None
    else:
        winner = BLACK if black_score > white_score else WHITE
    black_margin = (black_score - white_score) / 64.0

    records: list[SelfPlayRecord] = []
    for pending in active.records:
        if winner is None:
            outcome = 0
        else:
            outcome = 1 if winner == pending.color else -1
        if pending.color == BLACK:
            own, other = black, white
            margin = black_margin
        else:
            own, other = white, black
            margin = -black_margin
        records.append(
            SelfPlayRecord(
                player=pending.board.player,
                opponent=pending.board.opponent,
                visit_counts=pending.visits,
                outcome=outcome,
                margin=margin,
                ownership=_ownership_labels(own, other),
                generation=generation,
            )
        )
    return records, winner


def _advance_forced_passes(active: _ActiveGame) -> bool:
    """Advance passes and return False once the game is terminal."""

    while not active.board.legal_moves_bits():
        passed = active.board.pass_turn()
        if not passed.legal_moves_bits():
            return False
        active.board = passed
        active.color = opponent_color(active.color)
        active.root = active.root.child_for_pass()
    return True


def _mcts_config(config: TrainingConfig, simulations: int | None = None) -> MCTSConfig:
    return MCTSConfig(
        simulations=config.simulations if simulations is None else simulations,
        c_puct=config.c_puct,
        fpu_reduction=config.fpu_reduction,
        leaf_batch_size=config.leaf_batch_size,
        virtual_loss=config.virtual_loss,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_fraction=config.dirichlet_fraction,
        exact_endgame_empties=config.exact_endgame_empties,
    )


def collect_self_play(
    model: AlphaZeroNetwork,
    device: str | torch.device,
    config: TrainingConfig,
    generation: int,
    *,
    games: int | None = None,
    seed: int | None = None,
) -> tuple[list[SelfPlayRecord], SelfPlayMetrics]:
    """Collect self-play while batching one leaf per active game per MCTS wave."""

    target_games = config.games_per_generation if games is None else games
    if target_games < 1:
        raise ValueError("Self-play game count must be positive")
    selected_device = resolve_device(device)
    model.eval()
    evaluator = AlphaZeroEvaluator(
        model,
        selected_device,
        maximum_batch_size=config.inference_batch_size,
        cache_size=config.inference_cache_size,
        mixed_precision=config.mixed_precision,
    )
    mcts = NeuralMCTS(evaluator, _mcts_config(config))
    rng = np.random.default_rng(
        config.seed + generation * 1_000_003 if seed is None else seed
    )
    active: list[_ActiveGame] = [
        _ActiveGame.create() for _ in range(min(config.parallel_games, target_games))
    ]
    started_games = len(active)
    completed_games = 0
    records: list[SelfPlayRecord] = []
    black_wins = white_wins = draws = exact_roots = 0
    started_at = time.monotonic()

    while active:
        ready: list[_ActiveGame] = []
        survivors: list[_ActiveGame] = []
        for game in active:
            if _advance_forced_passes(game):
                ready.append(game)
                survivors.append(game)
                continue
            game_records, winner = _finish_game(game, generation)
            records.extend(game_records)
            completed_games += 1
            if winner == BLACK:
                black_wins += 1
            elif winner == WHITE:
                white_wins += 1
            else:
                draws += 1

        while started_games < target_games and len(survivors) < config.parallel_games:
            replacement = _ActiveGame.create()
            survivors.append(replacement)
            ready.append(replacement)
            started_games += 1

        active = survivors
        if not ready:
            continue

        results = mcts.search_many(
            [game.root for game in ready],
            add_root_noise=True,
            rng=rng,
        )
        for game, result in zip(ready, results, strict=True):
            exact_roots += int(result.exact)
            game.records.append(
                _PendingRecord(
                    board=game.board,
                    color=game.color,
                    visits=result.visit_counts.copy(),
                )
            )
            temperature = (
                config.temperature if game.moves < config.temperature_moves else 0.0
            )
            action = select_action(result.visit_counts, rng, temperature)
            game.root = game.root.child_for_action(action)
            game.board = game.root.board
            game.color = opponent_color(game.color)
            game.moves += 1

    elapsed = time.monotonic() - started_at
    return records, SelfPlayMetrics(
        games=completed_games,
        positions=len(records),
        black_wins=black_wins,
        white_wins=white_wins,
        draws=draws,
        exact_roots=exact_roots,
        elapsed_seconds=elapsed,
    )


def _cpu_self_play_worker(
    model_config: AlphaZeroModelConfig,
    state_dict: Mapping[str, Tensor],
    config: TrainingConfig,
    generation: int,
    games: int,
    seed: int,
) -> tuple[list[SelfPlayRecord], SelfPlayMetrics]:
    global _WORKER_MODEL, _WORKER_MODEL_CONFIG

    torch.set_num_threads(config.worker_torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work starts.
        pass
    if _WORKER_MODEL is None or _WORKER_MODEL_CONFIG != model_config:
        _WORKER_MODEL = AlphaZeroNetwork(model_config)
        _WORKER_MODEL_CONFIG = model_config
    model = _WORKER_MODEL
    model.load_state_dict(state_dict)
    return collect_self_play(
        model,
        "cpu",
        replace(config, self_play_workers=1, games_per_generation=games),
        generation,
        games=games,
        seed=seed,
    )


def collect_self_play_parallel(
    model: AlphaZeroNetwork,
    device: torch.device,
    config: TrainingConfig,
    generation: int,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[list[SelfPlayRecord], SelfPlayMetrics]:
    """Use process workers on CPU; retain one batched owner for CUDA."""

    if config.self_play_workers <= 1 or device.type != "cpu":
        if config.self_play_workers > 1 and device.type != "cpu":
            warnings.warn(
                "CUDA self-play uses one model owner with batched parallel games; "
                "self_play_workers is ignored to avoid duplicating CUDA models.",
                stacklevel=2,
            )
        return collect_self_play(model, device, config, generation)

    worker_count = min(config.self_play_workers, config.games_per_generation)
    base, remainder = divmod(config.games_per_generation, worker_count)
    counts = [base + int(index < remainder) for index in range(worker_count)]
    cpu_state = {
        name: tensor.detach().cpu().clone().share_memory_()
        for name, tensor in model.state_dict().items()
    }
    context = mp.get_context("spawn")
    started_at = time.monotonic()
    records: list[SelfPlayRecord] = []
    metrics: list[SelfPlayMetrics] = []
    owns_executor = executor is None
    active_executor = executor or ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    )
    try:
        futures = [
            active_executor.submit(
                _cpu_self_play_worker,
                model.config,
                cpu_state,
                config,
                generation,
                count,
                config.seed + generation * 1_000_003 + index * 10_007,
            )
            for index, count in enumerate(counts)
        ]
        for future in futures:
            worker_records, worker_metrics = future.result()
            records.extend(worker_records)
            metrics.append(worker_metrics)
    finally:
        if owns_executor:
            active_executor.shutdown()
    return records, SelfPlayMetrics(
        games=sum(item.games for item in metrics),
        positions=len(records),
        black_wins=sum(item.black_wins for item in metrics),
        white_wins=sum(item.white_wins for item in metrics),
        draws=sum(item.draws for item in metrics),
        exact_roots=sum(item.exact_roots for item in metrics),
        elapsed_seconds=time.monotonic() - started_at,
    )


def reanalyse_replay(
    model: AlphaZeroNetwork,
    device: torch.device,
    replay: ReplayBuffer,
    config: TrainingConfig,
    generation: int,
    rng: np.random.Generator,
) -> ReanalysisMetrics:
    """Refresh stale MCTS targets with the current network in large batches."""

    indices = replay.sample_reanalysis_indices(
        config.reanalysis_positions,
        generation,
        config.reanalysis_minimum_age,
        rng,
    )
    if not len(indices):
        return ReanalysisMetrics(0, 0, 0.0)

    started_at = time.monotonic()
    evaluator = AlphaZeroEvaluator(
        model,
        device,
        maximum_batch_size=config.inference_batch_size,
        cache_size=config.inference_cache_size,
        mixed_precision=config.mixed_precision,
    )
    mcts = NeuralMCTS(evaluator, _mcts_config(config))
    refreshed: list[np.ndarray] = []
    exact_roots = 0
    for start in range(0, len(indices), config.reanalysis_batch_size):
        chunk = indices[start : start + config.reanalysis_batch_size]
        roots = [MCTSNode(board) for board in replay.boards_at(chunk)]
        results = mcts.search_many(
            roots,
            add_root_noise=False,
            rng=rng,
        )
        refreshed.extend(result.visit_counts for result in results)
        exact_roots += sum(result.exact for result in results)
    replay.update_search_targets(indices, refreshed, generation)
    return ReanalysisMetrics(
        positions=len(indices),
        exact_roots=exact_roots,
        elapsed_seconds=time.monotonic() - started_at,
    )


def _to_device(batch: ReplayBatch, device: torch.device) -> tuple[Tensor, ...]:
    non_blocking = device.type == "cuda"
    states = torch.from_numpy(batch.states).to(
        device,
        non_blocking=non_blocking,
    )
    if device.type in {"cpu", "cuda"}:
        states = states.contiguous(memory_format=torch.channels_last)
    return (
        states,
        states[:, 2].flatten(1).to(dtype=torch.bool),
        torch.from_numpy(batch.policy_targets).to(
            device,
            non_blocking=non_blocking,
        ),
        torch.from_numpy(batch.wdl_targets).to(
            device,
            non_blocking=non_blocking,
        ),
        torch.from_numpy(batch.margin_targets).to(
            device,
            non_blocking=non_blocking,
        ),
        torch.from_numpy(batch.ownership_targets).to(
            device,
            non_blocking=non_blocking,
        ),
    )


def _explained_variance(targets: Tensor, predictions: Tensor) -> float:
    target_variance = torch.var(targets, unbiased=False)
    if float(target_variance) < 1e-12:
        return 0.0
    residual_variance = torch.var(targets - predictions, unbiased=False)
    return float((1.0 - residual_variance / target_variance).item())


@torch.no_grad()
def _update_ema_model(
    ema_model: AlphaZeroNetwork,
    model: AlphaZeroNetwork,
    decay: float,
) -> None:
    if ema_model.config != model.config:
        raise ValueError("EMA model config differs from training model config")
    blend = 1.0 - decay
    for ema_parameter, parameter in zip(
        ema_model.parameters(),
        model.parameters(),
        strict=True,
    ):
        ema_parameter.lerp_(parameter.detach(), blend)
    for ema_buffer, buffer in zip(
        ema_model.buffers(),
        model.buffers(),
        strict=True,
    ):
        ema_buffer.copy_(buffer.detach())


def update_network(
    model: AlphaZeroNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    device: torch.device,
    config: TrainingConfig,
    rng: np.random.Generator,
    scaler: torch.amp.GradScaler | None = None,
    ema_model: AlphaZeroNetwork | None = None,
) -> UpdateMetrics:
    if config.training_steps == 0:
        return UpdateMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    if len(replay) < config.minimum_replay_size:
        return UpdateMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

    amp_enabled = config.mixed_precision and device.type == "cuda"
    active_scaler = scaler or torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )
    totals = {
        "policy": 0.0,
        "wdl": 0.0,
        "margin": 0.0,
        "ownership": 0.0,
        "total": 0.0,
        "gradient": 0.0,
        "accuracy": 0.0,
        "explained": 0.0,
    }
    model.train()

    for _ in range(config.training_steps):
        batch = replay.sample(
            config.batch_size,
            rng,
            augment_symmetry=config.symmetry_augmentation,
        )
        (
            states,
            masks,
            policy_targets,
            wdl_targets,
            margin_targets,
            ownership_targets,
        ) = _to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            policy_logits, wdl_logits, margins, ownership_logits = model(states)
            masked_policy = model.masked_policy_logits(policy_logits, masks)
            policy_per_row = -(
                policy_targets * F.log_softmax(masked_policy.float(), dim=-1)
            ).sum(dim=-1)
            wdl_per_row = F.cross_entropy(
                wdl_logits.float(),
                wdl_targets,
                reduction="none",
            )
            margin_per_row = F.smooth_l1_loss(
                margins.float(),
                margin_targets,
                reduction="none",
            )
            ownership_per_square = F.cross_entropy(
                ownership_logits.float(),
                ownership_targets.reshape(
                    batch.size,
                    BOARD_SIZE,
                    BOARD_SIZE,
                ),
                reduction="none",
            )
            ownership_per_row = ownership_per_square.flatten(1).mean(dim=1)
            total_per_row = (
                policy_per_row
                + wdl_per_row
                + config.margin_coefficient * margin_per_row
                + config.ownership_coefficient * ownership_per_row
            )
            loss = total_per_row.mean()

        active_scaler.scale(loss).backward()
        active_scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm,
        )
        active_scaler.step(optimizer)
        active_scaler.update()
        if ema_model is not None:
            _update_ema_model(ema_model, model, config.ema_decay)

        policy_entropy = -(
            policy_targets
            * torch.log(torch.clamp(policy_targets, min=1e-12))
        ).sum(dim=-1)
        policy_kl = torch.clamp(
            policy_per_row.detach() - policy_entropy,
            min=0.0,
        )
        priorities = (
            (
                policy_kl
                + wdl_per_row.detach()
                + 0.25 * margin_per_row.detach()
            )
            .cpu()
            .numpy()
        )
        replay.update_priorities(batch.indices, priorities)
        with torch.no_grad():
            accuracy = (torch.argmax(wdl_logits, dim=-1) == wdl_targets).float().mean()
            scalar_targets = wdl_targets.float() - 1.0
            wdl_probabilities = torch.softmax(wdl_logits.float(), dim=-1)
            scalar_predictions = wdl_probabilities[:, 2] - wdl_probabilities[:, 0]
            explained = _explained_variance(
                scalar_targets,
                scalar_predictions,
            )

        totals["policy"] += float(policy_per_row.mean().item())
        totals["wdl"] += float(wdl_per_row.mean().item())
        totals["margin"] += float(margin_per_row.mean().item())
        totals["ownership"] += float(ownership_per_row.mean().item())
        totals["total"] += float(loss.item())
        totals["gradient"] += float(gradient_norm)
        totals["accuracy"] += float(accuracy.item())
        totals["explained"] += explained

    divisor = config.training_steps
    model.eval()
    return UpdateMetrics(
        policy_loss=totals["policy"] / divisor,
        wdl_loss=totals["wdl"] / divisor,
        margin_loss=totals["margin"] / divisor,
        ownership_loss=totals["ownership"] / divisor,
        total_loss=totals["total"] / divisor,
        gradient_norm=totals["gradient"] / divisor,
        wdl_accuracy=totals["accuracy"] / divisor,
        explained_variance=totals["explained"] / divisor,
        updates=divisor,
    )


def _random_opening(
    rng: np.random.Generator,
    maximum_plies: int,
) -> tuple[BitBoard, int]:
    board = BitBoard.initial()
    color = BLACK
    target = int(rng.integers(0, maximum_plies + 1)) if maximum_plies else 0
    moves = 0
    while moves < target:
        actions = board.legal_actions()
        if not actions:
            passed = board.pass_turn()
            if not passed.legal_moves_bits():
                break
            board = passed
            color = opponent_color(color)
            continue
        board = board.play(int(rng.choice(actions)))
        color = opponent_color(color)
        moves += 1
    return board, color


@dataclass(slots=True)
class _EvaluationState:
    board: BitBoard
    color: int
    candidate_black: bool
    result: tuple[float, float] | None = None


def _play_evaluation_games(
    games: list[_EvaluationState],
    candidate: NeuralMCTS,
    champion: NeuralMCTS,
) -> list[tuple[float, float]]:
    """Play an arena in lockstep so both networks receive large leaf batches."""

    unfinished = list(games)
    while unfinished:
        candidate_games: list[_EvaluationState] = []
        champion_games: list[_EvaluationState] = []
        survivors: list[_EvaluationState] = []
        for game in unfinished:
            while not game.board.legal_moves_bits():
                passed = game.board.pass_turn()
                if not passed.legal_moves_bits():
                    black, white = _absolute_bitboards(game.board, game.color)
                    difference = black.bit_count() - white.bit_count()
                    candidate_difference = (
                        difference if game.candidate_black else -difference
                    )
                    score = (
                        1.0
                        if candidate_difference > 0
                        else 0.0
                        if candidate_difference < 0
                        else 0.5
                    )
                    game.result = (score, candidate_difference / 64.0)
                    break
                game.board = passed
                game.color = opponent_color(game.color)
            if game.result is not None:
                continue
            survivors.append(game)
            candidate_turn = (game.color == BLACK) == game.candidate_black
            (candidate_games if candidate_turn else champion_games).append(game)

        for search, ready in (
            (candidate, candidate_games),
            (champion, champion_games),
        ):
            if not ready:
                continue
            results = search.search_many(
                [MCTSNode(game.board) for game in ready],
                add_root_noise=False,
            )
            for game, result in zip(ready, results, strict=True):
                action = int(np.argmax(result.visit_counts))
                game.board = game.board.play(action)
                game.color = opponent_color(game.color)
        unfinished = survivors

    return [
        game.result
        for game in games
        if game.result is not None
    ]


def evaluate_candidate(
    candidate_model: AlphaZeroNetwork,
    champion_model: AlphaZeroNetwork,
    device: torch.device,
    config: TrainingConfig,
    generation: int,
) -> dict[str, float]:
    candidate_evaluator = AlphaZeroEvaluator(
        candidate_model,
        device,
        maximum_batch_size=config.inference_batch_size,
        cache_size=config.inference_cache_size,
        mixed_precision=config.mixed_precision,
    )
    champion_evaluator = AlphaZeroEvaluator(
        champion_model,
        device,
        maximum_batch_size=config.inference_batch_size,
        cache_size=config.inference_cache_size,
        mixed_precision=config.mixed_precision,
    )
    search_config = _mcts_config(config, config.evaluation_simulations)
    candidate = NeuralMCTS(candidate_evaluator, search_config)
    champion = NeuralMCTS(
        champion_evaluator,
        search_config,
        exact_solver=candidate.exact_solver,
    )
    rng = np.random.default_rng(config.seed + 90_000_019 + generation)
    games: list[_EvaluationState] = []
    started_at = time.monotonic()
    for _ in range(config.evaluation_pairs):
        opening, color = _random_opening(
            rng,
            config.evaluation_opening_plies,
        )
        games.append(
            _EvaluationState(
                opening,
                color,
                candidate_black=True,
            )
        )
        games.append(
            _EvaluationState(
                opening,
                color,
                candidate_black=False,
            )
        )
    results = _play_evaluation_games(games, candidate, champion)
    scores = [score for score, _ in results]
    margins = [margin for _, margin in results]
    return {
        "evaluation_score": float(np.mean(scores)),
        "evaluation_margin": float(np.mean(margins)),
        "evaluation_games": float(len(scores)),
        "evaluation_seconds": time.monotonic() - started_at,
    }


def _learning_rate_factor(config: TrainingConfig, generation: int) -> float:
    progress = min(max(generation / max(config.generations, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = config.minimum_learning_rate_fraction
    return floor + (1.0 - floor) * cosine


def _config_payload(config: TrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_directory"] = str(config.output_directory)
    payload["resume"] = None if config.resume is None else str(config.resume)
    return payload


def _training_state(
    generation: int,
    total_games: int,
    total_positions: int,
    rng: np.random.Generator,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "generation": generation,
        "total_games": total_games,
        "total_positions": total_positions,
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.random.get_rng_state(),
        "scaler_state": scaler.state_dict(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    return state


def _restore_training_rng(
    state: Mapping[str, Any],
    rng: np.random.Generator,
    scaler: torch.amp.GradScaler,
) -> None:
    if "numpy_rng_state" in state:
        rng.bit_generator.state = state["numpy_rng_state"]
    if "torch_rng_state" in state:
        torch.random.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_states" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    if "scaler_state" in state:
        scaler.load_state_dict(state["scaler_state"])


def _append_metrics(path: Path, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(metrics), sort_keys=True) + "\n")


def train(config: TrainingConfig) -> Path:
    config.validate()
    output = config.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_directory = output / "snapshots"
    latest_path = output / "latest.az"
    best_path = output / "best.az"
    replay_path = output / "replay.npz"
    metrics_path = output / "metrics.jsonl"
    device = resolve_device(config.device)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(config.seed)
    amp_enabled = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_generation = 0
    total_games = 0
    total_positions = 0
    resume_payload: Mapping[str, Any] = {}
    if config.resume is not None:
        model, resume_payload = load_checkpoint(config.resume, device)
        saved_state = resume_payload.get("training_state", {})
        start_generation = int(saved_state.get("generation", 0))
        total_games = int(saved_state.get("total_games", 0))
        total_positions = int(saved_state.get("total_positions", 0))
        _restore_training_rng(saved_state, rng, scaler)
    else:
        model = AlphaZeroNetwork(
            AlphaZeroModelConfig(
                config.channels,
                config.residual_blocks,
                config.value_hidden,
            )
        )
        prepare_model(model, device)

    ema_model = AlphaZeroNetwork(model.config)
    ema_state = resume_payload.get("ema_model_state", model.state_dict())
    try:
        ema_model.load_state_dict(ema_state)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("Invalid EMA weights in AlphaZero checkpoint") from exc
    prepare_model(ema_model, device)
    ema_model.requires_grad_(False)
    ema_model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _learning_rate_factor(config, step),
    )
    if "optimizer_state" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer_state"])
    if "scheduler_state" in resume_payload:
        scheduler.load_state_dict(resume_payload["scheduler_state"])

    replay = ReplayBuffer(
        config.replay_capacity,
        recent_fraction=config.replay_recent_fraction,
        prioritized_fraction=config.replay_prioritized_fraction,
        recent_generations=config.replay_recent_generations,
        priority_alpha=config.replay_priority_alpha,
    )
    if config.resume is not None and replay_path.is_file():
        replay.load(replay_path)
    elif config.resume is not None:
        warnings.warn(
            f"Replay file {replay_path} is unavailable; resume will warm a new buffer.",
            stacklevel=2,
        )

    if start_generation >= config.generations:
        if latest_path.is_file():
            return latest_path
        return Path(config.resume).expanduser().resolve()

    print(
        f"Training AlphaZero on {device} from generation {start_generation + 1} "
        f"to {config.generations}; {config.games_per_generation} games/generation, "
        f"{config.simulations} simulations/move, {config.parallel_games} batched "
        f"games, {config.self_play_workers} self-play worker(s)."
    )

    latest_metrics: dict[str, Any] = {}
    self_play_executor = None
    if device.type == "cpu" and config.self_play_workers > 1:
        self_play_executor = ProcessPoolExecutor(
            max_workers=min(
                config.self_play_workers,
                config.games_per_generation,
            ),
            mp_context=mp.get_context("spawn"),
        )
    try:
        for generation_index in range(start_generation, config.generations):
            generation = generation_index + 1
            self_play_records, self_play = collect_self_play_parallel(
                ema_model,
                device,
                config,
                generation,
                self_play_executor,
            )
            replay.add(self_play_records)
            total_games += self_play.games
            total_positions += self_play.positions

            reanalysis = reanalyse_replay(
                ema_model,
                device,
                replay,
                config,
                generation,
                rng,
            )
            update_started = time.monotonic()
            update = update_network(
                model,
                optimizer,
                replay,
                device,
                config,
                rng,
                scaler,
                ema_model,
            )
            update_seconds = time.monotonic() - update_started
            scheduler.step()

            metrics: dict[str, Any] = {
                "generation": generation,
                "total_games": total_games,
                "total_positions": total_positions,
                "replay_size": len(replay),
                "self_play_games": self_play.games,
                "self_play_positions": self_play.positions,
                "self_play_black_wins": self_play.black_wins,
                "self_play_white_wins": self_play.white_wins,
                "self_play_draws": self_play.draws,
                "self_play_exact_roots": self_play.exact_roots,
                "self_play_seconds": self_play.elapsed_seconds,
                "positions_per_second": (
                    self_play.positions / max(self_play.elapsed_seconds, 1e-9)
                ),
                "reanalysis_positions": reanalysis.positions,
                "reanalysis_exact_roots": reanalysis.exact_roots,
                "reanalysis_seconds": reanalysis.elapsed_seconds,
                "policy_loss": update.policy_loss,
                "wdl_loss": update.wdl_loss,
                "margin_loss": update.margin_loss,
                "ownership_loss": update.ownership_loss,
                "total_loss": update.total_loss,
                "gradient_norm": update.gradient_norm,
                "wdl_accuracy": update.wdl_accuracy,
                "explained_variance": update.explained_variance,
                "updates": update.updates,
                "update_seconds": update_seconds,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }

            if config.evaluation_every and generation % config.evaluation_every == 0:
                if best_path.is_file():
                    champion_model, _ = load_checkpoint(
                        best_path,
                        device,
                        use_ema=True,
                    )
                    evaluation = evaluate_candidate(
                        ema_model,
                        champion_model,
                        device,
                        config,
                        generation,
                    )
                    metrics.update(evaluation)
                    if evaluation["evaluation_score"] >= config.promotion_score:
                        metrics["promoted"] = True
                        save_checkpoint(
                            best_path,
                            model,
                            ema_model=ema_model,
                            training_config=_config_payload(config),
                            training_state=_training_state(
                                generation,
                                total_games,
                                total_positions,
                                rng,
                                scaler,
                            ),
                            metrics=metrics,
                        )
                    else:
                        metrics["promoted"] = False
                else:
                    metrics["promoted"] = True
                    save_checkpoint(
                        best_path,
                        model,
                        ema_model=ema_model,
                        training_config=_config_payload(config),
                        training_state=_training_state(
                            generation,
                            total_games,
                            total_positions,
                            rng,
                            scaler,
                        ),
                        metrics=metrics,
                    )

            latest_metrics = metrics
            state = _training_state(
                generation,
                total_games,
                total_positions,
                rng,
                scaler,
            )
            should_checkpoint = (
                config.checkpoint_every and generation % config.checkpoint_every == 0
            )
            if should_checkpoint or generation == config.generations:
                save_checkpoint(
                    latest_path,
                    model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_config=_config_payload(config),
                    training_state=state,
                    metrics=metrics,
                )
            if config.snapshot_every and generation % config.snapshot_every == 0:
                save_checkpoint(
                    snapshot_directory / f"alphazero_gen_{generation:05d}.az",
                    model,
                    ema_model=ema_model,
                    training_config=_config_payload(config),
                    training_state=state,
                    metrics=metrics,
                )
            if (
                config.replay_save_every and generation % config.replay_save_every == 0
            ) or generation == config.generations:
                replay.save(
                    replay_path,
                    compressed=config.compress_replay,
                )
            _append_metrics(metrics_path, metrics)

            evaluation_text = (
                f", eval={metrics['evaluation_score']:.3f}"
                if "evaluation_score" in metrics
                else ""
            )
            print(
                f"Generation {generation}/{config.generations}: "
                f"{self_play.games} games, {self_play.positions} positions "
                f"({metrics['positions_per_second']:.1f}/s), replay={len(replay)}, "
                f"policy={update.policy_loss:.4f}, wdl={update.wdl_loss:.4f}, "
                f"accuracy={update.wdl_accuracy:.3f}, "
                f"EV={update.explained_variance:+.3f}{evaluation_text}"
            )
    except KeyboardInterrupt:
        state = _training_state(
            max(start_generation, int(latest_metrics.get("generation", 0))),
            total_games,
            total_positions,
            rng,
            scaler,
        )
        save_checkpoint(
            latest_path,
            model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_config=_config_payload(config),
            training_state=state,
            metrics=latest_metrics,
        )
        replay.save(
            replay_path,
            compressed=config.compress_replay,
        )
        print(f"Training interrupted; saved {latest_path} and {replay_path}.")
        return latest_path
    finally:
        if self_play_executor is not None:
            self_play_executor.shutdown()

    if not best_path.is_file():
        save_checkpoint(
            best_path,
            model,
            ema_model=ema_model,
            training_config=_config_payload(config),
            training_state=_training_state(
                config.generations,
                total_games,
                total_positions,
                rng,
                scaler,
            ),
            metrics=latest_metrics,
        )
    return latest_path


def build_parser() -> argparse.ArgumentParser:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        description="Train a neural MCTS AlphaZero Othello model without WTHOR."
    )
    parser.add_argument("--generations", type=int, default=defaults.generations)
    parser.add_argument(
        "--games-per-generation",
        type=int,
        default=defaults.games_per_generation,
    )
    parser.add_argument("--simulations", type=int, default=defaults.simulations)
    parser.add_argument(
        "--parallel-games",
        type=int,
        default=defaults.parallel_games,
    )
    parser.add_argument(
        "--self-play-workers",
        type=int,
        default=defaults.self_play_workers,
        help="spawned CPU workers; CUDA always uses one batched model owner",
    )
    parser.add_argument(
        "--worker-torch-threads",
        type=int,
        default=defaults.worker_torch_threads,
    )
    parser.add_argument("--channels", type=int, default=defaults.channels)
    parser.add_argument("--blocks", type=int, default=defaults.residual_blocks)
    parser.add_argument(
        "--value-hidden",
        type=int,
        default=defaults.value_hidden,
    )
    parser.add_argument("--c-puct", type=float, default=defaults.c_puct)
    parser.add_argument(
        "--fpu-reduction",
        type=float,
        default=defaults.fpu_reduction,
    )
    parser.add_argument(
        "--leaf-batch-size",
        type=int,
        default=defaults.leaf_batch_size,
        help="virtual-loss MCTS leaves selected per root and inference wave",
    )
    parser.add_argument(
        "--virtual-loss",
        type=float,
        default=defaults.virtual_loss,
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=defaults.dirichlet_alpha,
    )
    parser.add_argument(
        "--dirichlet-fraction",
        type=float,
        default=defaults.dirichlet_fraction,
    )
    parser.add_argument(
        "--exact-endgame-empties",
        type=int,
        default=defaults.exact_endgame_empties,
    )
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument(
        "--temperature-moves",
        type=int,
        default=defaults.temperature_moves,
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=defaults.inference_batch_size,
    )
    parser.add_argument(
        "--inference-cache-size",
        type=int,
        default=defaults.inference_cache_size,
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=defaults.replay_capacity,
    )
    parser.add_argument(
        "--minimum-replay-size",
        type=int,
        default=defaults.minimum_replay_size,
    )
    parser.add_argument(
        "--reanalysis-positions",
        type=int,
        default=defaults.reanalysis_positions,
    )
    parser.add_argument(
        "--reanalysis-minimum-age",
        type=int,
        default=defaults.reanalysis_minimum_age,
    )
    parser.add_argument(
        "--reanalysis-batch-size",
        type=int,
        default=defaults.reanalysis_batch_size,
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--training-steps",
        type=int,
        default=defaults.training_steps,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults.learning_rate,
    )
    parser.add_argument(
        "--minimum-learning-rate-fraction",
        type=float,
        default=defaults.minimum_learning_rate_fraction,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=defaults.weight_decay,
    )
    parser.add_argument(
        "--margin-coefficient",
        type=float,
        default=defaults.margin_coefficient,
    )
    parser.add_argument(
        "--ownership-coefficient",
        type=float,
        default=defaults.ownership_coefficient,
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=defaults.ema_decay,
        help="weight EMA used for self-play, arena evaluation, and deployment",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=defaults.max_grad_norm,
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="disable CUDA autocast and gradient scaling",
    )
    parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="disable random rotation/reflection augmentation",
    )
    parser.add_argument(
        "--evaluation-every",
        type=int,
        default=defaults.evaluation_every,
    )
    parser.add_argument(
        "--evaluation-pairs",
        type=int,
        default=defaults.evaluation_pairs,
    )
    parser.add_argument(
        "--evaluation-simulations",
        type=int,
        default=defaults.evaluation_simulations,
    )
    parser.add_argument(
        "--evaluation-opening-plies",
        type=int,
        default=defaults.evaluation_opening_plies,
    )
    parser.add_argument(
        "--promotion-score",
        type=float,
        default=defaults.promotion_score,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=defaults.checkpoint_every,
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=defaults.snapshot_every,
    )
    parser.add_argument(
        "--replay-save-every",
        type=int,
        default=defaults.replay_save_every,
    )
    parser.add_argument(
        "--compress-replay",
        action="store_true",
        help="use smaller but slower compressed replay checkpoints",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=defaults.output_directory,
    )
    parser.add_argument("--resume", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        generations=args.generations,
        games_per_generation=args.games_per_generation,
        simulations=args.simulations,
        parallel_games=args.parallel_games,
        self_play_workers=args.self_play_workers,
        worker_torch_threads=args.worker_torch_threads,
        channels=args.channels,
        residual_blocks=args.blocks,
        value_hidden=args.value_hidden,
        c_puct=args.c_puct,
        fpu_reduction=args.fpu_reduction,
        leaf_batch_size=args.leaf_batch_size,
        virtual_loss=args.virtual_loss,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_fraction=args.dirichlet_fraction,
        exact_endgame_empties=args.exact_endgame_empties,
        temperature=args.temperature,
        temperature_moves=args.temperature_moves,
        inference_batch_size=args.inference_batch_size,
        inference_cache_size=args.inference_cache_size,
        replay_capacity=args.replay_capacity,
        minimum_replay_size=args.minimum_replay_size,
        reanalysis_positions=args.reanalysis_positions,
        reanalysis_minimum_age=args.reanalysis_minimum_age,
        reanalysis_batch_size=args.reanalysis_batch_size,
        batch_size=args.batch_size,
        training_steps=args.training_steps,
        learning_rate=args.learning_rate,
        minimum_learning_rate_fraction=args.minimum_learning_rate_fraction,
        weight_decay=args.weight_decay,
        margin_coefficient=args.margin_coefficient,
        ownership_coefficient=args.ownership_coefficient,
        ema_decay=args.ema_decay,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=not args.no_mixed_precision,
        symmetry_augmentation=not args.no_symmetry,
        evaluation_every=args.evaluation_every,
        evaluation_pairs=args.evaluation_pairs,
        evaluation_simulations=args.evaluation_simulations,
        evaluation_opening_plies=args.evaluation_opening_plies,
        promotion_score=args.promotion_score,
        checkpoint_every=args.checkpoint_every,
        snapshot_every=args.snapshot_every,
        replay_save_every=args.replay_save_every,
        compress_replay=args.compress_replay,
        seed=args.seed,
        device=args.device,
        output_directory=args.output_directory,
        resume=args.resume,
    )
    checkpoint = train(config)
    print(f"AlphaZero checkpoint: {checkpoint}")


if __name__ == "__main__":
    mp.freeze_support()
    main()


__all__ = [
    "ReanalysisMetrics",
    "SelfPlayMetrics",
    "TrainingConfig",
    "UpdateMetrics",
    "collect_self_play",
    "collect_self_play_parallel",
    "evaluate_candidate",
    "reanalyse_replay",
    "train",
    "update_network",
]
