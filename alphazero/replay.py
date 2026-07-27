"""Compact prioritized replay storage for AlphaZero self-play records."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .board import (
    ACTION_DIM,
    INVERSE_ACTION_TRANSFORMS,
    BitBoard,
    encode_boards,
    legal_masks,
    transform_bits,
)

INVERSE_TRANSFORM_ARRAY = np.asarray(
    INVERSE_ACTION_TRANSFORMS,
    dtype=np.intp,
)


@dataclass(frozen=True)
class SelfPlayRecord:
    player: int
    opponent: int
    visit_counts: np.ndarray
    outcome: int
    margin: float
    ownership: np.ndarray
    generation: int

    def validate(self) -> None:
        board = BitBoard(self.player, self.opponent)
        visits = np.asarray(self.visit_counts)
        ownership = np.asarray(self.ownership)
        if visits.shape != (ACTION_DIM,):
            raise ValueError(f"Visit counts have shape {visits.shape}, expected (64,)")
        if ownership.shape != (ACTION_DIM,):
            raise ValueError(f"Ownership has shape {ownership.shape}, expected (64,)")
        if self.outcome not in (-1, 0, 1):
            raise ValueError("Outcome must be -1, 0, or 1")
        if not np.isfinite(self.margin) or not -1 <= self.margin <= 1:
            raise ValueError("Normalized disc margin must be in [-1, 1]")
        if not board.legal_moves_bits():
            raise ValueError("Replay record must be a legal decision state")
        if not np.all(np.isfinite(visits)) or np.any(visits < 0):
            raise ValueError("Replay visit counts must be finite and nonnegative")
        if visits.sum() <= 0:
            raise ValueError("Replay record must contain MCTS visits")
        legal = np.zeros(ACTION_DIM, dtype=np.bool_)
        legal[list(board.legal_actions())] = True
        if np.any(visits[~legal] != 0):
            raise ValueError("Replay visit counts include an illegal action")
        if not np.all(np.isfinite(ownership)) or np.any(
            (ownership < 0) | (ownership >= 3)
        ):
            raise ValueError("Ownership labels must be in 0..2")
        if self.generation < 0:
            raise ValueError("Replay generation cannot be negative")


@dataclass(frozen=True)
class ReplayBatch:
    indices: np.ndarray
    states: np.ndarray
    legal_masks: np.ndarray
    policy_targets: np.ndarray
    wdl_targets: np.ndarray
    margin_targets: np.ndarray
    ownership_targets: np.ndarray

    @property
    def size(self) -> int:
        return int(self.indices.shape[0])


class ReplayBuffer:
    """Fixed-size replay ring with atomic, optionally compressed persistence."""

    def __init__(
        self,
        capacity: int,
        *,
        recent_fraction: float = 0.50,
        prioritized_fraction: float = 0.20,
        recent_generations: int = 10,
        priority_alpha: float = 0.60,
    ) -> None:
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        if not 0 <= recent_fraction <= 1:
            raise ValueError("recent_fraction must be in [0, 1]")
        if not 0 <= prioritized_fraction <= 1:
            raise ValueError("prioritized_fraction must be in [0, 1]")
        if recent_fraction + prioritized_fraction > 1:
            raise ValueError("Replay sampling fractions cannot exceed one")
        if recent_generations < 1:
            raise ValueError("recent_generations must be positive")
        if priority_alpha < 0:
            raise ValueError("priority_alpha cannot be negative")

        self.capacity = int(capacity)
        self.recent_fraction = recent_fraction
        self.prioritized_fraction = prioritized_fraction
        self.recent_generations = recent_generations
        self.priority_alpha = priority_alpha
        self.player = np.zeros(capacity, dtype=np.uint64)
        self.opponent = np.zeros(capacity, dtype=np.uint64)
        self.legal_bits = np.zeros(capacity, dtype=np.uint64)
        self.visit_counts = np.zeros(
            (capacity, ACTION_DIM),
            dtype=np.uint16,
        )
        self.outcome = np.zeros(capacity, dtype=np.int8)
        self.margin = np.zeros(capacity, dtype=np.float32)
        self.ownership = np.zeros(
            (capacity, ACTION_DIM),
            dtype=np.int8,
        )
        self.generation = np.zeros(capacity, dtype=np.int32)
        self.priority = np.ones(capacity, dtype=np.float32)
        self.maximum_priority = 1.0
        self._priority_weights = np.zeros(capacity, dtype=np.float64)
        self._priority_tree = np.zeros(capacity + 1, dtype=np.float64)
        self._priority_total = 0.0
        self._pool_version = 0
        self._cached_pool_version = -1
        self._cached_valid = np.empty(0, dtype=np.int64)
        self._cached_recent = np.empty(0, dtype=np.int64)
        self.size = 0
        self.next_index = 0

    def __len__(self) -> int:
        return self.size

    def _valid_indices(self) -> np.ndarray:
        if self.size < self.capacity:
            return np.arange(self.size, dtype=np.int64)
        return np.arange(self.capacity, dtype=np.int64)

    def _ordered_indices(self) -> np.ndarray:
        if self.size < self.capacity:
            return np.arange(self.size, dtype=np.int64)
        return np.concatenate(
            (
                np.arange(self.next_index, self.capacity, dtype=np.int64),
                np.arange(0, self.next_index, dtype=np.int64),
            )
        )

    def _invalidate_pools(self) -> None:
        self._pool_version += 1

    def _set_priority(self, index: int, priority: float) -> None:
        stable_priority = float(priority)
        if not np.isfinite(stable_priority):
            raise ValueError("Replay priorities must be finite")
        stable_priority = max(stable_priority, 1e-3)
        new_weight = stable_priority**self.priority_alpha
        difference = new_weight - self._priority_weights[index]
        self.priority[index] = stable_priority
        self._priority_weights[index] = new_weight
        self._priority_total += difference
        tree_index = index + 1
        while tree_index <= self.capacity:
            self._priority_tree[tree_index] += difference
            tree_index += tree_index & -tree_index
        self.maximum_priority = max(self.maximum_priority, stable_priority)

    def _rebuild_priority_tree(self) -> None:
        self._priority_weights.fill(0.0)
        valid = self._valid_indices()
        self._priority_weights[valid] = np.maximum(
            self.priority[valid],
            1e-3,
        ).astype(np.float64) ** self.priority_alpha
        self._priority_tree.fill(0.0)
        self._priority_tree[1:] = self._priority_weights
        for tree_index in range(1, self.capacity + 1):
            parent = tree_index + (tree_index & -tree_index)
            if parent <= self.capacity:
                self._priority_tree[parent] += self._priority_tree[tree_index]
        self._priority_total = float(self._priority_weights.sum())
        self.maximum_priority = (
            max(1.0, float(self.priority[valid].max()))
            if len(valid)
            else 1.0
        )

    def _sampling_pools(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cached_pool_version != self._pool_version:
            valid = self._valid_indices()
            maximum_generation = int(self.generation[valid].max())
            recent_cutoff = maximum_generation - self.recent_generations + 1
            recent = valid[self.generation[valid] >= recent_cutoff]
            self._cached_valid = valid
            self._cached_recent = recent if len(recent) else valid
            self._cached_pool_version = self._pool_version
        return self._cached_valid, self._cached_recent

    def _sample_priorities(
        self,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        if self._priority_total <= 0:
            return rng.choice(
                self._valid_indices(),
                count,
                replace=True,
            )
        targets = rng.random(count) * self._priority_total
        indices = np.zeros(count, dtype=np.int64)
        step = 1 << (self.capacity.bit_length() - 1)
        while step:
            candidates = indices + step
            in_range = candidates <= self.capacity
            candidate_weights = np.zeros(count, dtype=np.float64)
            candidate_weights[in_range] = self._priority_tree[
                candidates[in_range]
            ]
            advance = in_range & (candidate_weights <= targets)
            targets[advance] -= candidate_weights[advance]
            indices[advance] = candidates[advance]
            step >>= 1
        return np.minimum(indices, self.capacity - 1)

    def add(self, records: Sequence[SelfPlayRecord]) -> None:
        for record in records:
            record.validate()
            index = self.next_index
            self.player[index] = record.player
            self.opponent[index] = record.opponent
            self.legal_bits[index] = BitBoard(
                record.player,
                record.opponent,
            ).legal_moves_bits()
            visits = np.asarray(record.visit_counts, dtype=np.int64)
            self.visit_counts[index] = np.clip(visits, 0, 65_535).astype(np.uint16)
            self.outcome[index] = record.outcome
            self.margin[index] = record.margin
            self.ownership[index] = np.asarray(record.ownership, dtype=np.int8)
            self.generation[index] = record.generation
            self._set_priority(index, self.maximum_priority)
            self.next_index = (index + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        if records:
            self._invalidate_pools()

    def _sample_indices(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.size == 0:
            raise ValueError("Cannot sample an empty replay buffer")
        valid, recent_pool = self._sampling_pools()
        fractions = np.asarray(
            (
                self.recent_fraction,
                self.prioritized_fraction,
                1.0 - self.recent_fraction - self.prioritized_fraction,
            ),
            dtype=np.float64,
        )
        exact_counts = fractions * batch_size
        counts = np.floor(exact_counts).astype(np.int64)
        remainder = batch_size - int(counts.sum())
        if remainder:
            fractional = exact_counts - counts
            order = np.argsort(-fractional, kind="stable")
            counts[order[:remainder]] += 1
        recent_count, priority_count, uniform_count = map(int, counts)

        sampled: list[np.ndarray] = []
        if recent_count:
            sampled.append(rng.choice(recent_pool, recent_count, replace=True))
        if priority_count:
            sampled.append(self._sample_priorities(priority_count, rng))
        if uniform_count:
            sampled.append(rng.choice(valid, uniform_count, replace=True))
        indices = np.concatenate(sampled).astype(np.int64, copy=False)
        rng.shuffle(indices)
        return indices

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        augment_symmetry: bool = True,
    ) -> ReplayBatch:
        if batch_size < 1:
            raise ValueError("Replay batch size must be positive")
        indices = self._sample_indices(batch_size, rng)
        if augment_symmetry:
            symmetries = rng.integers(0, 8, size=batch_size, dtype=np.intp)
        else:
            symmetries = np.zeros(batch_size, dtype=np.intp)
        inverse_mappings = INVERSE_TRANSFORM_ARRAY[symmetries]
        visits = np.take_along_axis(
            self.visit_counts[indices],
            inverse_mappings,
            axis=1,
        ).astype(np.float32)
        ownership = np.take_along_axis(
            self.ownership[indices],
            inverse_mappings,
            axis=1,
        ).astype(np.int64)
        boards = [
            BitBoard(
                transform_bits(int(self.player[index]), int(symmetry)),
                transform_bits(int(self.opponent[index]), int(symmetry)),
            )
            for index, symmetry in zip(indices, symmetries, strict=True)
        ]
        for board, index, symmetry in zip(
            boards,
            indices,
            symmetries,
            strict=True,
        ):
            object.__setattr__(
                board,
                "_legal_bits",
                transform_bits(int(self.legal_bits[index]), int(symmetry)),
            )

        totals = visits.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise RuntimeError("Replay contains a record without MCTS visits")
        policy_targets = visits / totals
        return ReplayBatch(
            indices=indices,
            states=encode_boards(boards),
            legal_masks=legal_masks(boards),
            policy_targets=policy_targets,
            wdl_targets=(self.outcome[indices].astype(np.int64) + 1),
            margin_targets=self.margin[indices].astype(np.float32, copy=True),
            ownership_targets=ownership,
        )

    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float32)
        if indices.shape != priorities.shape:
            raise ValueError("Priority indices and values must have equal shapes")
        if np.any((indices < 0) | (indices >= self.size)):
            raise IndexError("Replay priority index is outside the valid buffer")
        if not np.all(np.isfinite(priorities)):
            raise ValueError("Replay priorities must be finite")

        # A minibatch may sample the same replay row more than once with
        # different symmetries. Aggregate it so priority is order-independent.
        unique, inverse = np.unique(indices, return_inverse=True)
        totals = np.zeros(len(unique), dtype=np.float64)
        occurrences = np.zeros(len(unique), dtype=np.int64)
        np.add.at(totals, inverse, priorities)
        np.add.at(occurrences, inverse, 1)
        aggregated = totals / occurrences
        for index, priority in zip(unique, aggregated, strict=True):
            self._set_priority(int(index), float(priority))

    def sample_reanalysis_indices(
        self,
        count: int,
        current_generation: int,
        minimum_age: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Choose old, high-error positions whose search target may be stale."""

        if count <= 0 or self.size == 0:
            return np.empty(0, dtype=np.int64)
        valid = self._valid_indices()
        cutoff = current_generation - minimum_age
        eligible = valid[self.generation[valid] <= cutoff]
        if not eligible.size:
            return np.empty(0, dtype=np.int64)
        selected_count = min(count, len(eligible))
        weights = np.maximum(self.priority[eligible], 1e-3).astype(np.float64)
        weights **= self.priority_alpha
        weights /= weights.sum()
        return rng.choice(
            eligible,
            selected_count,
            replace=False,
            p=weights,
        ).astype(np.int64, copy=False)

    def boards_at(self, indices: np.ndarray) -> list[BitBoard]:
        return [
            BitBoard(int(self.player[index]), int(self.opponent[index]))
            for index in np.asarray(indices, dtype=np.int64)
        ]

    def update_search_targets(
        self,
        indices: np.ndarray,
        visit_counts: Sequence[np.ndarray],
        generation: int,
    ) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) != len(visit_counts):
            raise ValueError("Reanalysis indices and visit targets must match")
        for index, visits in zip(indices, visit_counts, strict=True):
            counts = np.asarray(visits, dtype=np.int64)
            if counts.shape != (ACTION_DIM,) or counts.sum() <= 0:
                raise ValueError("Reanalysis visit target must be a nonempty (64,) row")
            self.visit_counts[index] = np.clip(counts, 0, 65_535).astype(np.uint16)
            self.generation[index] = generation
            self._set_priority(
                int(index),
                max(float(self.priority[index]), 1.0),
            )
        if len(indices):
            self._invalidate_pools()

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        compressed: bool = False,
    ) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        ordered = self._ordered_indices()
        save_archive = np.savez_compressed if compressed else np.savez
        with temporary.open("wb") as stream:
            save_archive(
                stream,
                player=self.player[ordered],
                opponent=self.opponent[ordered],
                legal_bits=self.legal_bits[ordered],
                visit_counts=self.visit_counts[ordered],
                outcome=self.outcome[ordered],
                margin=self.margin[ordered],
                ownership=self.ownership[ordered],
                generation=self.generation[ordered],
                priority=self.priority[ordered],
            )
        os.replace(temporary, destination)
        return destination

    def load(self, path: str | os.PathLike[str]) -> None:
        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as payload:
            required = {
                "player",
                "opponent",
                "visit_counts",
                "outcome",
                "margin",
                "ownership",
                "generation",
                "priority",
            }
            if not required.issubset(payload.files):
                missing = sorted(required.difference(payload.files))
                raise ValueError(f"Replay file is missing arrays: {missing}")
            total = len(payload["player"])
            start = max(0, total - self.capacity)
            count = min(total, self.capacity)
            self.player[:count] = payload["player"][start:]
            self.opponent[:count] = payload["opponent"][start:]
            if "legal_bits" in payload.files:
                self.legal_bits[:count] = payload["legal_bits"][start:]
            else:
                # Backward-compatible one-time upgrade for existing replay
                # archives. Decision records are guaranteed to have moves.
                self.legal_bits[:count] = np.fromiter(
                    (
                        BitBoard(int(player), int(opponent)).legal_moves_bits()
                        for player, opponent in zip(
                            self.player[:count],
                            self.opponent[:count],
                            strict=True,
                        )
                    ),
                    dtype=np.uint64,
                    count=count,
                )
            self.visit_counts[:count] = payload["visit_counts"][start:]
            self.outcome[:count] = payload["outcome"][start:]
            self.margin[:count] = payload["margin"][start:]
            self.ownership[:count] = payload["ownership"][start:]
            self.generation[:count] = payload["generation"][start:]
            self.priority[:count] = payload["priority"][start:]
        self.size = count
        self.next_index = count % self.capacity
        self._rebuild_priority_tree()
        self._invalidate_pools()


__all__ = [
    "ReplayBatch",
    "ReplayBuffer",
    "SelfPlayRecord",
]
