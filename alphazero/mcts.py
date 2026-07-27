"""Batched policy/value-guided Monte Carlo tree search for Othello."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Protocol

import numpy as np

from .board import (
    ACTION_DIM,
    CORNERS,
    FULL_BOARD,
    NOT_FILE_A,
    NOT_FILE_H,
    BitBoard,
)


class Evaluator(Protocol):
    def evaluate(
        self,
        boards: Sequence[BitBoard],
    ) -> list[tuple[np.ndarray, float]]: ...


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 96
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.30
    dirichlet_fraction: float = 0.25
    exact_endgame_empties: int = 8
    fpu_reduction: float = 0.20
    leaf_batch_size: int = 8
    virtual_loss: float = 1.0
    exact_cache_size: int = 500_000

    def validate(self) -> None:
        if self.simulations < 1:
            raise ValueError("MCTS simulations must be positive")
        if self.c_puct <= 0:
            raise ValueError("c_puct must be positive")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if not 0 <= self.dirichlet_fraction <= 1:
            raise ValueError("dirichlet_fraction must be in [0, 1]")
        if self.exact_endgame_empties < 0:
            raise ValueError("exact_endgame_empties cannot be negative")
        if self.fpu_reduction < 0:
            raise ValueError("fpu_reduction cannot be negative")
        if self.leaf_batch_size < 1:
            raise ValueError("leaf_batch_size must be positive")
        if self.virtual_loss < 0:
            raise ValueError("virtual_loss cannot be negative")
        if self.exact_cache_size < 0:
            raise ValueError("exact_cache_size cannot be negative")


@dataclass(slots=True)
class MCTSNode:
    board: BitBoard
    expanded: bool = False
    network_value: float = 0.0
    base_priors: np.ndarray | None = None
    priors: np.ndarray | None = None
    visit_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(ACTION_DIM, dtype=np.int32)
    )
    value_sums: np.ndarray = field(
        default_factory=lambda: np.zeros(ACTION_DIM, dtype=np.float32)
    )
    legal_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.intp)
    )
    total_visits: int = 0
    children: dict[int, MCTSNode] = field(default_factory=dict)
    pass_child: MCTSNode | None = None

    def child_for_action(self, action: int) -> MCTSNode:
        child = self.children.get(action)
        if child is None:
            child = MCTSNode(self.board.play(action))
            self.children[action] = child
        return child

    def child_for_pass(self) -> MCTSNode:
        if self.pass_child is None:
            self.pass_child = MCTSNode(self.board.pass_turn())
        return self.pass_child

    def q_values(self) -> np.ndarray:
        return np.divide(
            self.value_sums,
            self.visit_counts,
            out=np.zeros(ACTION_DIM, dtype=np.float32),
            where=self.visit_counts > 0,
        )


@dataclass(frozen=True)
class SearchResult:
    visit_counts: np.ndarray
    policy: np.ndarray
    root_value: float
    exact: bool


@dataclass(frozen=True)
class _TTEntry:
    value: int
    flag: int
    best_action: int | None = None


class ExactSolver:
    """Negamax alpha-beta solver returning exact optimal disc margin."""

    EXACT = 0
    LOWER = 1
    UPPER = 2

    def __init__(self, maximum_cache_size: int = 500_000) -> None:
        self.maximum_cache_size = maximum_cache_size
        self.cache: dict[tuple[int, int], _TTEntry] = {}

    def clear(self) -> None:
        self.cache.clear()

    @staticmethod
    def _odd_region_mask(empty: int) -> int:
        """Return empty squares belonging to odd orthogonal regions."""

        remaining = empty
        odd_regions = 0
        while remaining:
            frontier = remaining & -remaining
            component = frontier
            while frontier:
                neighbours = (
                    ((frontier & NOT_FILE_H) << 1)
                    | ((frontier & NOT_FILE_A) >> 1)
                    | ((frontier << 8) & FULL_BOARD)
                    | (frontier >> 8)
                )
                frontier = neighbours & remaining & ~component
                component |= frontier
            if component.bit_count() & 1:
                odd_regions |= component
            remaining &= ~component
        return odd_regions

    @classmethod
    def _ordered_moves(
        cls,
        board: BitBoard,
        preferred_action: int | None = None,
    ) -> list[tuple[int, BitBoard]]:
        """Order endgame moves by TT hint, corner, parity, and mobility."""

        odd_regions = cls._odd_region_mask(board.empty)
        scored: list[tuple[tuple[int, ...], int, BitBoard]] = []
        for action, flips in board.legal_moves_with_flips():
            child = board._play_with_flips_unchecked(action, flips)
            score = (
                int(action == preferred_action),
                int(action in CORNERS),
                int(bool((1 << action) & odd_regions)),
                -child.legal_moves_bits().bit_count(),
                flips.bit_count(),
                -action,
            )
            scored.append((score, action, child))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [(action, child) for _, action, child in scored]

    @classmethod
    def _ordered_actions(cls, board: BitBoard) -> list[int]:
        return [action for action, _ in cls._ordered_moves(board)]

    def _store(
        self,
        key: tuple[int, int],
        entry: _TTEntry,
    ) -> None:
        if not self.maximum_cache_size:
            return
        if key not in self.cache and len(self.cache) >= self.maximum_cache_size:
            # Clearing the full table causes a severe re-search cliff near the
            # endgame threshold. Evict a small insertion-ordered tranche.
            eviction_count = max(1, self.maximum_cache_size // 16)
            for stale_key in tuple(islice(self.cache, eviction_count)):
                self.cache.pop(stale_key, None)
        self.cache[key] = entry

    def solve(self, board: BitBoard, alpha: int = -65, beta: int = 65) -> int:
        key = (board.player, board.opponent)
        original_alpha = alpha
        original_beta = beta
        cached = self.cache.get(key)
        preferred_action = None
        if cached is not None:
            preferred_action = cached.best_action
            if cached.flag == self.EXACT:
                return cached.value
            if cached.flag == self.LOWER:
                alpha = max(alpha, cached.value)
            else:
                beta = min(beta, cached.value)
            if alpha >= beta:
                return cached.value

        moves = self._ordered_moves(board, preferred_action)
        best_action = None
        if not moves:
            passed = board.pass_turn()
            if not passed.legal_moves_bits():
                value = board.disc_margin
            else:
                value = -self.solve(passed, -beta, -alpha)
        else:
            value = -65
            for action, child in moves:
                child_value = -self.solve(child, -beta, -alpha)
                if child_value > value:
                    value = child_value
                    best_action = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break

        if value <= original_alpha:
            flag = self.UPPER
        elif value >= original_beta:
            flag = self.LOWER
        else:
            flag = self.EXACT
        self._store(key, _TTEntry(value, flag, best_action))
        return value

    def best_actions(self, board: BitBoard) -> tuple[tuple[int, ...], int]:
        moves = self._ordered_moves(board)
        if not moves:
            raise ValueError("Exact best_actions requires a legal move")
        values = {action: -self.solve(child) for action, child in moves}
        best_value = max(values.values())
        return (
            tuple(action for action, _ in moves if values[action] == best_value),
            best_value,
        )


@dataclass(slots=True)
class _SelectedLeaf:
    leaf: MCTSNode
    path: list[tuple[MCTSNode, int | None]]
    value: float | None


class NeuralMCTS:
    """PUCT search whose leaf evaluations are batched across independent roots."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: MCTSConfig | None = None,
        exact_solver: ExactSolver | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.config.validate()
        self.exact_solver = exact_solver or ExactSolver(self.config.exact_cache_size)

    def _expand(
        self,
        node: MCTSNode,
        probabilities: np.ndarray,
        value: float,
    ) -> None:
        legal_bits = node.board.legal_moves_bits()
        source_priors = np.asarray(probabilities, dtype=np.float64)
        if source_priors.shape != (ACTION_DIM,):
            raise ValueError(
                f"Evaluator policy has shape {source_priors.shape}, expected (64,)"
            )
        legal = np.fromiter(
            node.board.legal_actions(),
            dtype=np.intp,
            count=legal_bits.bit_count(),
        )
        priors = np.zeros(ACTION_DIM, dtype=np.float64)
        legal_priors = source_priors[legal]
        legal_priors = np.where(np.isfinite(legal_priors), legal_priors, 0.0)
        priors[legal] = np.maximum(legal_priors, 0.0)
        total = float(priors.sum())
        legal_count = len(legal)
        if not math.isfinite(total) or total <= 0:
            if legal_count == 0:
                raise ValueError("Cannot expand a node with no legal moves")
            priors[legal] = 1.0 / legal_count
        else:
            priors /= total
        node.legal_indices = legal
        node.priors = priors.astype(np.float32)
        node.base_priors = None
        stable_value = float(value)
        if not math.isfinite(stable_value):
            stable_value = 0.0
        node.network_value = float(np.clip(stable_value, -1.0, 1.0))
        node.expanded = True

    def _set_root_noise(
        self,
        node: MCTSNode,
        add_noise: bool,
        rng: np.random.Generator,
    ) -> None:
        if node.priors is None:
            raise RuntimeError("Cannot add root noise before expanding the node")
        if node.base_priors is not None:
            node.priors = node.base_priors.copy()
        if not add_noise:
            return
        if node.base_priors is None:
            node.base_priors = node.priors.copy()
        actions = node.legal_indices
        if len(actions) <= 1:
            return
        noise = rng.dirichlet(np.full(len(actions), self.config.dirichlet_alpha))
        fraction = self.config.dirichlet_fraction
        for action, amount in zip(actions, noise, strict=True):
            node.priors[action] = (1.0 - fraction) * node.base_priors[
                action
            ] + fraction * float(amount)

    def _select_action(self, node: MCTSNode) -> int:
        parent_visits = max(1, node.total_visits)
        if node.priors is None:
            raise RuntimeError("Cannot select an action before expanding the node")
        exploration_scale = self.config.c_puct * math.sqrt(parent_visits)
        explored_prior = sum(
            float(node.priors[action])
            for action in node.legal_indices
            if node.visit_counts[action] > 0
        )
        first_play_value = max(
            -1.0,
            node.network_value
            - self.config.fpu_reduction * math.sqrt(explored_prior),
        )
        best_action = -1
        best_score = -math.inf
        best_prior = -math.inf
        for action in node.legal_indices:
            action = int(action)
            count = int(node.visit_counts[action])
            q_value = (
                float(node.value_sums[action]) / count
                if count
                else first_play_value
            )
            prior = float(node.priors[action])
            score = q_value + exploration_scale * prior / (1.0 + count)
            if score > best_score or (
                score == best_score and prior > best_prior
            ):
                best_action = action
                best_score = score
                best_prior = prior
        if best_action < 0:
            raise RuntimeError("Cannot select an action from an empty node")
        return best_action

    def _select_leaf(self, root: MCTSNode) -> _SelectedLeaf:
        node = root
        path: list[tuple[MCTSNode, int | None]] = []
        while True:
            legal_bits = node.board.legal_moves_bits()
            if not legal_bits:
                passed = node.board.pass_turn()
                if not passed.legal_moves_bits():
                    value = float(np.sign(node.board.disc_margin))
                    return _SelectedLeaf(node, path, value)
                path.append((node, None))
                node = node.child_for_pass()
                continue

            if (
                self.config.exact_endgame_empties
                and node.board.empty_count <= self.config.exact_endgame_empties
            ):
                margin = self.exact_solver.solve(node.board)
                return _SelectedLeaf(node, path, float(np.sign(margin)))

            if not node.expanded:
                return _SelectedLeaf(node, path, None)

            action = self._select_action(node)
            path.append((node, action))
            node = node.child_for_action(action)

    @staticmethod
    def _backup(
        path: Sequence[tuple[MCTSNode, int | None]],
        leaf_value: float,
    ) -> None:
        value = float(leaf_value)
        for parent, action in reversed(path):
            value = -value
            if action is not None:
                parent.visit_counts[action] += 1
                parent.value_sums[action] += value
                parent.total_visits += 1

    def _reserve(
        self,
        path: Sequence[tuple[MCTSNode, int | None]],
    ) -> None:
        """Apply temporary virtual loss while assembling an inference batch."""

        loss = self.config.virtual_loss
        for parent, action in path:
            if action is not None:
                parent.visit_counts[action] += 1
                parent.value_sums[action] -= loss
                parent.total_visits += 1

    def _release(
        self,
        path: Sequence[tuple[MCTSNode, int | None]],
    ) -> None:
        loss = self.config.virtual_loss
        for parent, action in path:
            if action is not None:
                parent.visit_counts[action] -= 1
                parent.value_sums[action] += loss
                parent.total_visits -= 1

    def _select_batch(
        self,
        root: MCTSNode,
        count: int,
    ) -> tuple[list[_SelectedLeaf], list[Sequence[tuple[MCTSNode, int | None]]]]:
        """Select collision-aware leaves and reserve their paths temporarily."""

        selected: list[_SelectedLeaf] = []
        reservations: list[Sequence[tuple[MCTSNode, int | None]]] = []
        pending_leaves: set[int] = set()
        # Once the current frontier is exhausted, evaluate it instead of
        # repeatedly backing up the same unevaluated leaf.
        maximum_attempts = count + max(count * 2, len(root.legal_indices))
        attempts = 0
        try:
            while len(selected) < count and attempts < maximum_attempts:
                item = self._select_leaf(root)
                self._reserve(item.path)
                reservations.append(item.path)
                attempts += 1
                pending = item.value is None and not item.leaf.expanded
                identity = id(item.leaf)
                if pending and identity in pending_leaves:
                    continue
                if pending:
                    pending_leaves.add(identity)
                selected.append(item)
        except Exception:
            for path in reservations:
                self._release(path)
            raise
        return selected, reservations

    def _evaluate_selected(self, selected: Sequence[_SelectedLeaf]) -> None:
        pending: dict[tuple[int, int], dict[int, MCTSNode]] = {}
        unique_boards: list[BitBoard] = []
        for item in selected:
            if item.value is not None or item.leaf.expanded:
                continue
            key = (item.leaf.board.player, item.leaf.board.opponent)
            if key not in pending:
                pending[key] = {}
                unique_boards.append(item.leaf.board)
            pending[key][id(item.leaf)] = item.leaf

        if unique_boards:
            evaluations = self.evaluator.evaluate(unique_boards)
            for board, (probabilities, value) in zip(
                unique_boards,
                evaluations,
                strict=True,
            ):
                for node in pending[(board.player, board.opponent)].values():
                    self._expand(node, probabilities, value)

        for item in selected:
            value = item.value
            if value is None:
                value = item.leaf.network_value
            self._backup(item.path, value)

    def search_many(
        self,
        roots: Sequence[MCTSNode],
        *,
        add_root_noise: bool,
        rng: np.random.Generator | None = None,
    ) -> list[SearchResult]:
        if not roots:
            return []
        generator = rng or np.random.default_rng()

        results: list[SearchResult | None] = [None] * len(roots)
        searchable: list[tuple[int, MCTSNode]] = []
        to_expand: list[MCTSNode] = []
        for index, root in enumerate(roots):
            if not root.board.legal_moves_bits():
                raise ValueError("MCTS root must be a legal decision state")
            if (
                self.config.exact_endgame_empties
                and root.board.empty_count <= self.config.exact_endgame_empties
            ):
                actions, margin = self.exact_solver.best_actions(root.board)
                visits = np.zeros(ACTION_DIM, dtype=np.int32)
                visits[list(actions)] = 1
                policy = visits.astype(np.float32) / len(actions)
                results[index] = SearchResult(
                    visit_counts=visits,
                    policy=policy,
                    root_value=float(np.sign(margin)),
                    exact=True,
                )
                continue
            searchable.append((index, root))
            if not root.expanded:
                to_expand.append(root)

        if to_expand:
            unique: dict[tuple[int, int], MCTSNode] = {}
            for root in to_expand:
                unique.setdefault((root.board.player, root.board.opponent), root)
            unique_roots = list(unique.values())
            evaluations = self.evaluator.evaluate([root.board for root in unique_roots])
            by_key = {
                (root.board.player, root.board.opponent): result
                for root, result in zip(unique_roots, evaluations, strict=True)
            }
            for root in to_expand:
                probabilities, value = by_key[(root.board.player, root.board.opponent)]
                self._expand(root, probabilities, value)

        for _, root in searchable:
            self._set_root_noise(root, add_root_noise, generator)

        remaining = [self.config.simulations] * len(searchable)
        while any(remaining):
            selected: list[_SelectedLeaf] = []
            reservations: list[
                Sequence[tuple[MCTSNode, int | None]]
            ] = []
            completed = [0] * len(searchable)
            try:
                for search_index, (_, root) in enumerate(searchable):
                    target = min(
                        self.config.leaf_batch_size,
                        remaining[search_index],
                    )
                    if target <= 0:
                        continue
                    root_selected, root_reservations = self._select_batch(
                        root,
                        target,
                    )
                    selected.extend(root_selected)
                    reservations.extend(root_reservations)
                    completed[search_index] = len(root_selected)
            finally:
                for path in reservations:
                    self._release(path)
            self._evaluate_selected(selected)
            for search_index, count in enumerate(completed):
                if remaining[search_index] and count <= 0:
                    raise RuntimeError("Batched MCTS made no search progress")
                remaining[search_index] -= count

        for index, root in searchable:
            visits = root.visit_counts.copy()
            total = int(visits.sum())
            if total <= 0:
                raise RuntimeError("MCTS completed without visiting a root action")
            policy = visits.astype(np.float32) / total
            q_values = root.q_values()
            root_value = float(np.sum(policy * q_values))
            results[index] = SearchResult(
                visit_counts=visits,
                policy=policy,
                root_value=root_value,
                exact=False,
            )
        return [item for item in results if item is not None]

    def search(
        self,
        root: MCTSNode,
        *,
        add_root_noise: bool = False,
        rng: np.random.Generator | None = None,
    ) -> SearchResult:
        return self.search_many(
            [root],
            add_root_noise=add_root_noise,
            rng=rng,
        )[0]


def select_action(
    visit_counts: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
) -> int:
    counts = np.asarray(visit_counts, dtype=np.float64)
    if counts.shape != (ACTION_DIM,):
        raise ValueError(f"Expected 64 visit counts, received {counts.shape}")
    if not np.any(counts > 0):
        raise ValueError("Cannot select from empty MCTS visit counts")
    if temperature <= 1e-8:
        return int(np.argmax(counts))
    positive = counts > 0
    logits = np.full(ACTION_DIM, -np.inf, dtype=np.float64)
    logits[positive] = np.log(counts[positive]) / temperature
    logits -= np.max(logits[positive])
    probabilities = np.zeros(ACTION_DIM, dtype=np.float64)
    probabilities[positive] = np.exp(logits[positive])
    probabilities /= probabilities.sum()
    return int(rng.choice(ACTION_DIM, p=probabilities))


__all__ = [
    "ExactSolver",
    "MCTSConfig",
    "MCTSNode",
    "NeuralMCTS",
    "SearchResult",
    "select_action",
]
