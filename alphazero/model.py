"""Policy/value network, inference service, and checkpoints for AlphaZero.

This module is intentionally independent of ``ppo_model``.  AlphaZero learns
from MCTS visit distributions and game outcomes; it does not use PPO ratios,
advantages, clipping, or PPO checkpoints.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn

from .board import (
    ACTION_DIM,
    ACTION_TRANSFORMS,
    BOARD_SIZE,
    BitBoard,
    action_to_coord,
    encode_boards,
    legal_masks,
    transform_board,
)

if TYPE_CHECKING:
    from game import Game, LegalMove


CHECKPOINT_FORMAT = "cosmos-alphazero-othello"
CHECKPOINT_VERSION = 1
INPUT_PLANES = 4
WDL_DIM = 3
OWNERSHIP_CLASSES = 3
DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent.parent / "models" / "alphazero"
)


@dataclass(frozen=True)
class AlphaZeroModelConfig:
    channels: int = 96
    residual_blocks: int = 6
    value_hidden: int = 128

    def validate(self) -> None:
        if self.channels < 8:
            raise ValueError("AlphaZero channels must be at least 8")
        if self.residual_blocks < 1:
            raise ValueError("AlphaZero requires at least one residual block")
        if self.value_hidden < 16:
            raise ValueError("AlphaZero value_hidden must be at least 16")


def _group_count(channels: int) -> int:
    return math.gcd(8, channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.activation(self.norm1(self.conv1(inputs)))
        hidden = self.norm2(self.conv2(hidden))
        return self.activation(inputs + hidden)


class AlphaZeroNetwork(nn.Module):
    """Residual policy/WDL network with dense Othello auxiliary heads."""

    def __init__(self, config: AlphaZeroModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or AlphaZeroModelConfig()
        self.config.validate()
        channels = self.config.channels

        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(self.config.residual_blocks))
        )

        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_norm = nn.GroupNorm(1, 2)
        self.policy_fc = nn.Linear(2 * ACTION_DIM, ACTION_DIM)

        self.value_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.value_norm = nn.GroupNorm(1, 2)
        self.value_fc = nn.Linear(2 * ACTION_DIM, self.config.value_hidden)
        self.wdl_fc = nn.Linear(self.config.value_hidden, WDL_DIM)
        self.margin_fc = nn.Linear(self.config.value_hidden, 1)

        self.ownership_conv = nn.Conv2d(
            channels,
            OWNERSHIP_CLASSES,
            1,
            bias=True,
        )
        self.activation = nn.ReLU(inplace=True)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.policy_fc.weight, std=0.01)
        nn.init.normal_(self.wdl_fc.weight, std=0.01)
        nn.init.normal_(self.margin_fc.weight, std=0.01)
        nn.init.normal_(self.ownership_conv.weight, std=0.01)
        # Fresh residual blocks start as identity mappings, which stabilizes
        # optimization without changing checkpoint structure or old weights.
        for block in self.trunk:
            nn.init.zeros_(block.norm2.weight)

    @staticmethod
    def _validate_inputs(inputs: Tensor) -> None:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (
            INPUT_PLANES,
            BOARD_SIZE,
            BOARD_SIZE,
        ):
            raise ValueError(
                "AlphaZeroNetwork inputs must have shape "
                f"(batch, {INPUT_PLANES}, {BOARD_SIZE}, {BOARD_SIZE})"
            )

    def _hidden(self, inputs: Tensor) -> Tensor:
        self._validate_inputs(inputs)
        return self.trunk(self.stem(inputs))

    def _policy_logits(self, hidden: Tensor) -> Tensor:
        policy = self.activation(self.policy_norm(self.policy_conv(hidden)))
        return self.policy_fc(policy.flatten(1))

    def _value_features(self, hidden: Tensor) -> Tensor:
        value = self.activation(self.value_norm(self.value_conv(hidden)))
        return self.activation(self.value_fc(value.flatten(1)))

    def inference(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return only the heads needed by MCTS leaf evaluation."""

        hidden = self._hidden(inputs)
        value = self._value_features(hidden)
        return self._policy_logits(hidden), self.wdl_fc(value)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        hidden = self._hidden(inputs)
        policy_logits = self._policy_logits(hidden)
        value = self._value_features(hidden)
        wdl_logits = self.wdl_fc(value)
        margin = torch.tanh(self.margin_fc(value)).squeeze(-1)
        ownership_logits = self.ownership_conv(hidden)
        return policy_logits, wdl_logits, margin, ownership_logits

    @staticmethod
    def masked_policy_logits(policy_logits: Tensor, legal_mask: Tensor) -> Tensor:
        if policy_logits.shape != legal_mask.shape:
            raise ValueError(
                "Policy logits and legal mask differ: "
                f"{policy_logits.shape} != {legal_mask.shape}"
            )
        legal_mask = legal_mask.to(dtype=torch.bool)
        if not bool(torch.all(legal_mask.any(dim=-1))):
            raise ValueError("Every policy row must have at least one legal move")
        return policy_logits.masked_fill(
            ~legal_mask,
            torch.finfo(policy_logits.dtype).min,
        )


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return selected


def prepare_model(
    model: AlphaZeroNetwork,
    device: str | torch.device,
) -> AlphaZeroNetwork:
    """Move a model using the convolution-friendly NHWC memory layout."""

    selected_device = resolve_device(device)
    if selected_device.type in {"cpu", "cuda"}:
        return model.to(
            selected_device,
            memory_format=torch.channels_last,
        )
    return model.to(selected_device)


class AlphaZeroEvaluator:
    """Batched, cached neural inference for MCTS leaf expansion."""

    def __init__(
        self,
        model: AlphaZeroNetwork,
        device: str | torch.device = "auto",
        *,
        maximum_batch_size: int = 512,
        cache_size: int = 50_000,
        mixed_precision: bool = True,
        symmetry_ensemble: bool = False,
    ) -> None:
        if maximum_batch_size < 1:
            raise ValueError("maximum_batch_size must be positive")
        if cache_size < 0:
            raise ValueError("cache_size cannot be negative")
        self.model = model
        self.device = resolve_device(device)
        self.maximum_batch_size = maximum_batch_size
        self.cache_size = cache_size
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.symmetry_ensemble = symmetry_ensemble
        self.cache: OrderedDict[
            tuple[int, int],
            tuple[np.ndarray, float],
        ] = OrderedDict()
        prepare_model(self.model, self.device)
        self.model.eval()

    def clear_cache(self) -> None:
        self.cache.clear()

    def _cache_get(self, board: BitBoard) -> tuple[np.ndarray, float] | None:
        key = (board.player, board.opponent)
        result = self.cache.get(key)
        if result is not None:
            self.cache.move_to_end(key)
        return result

    def _cache_put(
        self,
        board: BitBoard,
        result: tuple[np.ndarray, float],
    ) -> None:
        if self.cache_size == 0:
            return
        key = (board.player, board.opponent)
        self.cache[key] = result
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def _network_batch(
        self,
        boards: Sequence[BitBoard],
    ) -> list[tuple[np.ndarray, float]]:
        if self.symmetry_ensemble:
            expanded = [
                transform_board(board, symmetry)
                for board in boards
                for symmetry in range(8)
            ]
        else:
            expanded = list(boards)

        results: list[tuple[np.ndarray, float]] = []
        for start in range(0, len(expanded), self.maximum_batch_size):
            chunk = expanded[start : start + self.maximum_batch_size]
            states = torch.from_numpy(encode_boards(chunk)).to(self.device)
            if self.device.type in {"cpu", "cuda"}:
                states = states.contiguous(memory_format=torch.channels_last)
            # The legal plane is already encoded and transferred with states.
            masks = states[:, 2].flatten(1).to(dtype=torch.bool)
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.mixed_precision,
                ),
            ):
                policy_logits, wdl_logits = self.model.inference(states)
                masked = self.model.masked_policy_logits(policy_logits, masks)
                probabilities = torch.softmax(masked.float(), dim=-1)
                wdl = torch.softmax(wdl_logits.float(), dim=-1)
                values = wdl[:, 2] - wdl[:, 0]
            policy_rows = probabilities.cpu().numpy()
            value_rows = values.cpu().numpy()
            results.extend(
                (policy_rows[index], float(value_rows[index]))
                for index in range(len(chunk))
            )

        if not self.symmetry_ensemble:
            return results

        combined: list[tuple[np.ndarray, float]] = []
        for board_index in range(len(boards)):
            policy = np.zeros(ACTION_DIM, dtype=np.float64)
            values = []
            for symmetry in range(8):
                transformed_policy, value = results[board_index * 8 + symmetry]
                mapping = np.asarray(ACTION_TRANSFORMS[symmetry])
                policy += transformed_policy[mapping]
                values.append(value)
            legal_array = legal_masks([boards[board_index]])[0]
            policy[~legal_array] = 0.0
            total = float(policy.sum())
            if total:
                policy /= total
            combined.append((policy.astype(np.float32), float(np.mean(values))))
        return combined

    def evaluate(
        self,
        boards: Sequence[BitBoard],
    ) -> list[tuple[np.ndarray, float]]:
        if not boards:
            return []
        output: list[tuple[np.ndarray, float] | None] = [None] * len(boards)
        missing_by_key: dict[tuple[int, int], list[int]] = {}
        unique_missing: list[BitBoard] = []
        for index, board in enumerate(boards):
            if not board.legal_moves_bits():
                raise ValueError("Neural evaluation requires a legal decision state")
            cached = self._cache_get(board)
            if cached is not None:
                output[index] = cached
                continue
            key = (board.player, board.opponent)
            if key not in missing_by_key:
                missing_by_key[key] = []
                unique_missing.append(board)
            missing_by_key[key].append(index)

        evaluated = self._network_batch(unique_missing)
        for board, result in zip(unique_missing, evaluated, strict=True):
            stable_result = (
                np.asarray(result[0], dtype=np.float32),
                float(result[1]),
            )
            self._cache_put(board, stable_result)
            for index in missing_by_key[(board.player, board.opponent)]:
                output[index] = stable_result
        return [item for item in output if item is not None]


def checkpoint_payload(
    model: AlphaZeroNetwork,
    *,
    ema_model: AlphaZeroNetwork | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    training_config: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "training_config": dict(training_config or {}),
        "training_state": dict(training_state or {}),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if ema_model is not None:
        if ema_model.config != model.config:
            raise ValueError("EMA model config differs from training model config")
        payload["ema_model_state"] = ema_model.state_dict()
    return payload


def save_checkpoint(
    path: str | os.PathLike[str],
    model: AlphaZeroNetwork,
    **payload_kwargs: Any,
) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp")
    torch.save(checkpoint_payload(model, **payload_kwargs), temporary)
    os.replace(temporary, checkpoint)
    return checkpoint


def _torch_load(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid AlphaZero checkpoint payload: {path}")
    return payload


def load_checkpoint(
    path: str | os.PathLike[str],
    device: str | torch.device = "cpu",
    *,
    use_ema: bool = False,
) -> tuple[AlphaZeroNetwork, Mapping[str, Any]]:
    checkpoint = Path(path).expanduser().resolve()
    selected_device = resolve_device(device)
    payload = _torch_load(checkpoint, selected_device)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Not a COSMOS AlphaZero checkpoint: {checkpoint}")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            "Unsupported AlphaZero checkpoint version "
            f"{payload.get('version')!r}; expected {CHECKPOINT_VERSION}"
        )
    raw_config = payload.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"AlphaZero checkpoint has no model config: {checkpoint}")
    model = AlphaZeroNetwork(AlphaZeroModelConfig(**dict(raw_config)))
    state = (
        payload.get("ema_model_state", payload.get("model_state"))
        if use_ema
        else payload.get("model_state")
    )
    try:
        model.load_state_dict(state)
    except (KeyError, RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid AlphaZero model weights: {checkpoint}") from exc
    prepare_model(model, selected_device)
    model.eval()
    return model, payload


class AlphaZeroPlayer:
    """COSMOS player adapter using neural-guided MCTS."""

    def __init__(
        self,
        checkpoint: str | os.PathLike[str],
        device: str | torch.device = "auto",
        *,
        simulations: int = 512,
        c_puct: float = 1.5,
        fpu_reduction: float = 0.20,
        leaf_batch_size: int = 8,
        virtual_loss: float = 1.0,
        exact_endgame_empties: int = 10,
        inference_batch_size: int = 512,
        cache_size: int = 50_000,
        symmetry_ensemble: bool = False,
    ) -> None:
        from .mcts import MCTSConfig, NeuralMCTS

        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = resolve_device(device)
        self.model, self.payload = load_checkpoint(
            self.checkpoint,
            self.device,
            use_ema=True,
        )
        self.evaluator = AlphaZeroEvaluator(
            self.model,
            self.device,
            maximum_batch_size=inference_batch_size,
            cache_size=cache_size,
            symmetry_ensemble=symmetry_ensemble,
        )
        self.search_config = MCTSConfig(
            simulations=simulations,
            c_puct=c_puct,
            fpu_reduction=fpu_reduction,
            leaf_batch_size=leaf_batch_size,
            virtual_loss=virtual_loss,
            exact_endgame_empties=exact_endgame_empties,
        )
        self.mcts = NeuralMCTS(self.evaluator, self.search_config)
        self._root = None
        generation = self.payload.get("training_state", {}).get("generation")
        generation_text = "" if generation is None else f", generation {generation}"
        config = self.model.config
        self.name = (
            f"AlphaZero ({self.checkpoint.name}{generation_text}, "
            f"{config.channels} channels, {config.residual_blocks} blocks, "
            f"{simulations} simulations, batch {leaf_batch_size}, "
            f"exact {exact_endgame_empties})"
        )

    def choose_move(
        self,
        game: Game,
        color: int,
        legal_moves: Sequence[LegalMove],
        rng: Any,
    ) -> tuple[int, int]:
        del rng
        if not legal_moves:
            raise ValueError("AlphaZeroPlayer cannot choose a move while passing")
        from .mcts import MCTSNode

        board = BitBoard.from_game(game, color)
        root = self._synchronize_root(board, MCTSNode)
        result = self.mcts.search(root, add_root_noise=False)
        action = int(np.argmax(result.visit_counts))
        self._root = root.child_for_action(action)
        coordinate = action_to_coord(action)
        legal_coordinates = {(move.x, move.y) for move in legal_moves}
        if coordinate not in legal_coordinates:
            raise RuntimeError(f"AlphaZero search selected illegal move {coordinate}")
        return coordinate

    def _synchronize_root(self, board: BitBoard, node_type: Any) -> Any:
        """Recover the played opponent branch before falling back to a new tree."""

        root = self._root
        if root is None:
            return node_type(board)
        if root.board == board:
            return root
        if not root.board.legal_moves_bits():
            passed = root.board.pass_turn()
            if passed == board:
                return root.child_for_pass()
        else:
            for action in root.board.legal_actions():
                expected = root.board.play(action)
                if expected == board:
                    return root.child_for_action(action)
        return node_type(board)

    def get_value_prediction(self, game: Game, color: int) -> float:
        board = BitBoard.from_game(game, color)
        if not board.legal_moves_bits():
            if board.is_terminal():
                return float(np.sign(board.disc_margin))
            board = board.pass_turn()
            _, value = self.evaluator.evaluate([board])[0]
            return -value
        _, value = self.evaluator.evaluate([board])[0]
        return value


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "DEFAULT_OUTPUT_DIRECTORY",
    "INPUT_PLANES",
    "OWNERSHIP_CLASSES",
    "WDL_DIM",
    "AlphaZeroEvaluator",
    "AlphaZeroModelConfig",
    "AlphaZeroNetwork",
    "AlphaZeroPlayer",
    "checkpoint_payload",
    "load_checkpoint",
    "prepare_model",
    "resolve_device",
    "save_checkpoint",
]
