"""PPO actor-critic model and inference adapter for COSMOS Othello.

The network always sees the board from the player-to-move's perspective.  Its
policy has one logit per square; illegal squares are masked before sampling or
selecting a move.  Passes are handled by the game driver and therefore do not
need a separate action.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from game import Game, LegalMove


CHECKPOINT_FORMAT = "cosmos-ppo-othello"
CHECKPOINT_VERSION = 1
BOARD_SIZE = 8
ACTION_DIM = BOARD_SIZE * BOARD_SIZE
INPUT_PLANES = 3


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the residual actor-critic network."""

    channels: int = 64
    residual_blocks: int = 4

    def validate(self) -> None:
        if self.channels < 8:
            raise ValueError("channels must be at least 8")
        if self.residual_blocks < 1:
            raise ValueError("residual_blocks must be at least 1")


@dataclass(frozen=True)
class SearchConfig:
    """Inference-time policy/value-guided alpha-beta settings."""

    depth: int = 2
    endgame_exact_empties: int = 8
    symmetry_ensemble: bool = False

    def validate(self) -> None:
        if self.depth < 0:
            raise ValueError("search depth cannot be negative")
        if self.endgame_exact_empties < 0:
            raise ValueError("endgame_exact_empties cannot be negative")


def opponent(color: int) -> int:
    return Game.WHITE if color == Game.BLACK else Game.BLACK


def coord_to_action(x: int, y: int) -> int:
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        raise ValueError(f"Coordinate is outside the board: ({x}, {y})")
    return y * BOARD_SIZE + x


def action_to_coord(action: int) -> tuple[int, int]:
    if not 0 <= int(action) < ACTION_DIM:
        raise ValueError(f"Action must be between 0 and {ACTION_DIM - 1}: {action}")
    y, x = divmod(int(action), BOARD_SIZE)
    return x, y


def legal_moves_mask(
    legal_moves: Sequence[LegalMove] | Sequence[tuple[int, int]],
) -> np.ndarray:
    """Return a 64-element boolean mask for LegalMove or coordinate objects."""

    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    for move in legal_moves:
        if isinstance(move, LegalMove):
            x, y = move.x, move.y
        else:
            x, y = move
        mask[coord_to_action(x, y)] = True
    return mask


def encode_state(
    game_or_board: Game | Sequence[Sequence[int]],
    color: int,
    legal_moves: Sequence[LegalMove] | Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Encode own discs, opponent discs, and legal moves as three 8x8 planes."""

    if color not in (Game.BLACK, Game.WHITE):
        raise ValueError(f"Unknown Othello color: {color}")
    if isinstance(game_or_board, Game):
        game = game_or_board
        board = game.board
        if legal_moves is None:
            legal_moves = game.legal_moves(color)
    else:
        board = game_or_board
        if legal_moves is None:
            raise ValueError("legal_moves are required when encoding a raw board")

    board_array = np.asarray(board, dtype=np.int8)
    if board_array.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(
            f"PPO expects an 8x8 board; received shape {board_array.shape}"
        )

    other = opponent(color)
    legal = legal_moves_mask(legal_moves).reshape(BOARD_SIZE, BOARD_SIZE)
    planes = np.stack(
        (board_array == color, board_array == other, legal),
        axis=0,
    )
    return planes.astype(np.float32, copy=False)


def _make_action_transforms() -> tuple[np.ndarray, np.ndarray]:
    transforms = np.empty((8, ACTION_DIM), dtype=np.int64)
    for symmetry in range(8):
        rotations = symmetry % 4
        reflect = symmetry >= 4
        for action in range(ACTION_DIM):
            marker = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
            x, y = action_to_coord(action)
            marker[y, x] = 1
            transformed = np.rot90(marker, rotations)
            if reflect:
                transformed = np.flip(transformed, axis=1)
            transforms[symmetry, action] = int(np.flatnonzero(transformed)[0])

    inverses = np.empty_like(transforms)
    for symmetry in range(8):
        for original, transformed in enumerate(transforms[symmetry]):
            inverses[symmetry, transformed] = original
    return transforms, inverses


ACTION_TRANSFORMS, INVERSE_ACTION_TRANSFORMS = _make_action_transforms()


def transform_planes(planes: np.ndarray, symmetry: int) -> np.ndarray:
    """Apply one of the board's eight rotations/reflections to encoded planes."""

    if not 0 <= symmetry < 8:
        raise ValueError("symmetry must be in the range 0..7")
    if planes.shape != (INPUT_PLANES, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Unexpected encoded-state shape: {planes.shape}")
    transformed = np.rot90(planes, symmetry % 4, axes=(-2, -1))
    if symmetry >= 4:
        transformed = np.flip(transformed, axis=-1)
    return np.ascontiguousarray(transformed)


def transform_mask(mask: np.ndarray, symmetry: int) -> np.ndarray:
    if mask.shape != (ACTION_DIM,):
        raise ValueError(f"Unexpected action-mask shape: {mask.shape}")
    transformed = np.zeros_like(mask)
    transformed[ACTION_TRANSFORMS[symmetry]] = mask
    return transformed


def transform_action(action: int, symmetry: int) -> int:
    return int(ACTION_TRANSFORMS[symmetry, action])


def inverse_transform_action(action: int, symmetry: int) -> int:
    return int(INVERSE_ACTION_TRANSFORMS[symmetry, action])


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
        residual = inputs
        hidden = self.activation(self.norm1(self.conv1(inputs)))
        hidden = self.norm2(self.conv2(hidden))
        return self.activation(hidden + residual)


class PPOActorCritic(nn.Module):
    """Small residual actor-critic suitable for an 8x8 board."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
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
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_norm = nn.GroupNorm(1, 1)
        self.value_fc1 = nn.Linear(ACTION_DIM, 128)
        self.value_fc2 = nn.Linear(128, 1)
        self.activation = nn.ReLU(inplace=True)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_fc.weight, gain=0.01)
        nn.init.orthogonal_(self.value_fc2.weight, gain=1.0)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 4 or inputs.shape[1:] != (
            INPUT_PLANES,
            BOARD_SIZE,
            BOARD_SIZE,
        ):
            raise ValueError(
                "PPOActorCritic input must have shape "
                f"(batch, {INPUT_PLANES}, {BOARD_SIZE}, {BOARD_SIZE})"
            )
        hidden = self.trunk(self.stem(inputs))
        policy = self.activation(self.policy_norm(self.policy_conv(hidden)))
        logits = self.policy_fc(policy.flatten(1))
        value = self.activation(self.value_norm(self.value_conv(hidden)))
        value = self.activation(self.value_fc1(value.flatten(1)))
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)
        return logits, value

    @staticmethod
    def masked_logits(logits: Tensor, legal_mask: Tensor) -> Tensor:
        if logits.shape != legal_mask.shape:
            raise ValueError(
                f"Logits and legal mask must match: {logits.shape} != {legal_mask.shape}"
            )
        legal_mask = legal_mask.to(dtype=torch.bool)
        if not bool(torch.all(legal_mask.any(dim=-1))):
            raise ValueError("Every policy row must contain at least one legal action")
        return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)

    def distribution_and_value(
        self,
        inputs: Tensor,
        legal_mask: Tensor,
    ) -> tuple[Categorical, Tensor]:
        logits, value = self(inputs)
        return Categorical(logits=self.masked_logits(logits, legal_mask)), value


def predict_policy_value(
    model: PPOActorCritic,
    device: torch.device,
    game: Game,
    color: int,
    legal_moves: Sequence[LegalMove],
    symmetry_ensemble: bool = False,
) -> tuple[np.ndarray, float]:
    """Return legal policy probabilities and value from ``color``'s view."""

    mask = legal_moves_mask(legal_moves)
    if not mask.any():
        raise ValueError("Policy evaluation requires at least one legal move")
    encoded = encode_state(game, color, legal_moves)
    if symmetry_ensemble:
        symmetries = range(8)
        encoded_batch = np.stack(
            [transform_planes(encoded, symmetry) for symmetry in symmetries]
        )
        mask_batch = np.stack(
            [transform_mask(mask, symmetry) for symmetry in symmetries]
        )
    else:
        symmetries = range(1)
        encoded_batch = encoded[None, ...]
        mask_batch = mask[None, ...]

    states = torch.from_numpy(encoded_batch).to(device)
    masks = torch.from_numpy(mask_batch).to(device)
    with torch.inference_mode():
        distribution, values = model.distribution_and_value(states, masks)
        transformed_probabilities = distribution.probs.cpu().numpy()

    probabilities = np.zeros(ACTION_DIM, dtype=np.float64)
    for row, symmetry in zip(transformed_probabilities, symmetries, strict=True):
        probabilities += row[ACTION_TRANSFORMS[symmetry]]
    probabilities /= len(transformed_probabilities)
    probabilities[~mask] = 0.0
    total = probabilities.sum()
    if total > 0:
        probabilities /= total
    return probabilities, float(values.mean().item())


def _terminal_search_value(game: Game, color: int) -> float:
    scores = game.get_score()
    difference = scores[color] - scores[opponent(color)]
    if difference == 0:
        return 0.0
    # Preserve win/loss as the primary objective and use disc margin only to
    # order outcomes with the same result.
    return math.copysign(1.0, difference) + difference / 6400.0


def _empty_count(game: Game) -> int:
    return sum(square == Game.EMPTY for row in game.board for square in row)


def _ordered_search_moves(
    legal_moves: Sequence[LegalMove],
    probabilities: np.ndarray,
) -> list[LegalMove]:
    return sorted(
        legal_moves,
        key=lambda move: (
            probabilities[coord_to_action(move.x, move.y)],
            len(move.flips),
            -move.y,
            -move.x,
        ),
        reverse=True,
    )


def _static_ordered_search_moves(
    legal_moves: Sequence[LegalMove],
) -> list[LegalMove]:
    """Order exact-search moves without paying for a network evaluation."""

    corners = {(0, 0), (0, 7), (7, 0), (7, 7)}
    x_squares = {(1, 1), (1, 6), (6, 1), (6, 6)}
    c_squares = {
        (0, 1),
        (1, 0),
        (0, 6),
        (1, 7),
        (6, 0),
        (7, 1),
        (6, 7),
        (7, 6),
    }

    def priority(move: LegalMove) -> tuple[int, int, int, int]:
        coordinate = (move.x, move.y)
        if coordinate in corners:
            square_value = 100
        elif coordinate in x_squares:
            square_value = -20
        elif coordinate in c_squares:
            square_value = -10
        elif move.x in (0, 7) or move.y in (0, 7):
            square_value = 10
        else:
            square_value = 0
        return square_value, len(move.flips), -move.y, -move.x

    return sorted(legal_moves, key=priority, reverse=True)


def _predict_encoded_values(
    model: PPOActorCritic,
    device: torch.device,
    encoded_states: Sequence[np.ndarray],
    symmetry_ensemble: bool,
) -> np.ndarray:
    """Evaluate a collection of value leaves in one network call."""

    if not encoded_states:
        return np.empty(0, dtype=np.float32)
    if symmetry_ensemble:
        batch = np.stack(
            [
                transform_planes(encoded, symmetry)
                for encoded in encoded_states
                for symmetry in range(8)
            ]
        )
    else:
        batch = np.stack(encoded_states)
    with torch.inference_mode():
        _, values = model(torch.from_numpy(batch).to(device))
    output = values.cpu().numpy()
    if symmetry_ensemble:
        output = output.reshape(len(encoded_states), 8).mean(axis=1)
    return output


def _batched_shallow_search(
    model: PPOActorCritic,
    device: torch.device,
    game: Game,
    color: int,
    root_moves: Sequence[LegalMove],
    depth: int,
    config: SearchConfig,
) -> LegalMove:
    """Evaluate depth-one/two minimax leaves with one batched forward pass."""

    if depth not in (1, 2):
        raise ValueError("Batched shallow search supports depth 1 or 2")
    encoded_leaves: list[np.ndarray] = []
    leaf_signs: list[float] = []
    branches: list[list[float | tuple[int]]] = []

    def pending_leaf(
        perspective: int,
        legal: Sequence[LegalMove],
        sign: float,
    ) -> tuple[int]:
        index = len(encoded_leaves)
        encoded_leaves.append(encode_state(game, perspective, legal))
        leaf_signs.append(sign)
        return (index,)

    def value_or_leaf(next_color: int) -> float | tuple[int]:
        legal = game.legal_moves(next_color)
        if legal:
            sign = 1.0 if next_color == color else -1.0
            return pending_leaf(next_color, legal, sign)
        other = opponent(next_color)
        other_legal = game.legal_moves(other)
        if not other_legal:
            return _terminal_search_value(game, color)
        sign = 1.0 if other == color else -1.0
        return pending_leaf(other, other_legal, sign)

    for root_move in root_moves:
        game.play(color, root_move)
        try:
            opponent_moves = game.legal_moves(opponent(color))
            if depth == 1 or not opponent_moves:
                branches.append([value_or_leaf(opponent(color))])
                continue

            branch: list[float | tuple[int]] = []
            for reply in _static_ordered_search_moves(opponent_moves):
                game.play(opponent(color), reply)
                try:
                    branch.append(value_or_leaf(color))
                finally:
                    game.undo(opponent(color), reply)
            branches.append(branch)
        finally:
            game.undo(color, root_move)

    predicted = _predict_encoded_values(
        model,
        device,
        encoded_leaves,
        config.symmetry_ensemble,
    )

    def resolve(item: float | tuple[int]) -> float:
        if isinstance(item, tuple):
            index = item[0]
            return float(predicted[index]) * leaf_signs[index]
        return item

    # At depth two each branch contains all opponent replies, so the opponent
    # selects the minimum root-perspective value. At depth one there is one item.
    branch_values = [min(resolve(item) for item in branch) for branch in branches]
    return root_moves[int(np.argmax(branch_values))]


def _negamax(
    model: PPOActorCritic,
    device: torch.device,
    game: Game,
    color: int,
    depth: int,
    alpha: float,
    beta: float,
    config: SearchConfig,
) -> float:
    legal_moves = game.legal_moves(color)
    if not legal_moves:
        other = opponent(color)
        if not game.legal_moves(other):
            return _terminal_search_value(game, color)
        return -_negamax(
            model,
            device,
            game,
            other,
            depth,
            -beta,
            -alpha,
            config,
        )

    exact_endgame = _empty_count(game) <= config.endgame_exact_empties
    if depth <= 0 and not exact_endgame:
        _, value = predict_policy_value(
            model,
            device,
            game,
            color,
            legal_moves,
            config.symmetry_ensemble,
        )
        return value

    if exact_endgame:
        ordered_moves = _static_ordered_search_moves(legal_moves)
    else:
        probabilities, _ = predict_policy_value(
            model,
            device,
            game,
            color,
            legal_moves,
            config.symmetry_ensemble,
        )
        ordered_moves = _ordered_search_moves(legal_moves, probabilities)
    best_value = -float("inf")
    next_depth = depth - 1
    for move in ordered_moves:
        game.play(color, move)
        try:
            value = -_negamax(
                model,
                device,
                game,
                opponent(color),
                next_depth,
                -beta,
                -alpha,
                config,
            )
        finally:
            game.undo(color, move)
        best_value = max(best_value, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return best_value


def choose_search_move(
    model: PPOActorCritic,
    device: torch.device,
    game: Game,
    color: int,
    legal_moves: Sequence[LegalMove],
    config: SearchConfig | None = None,
) -> tuple[int, int]:
    """Choose a move using policy ordering and value-guided alpha-beta."""

    game = game.copy()

    search = config or SearchConfig()
    search.validate()
    if not legal_moves:
        raise ValueError("Search cannot choose a move when the player must pass")
    probabilities, _ = predict_policy_value(
        model,
        device,
        game,
        color,
        legal_moves,
        search.symmetry_ensemble,
    )
    if search.depth == 0 and _empty_count(game) > search.endgame_exact_empties:
        return action_to_coord(int(np.argmax(probabilities)))

    ordered_moves = _ordered_search_moves(legal_moves, probabilities)
    if search.depth in (1, 2) and _empty_count(game) > search.endgame_exact_empties:
        best_move = _batched_shallow_search(
            model,
            device,
            game,
            color,
            ordered_moves,
            search.depth,
            search,
        )
        return best_move.x, best_move.y

    best_move: LegalMove | None = None
    best_value = -float("inf")
    alpha = -float("inf")
    for move in ordered_moves:
        game.play(color, move)
        try:
            value = -_negamax(
                model,
                device,
                game,
                opponent(color),
                search.depth - 1,
                -float("inf"),
                -alpha,
                search,
            )
        finally:
            game.undo(color, move)
        if value > best_value:
            best_value = value
            best_move = move
        alpha = max(alpha, value)
    assert best_move is not None
    return best_move.x, best_move.y


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    normalized = device.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(normalized)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA device"
        )
    return selected


def _torch_load(path: Path, map_location: torch.device | str) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unrecognized PPO checkpoint payload: {path}")
    return payload


def checkpoint_payload(
    model: PPOActorCritic,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    training_config: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    league: Sequence[str] | None = None,
    hall_of_fame: Sequence[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "training_config": dict(training_config or {}),
        "training_state": dict(training_state or {}),
        "metrics": dict(metrics or {}),
        "league": list(league or ()),
        "hall_of_fame": list(hall_of_fame or ()),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    return payload


def save_checkpoint(
    path: str | os.PathLike[str],
    model: PPOActorCritic,
    **payload_kwargs: Any,
) -> Path:
    """Atomically save an inference or resumable training checkpoint."""

    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp")
    torch.save(checkpoint_payload(model, **payload_kwargs), temporary)
    os.replace(temporary, checkpoint)
    return checkpoint


def load_checkpoint(
    path: str | os.PathLike[str],
    device: str | torch.device = "cpu",
) -> tuple[PPOActorCritic, Mapping[str, Any]]:
    checkpoint = Path(path)
    selected_device = resolve_device(device)
    payload = _torch_load(checkpoint, selected_device)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Not a COSMOS PPO checkpoint: {checkpoint}")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported PPO checkpoint version {payload.get('version')!r}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    raw_config = payload.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"PPO checkpoint has no model configuration: {checkpoint}")
    model = PPOActorCritic(ModelConfig(**dict(raw_config))).to(selected_device)
    try:
        model.load_state_dict(payload["model_state"])
    except (KeyError, RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid PPO model weights: {checkpoint}") from exc
    model.eval()
    return model, payload


class PPOPlayer:
    """Benchmark/spectator adapter for a trained PPO checkpoint."""

    def __init__(
        self,
        checkpoint: str | os.PathLike[str],
        device: str | torch.device = "auto",
        search_depth: int = 2,
        endgame_exact_empties: int = 8,
        symmetry_ensemble: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        self.device = resolve_device(device)
        self.model, self.payload = load_checkpoint(self.checkpoint, self.device)
        self.search = SearchConfig(
            depth=search_depth,
            endgame_exact_empties=endgame_exact_empties,
            symmetry_ensemble=symmetry_ensemble,
        )
        self.search.validate()
        config = self.model.config
        iteration = self.payload.get("training_state", {}).get("iteration")
        iteration_text = "" if iteration is None else f", iteration {iteration}"
        self.name = (
            f"PPO ({self.checkpoint.name}{iteration_text}, {config.channels} channels, "
            f"{config.residual_blocks} blocks, search {self.search.depth}, "
            f"exact {self.search.endgame_exact_empties})"
        )

    def _predict(
        self,
        game: Game,
        color: int,
        legal_moves: Sequence[LegalMove],
    ) -> tuple[int, float]:
        mask = legal_moves_mask(legal_moves)
        if not mask.any():
            raise ValueError("PPOPlayer cannot choose a move when the player must pass")
        encoded = encode_state(game, color, legal_moves)
        states = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        masks = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            distribution, value = self.model.distribution_and_value(states, masks)
            action = int(torch.argmax(distribution.logits, dim=-1).item())
        return action, float(value.item())

    def choose_move(
        self,
        game: Game,
        color: int,
        legal_moves: Sequence[LegalMove],
        rng: Any,
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

    def get_value_prediction(self, game: Game, color: int) -> float:
        legal_moves = game.legal_moves(color)
        if not legal_moves:
            return 0.0
        _, value = self._predict(game, color, legal_moves)
        return value


__all__ = [
    "ACTION_DIM",
    "ACTION_TRANSFORMS",
    "BOARD_SIZE",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "INVERSE_ACTION_TRANSFORMS",
    "INPUT_PLANES",
    "ModelConfig",
    "PPOActorCritic",
    "PPOPlayer",
    "SearchConfig",
    "action_to_coord",
    "checkpoint_payload",
    "coord_to_action",
    "choose_search_move",
    "encode_state",
    "inverse_transform_action",
    "legal_moves_mask",
    "load_checkpoint",
    "opponent",
    "predict_policy_value",
    "resolve_device",
    "save_checkpoint",
    "transform_action",
    "transform_mask",
    "transform_planes",
]
