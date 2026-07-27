"""Fast, immutable Othello bitboards for AlphaZero search and self-play.

The rest of COSMOS keeps using :class:`game.Game` as the reference rules
implementation.  AlphaZero explores many thousands of hypothetical positions,
so its hot path stores a position as two Python integers instead.  ``player``
always means the player to move and ``opponent`` the other player; applying a
move swaps those perspectives in the returned board.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

BOARD_SIZE = 8
ACTION_DIM = BOARD_SIZE * BOARD_SIZE
EMPTY = 0
BLACK = 1
WHITE = 2
FULL_BOARD = (1 << ACTION_DIM) - 1
FILE_A = sum(1 << (row * BOARD_SIZE) for row in range(BOARD_SIZE))
FILE_H = FILE_A << (BOARD_SIZE - 1)
NOT_FILE_A = FULL_BOARD ^ FILE_A
NOT_FILE_H = FULL_BOARD ^ FILE_H
DIRECTIONS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
CORNERS = (0, 7, 56, 63)


def opponent_color(color: int) -> int:
    if color == BLACK:
        return WHITE
    if color == WHITE:
        return BLACK
    raise ValueError(f"Unknown Othello color: {color}")


def coord_to_action(x: int, y: int) -> int:
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        raise ValueError(f"Coordinate is outside the board: ({x}, {y})")
    return y * BOARD_SIZE + x


def action_to_coord(action: int) -> tuple[int, int]:
    if not 0 <= int(action) < ACTION_DIM:
        raise ValueError(f"Action must be in 0..{ACTION_DIM - 1}: {action}")
    y, x = divmod(int(action), BOARD_SIZE)
    return x, y


def iter_actions(bits: int) -> Iterable[int]:
    """Yield set-bit indices from least to most significant."""

    remaining = int(bits)
    while remaining:
        least = remaining & -remaining
        yield least.bit_length() - 1
        remaining ^= least


def _build_move_rays() -> tuple[tuple[tuple[int, ...], ...], ...]:
    rays_by_action = []
    for action in range(ACTION_DIM):
        x, y = action_to_coord(action)
        action_rays = []
        for dx, dy in DIRECTIONS:
            ray = []
            next_x, next_y = x + dx, y + dy
            while 0 <= next_x < BOARD_SIZE and 0 <= next_y < BOARD_SIZE:
                ray.append(1 << coord_to_action(next_x, next_y))
                next_x += dx
                next_y += dy
            action_rays.append(tuple(ray))
        rays_by_action.append(tuple(action_rays))
    return tuple(rays_by_action)


MOVE_RAYS = _build_move_rays()


def _legal_moves(player: int, opponent: int) -> int:
    """Dumb7Fill move generation using native 64-bit integer operations."""

    empty = FULL_BOARD ^ (player | opponent)
    legal = 0
    for shift, edge_mask in (
        (1, NOT_FILE_A),
        (7, NOT_FILE_H),
        (8, FULL_BOARD),
        (9, NOT_FILE_A),
    ):
        propagator = opponent & edge_mask
        flood = propagator & ((player << shift) & FULL_BOARD)
        flood |= propagator & ((flood << shift) & FULL_BOARD)
        flood |= propagator & ((flood << shift) & FULL_BOARD)
        flood |= propagator & ((flood << shift) & FULL_BOARD)
        flood |= propagator & ((flood << shift) & FULL_BOARD)
        flood |= propagator & ((flood << shift) & FULL_BOARD)
        legal |= empty & edge_mask & ((flood << shift) & FULL_BOARD)
    for shift, edge_mask in (
        (1, NOT_FILE_H),
        (7, NOT_FILE_A),
        (8, FULL_BOARD),
        (9, NOT_FILE_H),
    ):
        propagator = opponent & edge_mask
        flood = propagator & (player >> shift)
        flood |= propagator & (flood >> shift)
        flood |= propagator & (flood >> shift)
        flood |= propagator & (flood >> shift)
        flood |= propagator & (flood >> shift)
        flood |= propagator & (flood >> shift)
        legal |= empty & edge_mask & (flood >> shift)
    return legal


@dataclass(frozen=True, slots=True)
class BitBoard:
    """An Othello position canonicalized to the player-to-move."""

    player: int
    opponent: int
    _legal_bits: int | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _legal_actions: tuple[int, ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _flip_cache: dict[int, int] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        player = int(self.player)
        opponent = int(self.opponent)
        if player & opponent:
            raise ValueError("Player and opponent bitboards overlap")
        if (player | opponent) & ~FULL_BOARD:
            raise ValueError("Bitboard contains squares outside the 8x8 board")
        object.__setattr__(self, "player", player)
        object.__setattr__(self, "opponent", opponent)

    @classmethod
    def initial(cls) -> BitBoard:
        # Match the reference engine without constructing its mutable 8x8 list.
        black = (1 << coord_to_action(4, 3)) | (1 << coord_to_action(3, 4))
        white = (1 << coord_to_action(3, 3)) | (1 << coord_to_action(4, 4))
        return cls(black, white)

    @classmethod
    def from_game(cls, game: Any, color: int) -> BitBoard:
        """Convert a reference-engine-compatible game without importing it."""

        if game.side != BOARD_SIZE:
            raise ValueError("AlphaZero supports only an 8x8 Othello board")
        own = 0
        other = 0
        other_color = opponent_color(color)
        for y, row in enumerate(game.board):
            for x, square in enumerate(row):
                bit = 1 << coord_to_action(x, y)
                if square == color:
                    own |= bit
                elif square == other_color:
                    other |= bit
        return cls(own, other)

    @property
    def occupied(self) -> int:
        return self.player | self.opponent

    @property
    def empty(self) -> int:
        return FULL_BOARD ^ self.occupied

    @property
    def empty_count(self) -> int:
        return ACTION_DIM - self.occupied.bit_count()

    @property
    def disc_margin(self) -> int:
        return self.player.bit_count() - self.opponent.bit_count()

    def legal_moves_bits(self) -> int:
        cached = self._legal_bits
        if cached is not None:
            return cached
        legal = _legal_moves(self.player, self.opponent)
        object.__setattr__(self, "_legal_bits", legal)
        return legal

    def legal_actions(self) -> tuple[int, ...]:
        cached = self._legal_actions
        if cached is None:
            cached = tuple(iter_actions(self.legal_moves_bits()))
            object.__setattr__(self, "_legal_actions", cached)
        return cached

    def flips_for_action(self, action: int) -> int:
        action = int(action)
        if not 0 <= action < ACTION_DIM:
            return 0
        cache = self._flip_cache
        if cache is None:
            cache = {}
            object.__setattr__(self, "_flip_cache", cache)
        elif action in cache:
            return cache[action]

        move = 1 << action
        if not move & self.empty:
            cache[action] = 0
            return 0
        flips = 0
        for ray in MOVE_RAYS[action]:
            captured = 0
            for bit in ray:
                if bit & self.opponent:
                    captured |= bit
                elif bit & self.player:
                    flips |= captured
                    break
                else:
                    break
        cache[action] = flips
        return flips

    def legal_moves_with_flips(self) -> tuple[tuple[int, int], ...]:
        """Return legal actions and cached flip masks for exact search."""

        return tuple(
            (action, self.flips_for_action(action))
            for action in self.legal_actions()
        )

    def play(self, action: int) -> BitBoard:
        if not 0 <= int(action) < ACTION_DIM:
            raise ValueError(f"Action must be in 0..{ACTION_DIM - 1}: {action}")
        flips = self.flips_for_action(action)
        if not flips:
            raise ValueError(f"Illegal Othello action: {action}")
        return self._play_with_flips_unchecked(int(action), flips)

    def play_with_flips(self, action: int, flips: int) -> BitBoard:
        """Play a move with a previously validated nonzero flip mask."""

        action = int(action)
        flips = int(flips)
        if not 0 <= action < ACTION_DIM:
            raise ValueError(f"Action must be in 0..{ACTION_DIM - 1}: {action}")
        if flips != self.flips_for_action(action):
            raise ValueError(f"Invalid flips for Othello action: {action}")
        return self._play_with_flips_unchecked(action, flips)

    def _play_with_flips_unchecked(self, action: int, flips: int) -> BitBoard:
        """Internal exact-search fast path for a known legal flip mask."""

        move = 1 << action
        own_after = self.player | move | flips
        other_after = self.opponent & ~flips
        return BitBoard(other_after, own_after)

    def pass_turn(self) -> BitBoard:
        if self.legal_moves_bits():
            raise ValueError("Cannot pass while a legal move is available")
        return BitBoard(self.opponent, self.player)

    def is_terminal(self) -> bool:
        if self.legal_moves_bits():
            return False
        return not self.pass_turn().legal_moves_bits()


def _transform_coordinate(x: int, y: int, symmetry: int) -> tuple[int, int]:
    if not 0 <= symmetry < 8:
        raise ValueError("Symmetry must be in 0..7")
    rotations = symmetry % 4
    for _ in range(rotations):
        x, y = y, BOARD_SIZE - 1 - x
    if symmetry >= 4:
        x = BOARD_SIZE - 1 - x
    return x, y


ACTION_TRANSFORMS = tuple(
    tuple(
        coord_to_action(*_transform_coordinate(*action_to_coord(action), symmetry))
        for action in range(ACTION_DIM)
    )
    for symmetry in range(8)
)
INVERSE_ACTION_TRANSFORMS = tuple(
    tuple(
        next(
            original
            for original, transformed in enumerate(ACTION_TRANSFORMS[symmetry])
            if transformed == action
        )
        for action in range(ACTION_DIM)
    )
    for symmetry in range(8)
)


def transform_action(action: int, symmetry: int) -> int:
    return ACTION_TRANSFORMS[symmetry][int(action)]


def inverse_transform_action(action: int, symmetry: int) -> int:
    return INVERSE_ACTION_TRANSFORMS[symmetry][int(action)]


def _build_byte_transform_lut() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Map each byte of a bitboard through each D4 symmetry."""

    symmetries = []
    for symmetry in range(8):
        byte_tables = []
        for byte_index in range(8):
            first_action = byte_index * 8
            table = []
            for byte_value in range(256):
                transformed = 0
                for bit_index in range(8):
                    if byte_value & (1 << bit_index):
                        transformed |= 1 << ACTION_TRANSFORMS[symmetry][
                            first_action + bit_index
                        ]
                table.append(transformed)
            byte_tables.append(tuple(table))
        symmetries.append(tuple(byte_tables))
    return tuple(symmetries)


BYTE_TRANSFORM_LUT = _build_byte_transform_lut()


def transform_bits(bits: int, symmetry: int) -> int:
    if not 0 <= symmetry < 8:
        raise ValueError("Symmetry must be in 0..7")
    value = int(bits)
    tables = BYTE_TRANSFORM_LUT[symmetry]
    return (
        tables[0][value & 0xFF]
        | tables[1][(value >> 8) & 0xFF]
        | tables[2][(value >> 16) & 0xFF]
        | tables[3][(value >> 24) & 0xFF]
        | tables[4][(value >> 32) & 0xFF]
        | tables[5][(value >> 40) & 0xFF]
        | tables[6][(value >> 48) & 0xFF]
        | tables[7][(value >> 56) & 0xFF]
    )


def transform_board(board: BitBoard, symmetry: int) -> BitBoard:
    return BitBoard(
        transform_bits(board.player, symmetry),
        transform_bits(board.opponent, symmetry),
    )


def transform_dense(values: Sequence[int] | np.ndarray, symmetry: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (ACTION_DIM,):
        raise ValueError(f"Expected 64 action values, received {array.shape}")
    transformed = np.empty_like(array)
    transformed[np.asarray(ACTION_TRANSFORMS[symmetry])] = array
    return transformed


def encode_boards(boards: Sequence[BitBoard]) -> np.ndarray:
    """Vector-ready four-plane encoding for a batch of canonical bitboards."""

    count = len(boards)
    encoded = np.empty((count, 4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if not count:
        return encoded
    player = _unpack_bitboards(board.player for board in boards)
    opponent = _unpack_bitboards(board.opponent for board in boards)
    legal = _unpack_bitboards(board.legal_moves_bits() for board in boards)
    encoded[:, 0] = player.reshape(count, BOARD_SIZE, BOARD_SIZE)
    encoded[:, 1] = opponent.reshape(count, BOARD_SIZE, BOARD_SIZE)
    encoded[:, 2] = legal.reshape(count, BOARD_SIZE, BOARD_SIZE)
    occupied = np.fromiter(
        (board.occupied.bit_count() for board in boards),
        dtype=np.float32,
        count=count,
    )
    encoded[:, 3] = ((ACTION_DIM - occupied) / 60.0)[:, None, None]
    return encoded


def legal_masks(boards: Sequence[BitBoard]) -> np.ndarray:
    return _unpack_bitboards(
        (board.legal_moves_bits() for board in boards),
    ).astype(np.bool_, copy=False)


def _unpack_bitboards(values: Iterable[int]) -> np.ndarray:
    """Expand little-endian uint64 bitboards to rows of 64 binary values."""

    packed = np.fromiter(values, dtype="<u8")
    if not len(packed):
        return np.empty((0, ACTION_DIM), dtype=np.uint8)
    return np.unpackbits(
        packed.view(np.uint8).reshape(-1, 8),
        axis=1,
        bitorder="little",
    )


__all__ = [
    "ACTION_DIM",
    "ACTION_TRANSFORMS",
    "BLACK",
    "BOARD_SIZE",
    "BYTE_TRANSFORM_LUT",
    "CORNERS",
    "EMPTY",
    "FULL_BOARD",
    "INVERSE_ACTION_TRANSFORMS",
    "WHITE",
    "BitBoard",
    "action_to_coord",
    "coord_to_action",
    "encode_boards",
    "inverse_transform_action",
    "iter_actions",
    "legal_masks",
    "opponent_color",
    "transform_action",
    "transform_bits",
    "transform_board",
    "transform_dense",
]
