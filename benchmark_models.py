#!/usr/bin/env python3
"""Run headless round-robin-style matches between COSMOS Othello players.

Run without arguments for an interactive model picker, or pass model specs:

    python benchmark_models.py --player-1 random --player-2 greedy --games 500
    python benchmark_models.py --player-1 minimax --player-2 greedy --games 100
    python benchmark_models.py --player-1 greedy \
        --player-2 models/v1/othello_100k.pth --games 100

The program is deliberately independent of the Pygame game loop so benchmarks
can run without opening a window. PyTorch is imported only when a DQN checkpoint
is selected.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence


EMPTY = 0
BLACK = 1
WHITE = 2
BOARD_SIZE = 8
DEFAULT_MINIMAX_DEPTH = 2 # reduced for benchmark test speed
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

# Positional values used by minimax. Corners are highly valuable, while the
# squares immediately next to an unclaimed corner are deliberately risky.
POSITION_WEIGHTS = (
    (120, -25, 20, 5, 5, 20, -25, 120),
    (-25, -45, -5, -5, -5, -5, -45, -25),
    (20, -5, 15, 3, 3, 15, -5, 20),
    (5, -5, 3, 3, 3, 3, -5, 5),
    (5, -5, 3, 3, 3, 3, -5, 5),
    (20, -5, 15, 3, 3, 15, -5, 20),
    (-25, -45, -5, -5, -5, -5, -45, -25),
    (120, -25, 20, 5, 5, 20, -25, 120),
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIRECTORY = PROJECT_ROOT / "models"


def opponent(color: int) -> int:
    return WHITE if color == BLACK else BLACK


@dataclass(frozen=True)
class LegalMove:
    x: int
    y: int
    flips: tuple[tuple[int, int], ...]


class HeadlessOthello:
    """Minimal 8x8 rules engine used only by the benchmark."""

    def __init__(self) -> None:
        self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.board[3][3] = WHITE
        self.board[4][4] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK

    @staticmethod
    def _inside(x: int, y: int) -> bool:
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def _flips_for(self, color: int, x: int, y: int) -> tuple[tuple[int, int], ...]:
        if not self._inside(x, y) or self.board[y][x] != EMPTY:
            return ()

        flips: list[tuple[int, int]] = []
        other = opponent(color)

        for dx, dy in DIRECTIONS:
            line: list[tuple[int, int]] = []
            scan_x, scan_y = x + dx, y + dy

            while self._inside(scan_x, scan_y) and self.board[scan_y][scan_x] == other:
                line.append((scan_x, scan_y))
                scan_x += dx
                scan_y += dy

            if (
                line
                and self._inside(scan_x, scan_y)
                and self.board[scan_y][scan_x] == color
            ):
                flips.extend(line)

        return tuple(flips)

    def legal_moves(self, color: int) -> list[LegalMove]:
        moves: list[LegalMove] = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                flips = self._flips_for(color, x, y)
                if flips:
                    moves.append(LegalMove(x=x, y=y, flips=flips))
        return moves

    def play(self, color: int, move: LegalMove) -> None:
        if self.board[move.y][move.x] != EMPTY or not move.flips:
            raise ValueError(f"Illegal move ({move.x}, {move.y})")

        self.board[move.y][move.x] = color
        for x, y in move.flips:
            self.board[y][x] = color

    def undo(self, color: int, move: LegalMove) -> None:
        """Undo a move previously applied with play()."""
        self.board[move.y][move.x] = EMPTY
        other = opponent(color)
        for x, y in move.flips:
            self.board[y][x] = other

    def score(self) -> dict[int, int]:
        return {
            BLACK: sum(row.count(BLACK) for row in self.board),
            WHITE: sum(row.count(WHITE) for row in self.board),
        }


class Player(Protocol):
    name: str

    def choose_move(
        self,
        game: HeadlessOthello,
        color: int,
        legal_moves: Sequence[LegalMove],
        rng: random.Random,
    ) -> tuple[int, int]: ...


class RandomPlayer:
    name = "Random"

    def choose_move(
        self,
        game: HeadlessOthello,
        color: int,
        legal_moves: Sequence[LegalMove],
        rng: random.Random,
    ) -> tuple[int, int]:
        del game, color
        move = rng.choice(legal_moves)
        return move.x, move.y


class GreedyPlayer:
    name = "Greedy"

    def choose_move(
        self,
        game: HeadlessOthello,
        color: int,
        legal_moves: Sequence[LegalMove],
        rng: random.Random,
    ) -> tuple[int, int]:
        del game, color, rng
        # max() keeps the first row-major move when several moves flip the same
        # number of pieces, matching the current greedy player in computer.py.
        move = max(legal_moves, key=lambda candidate: len(candidate.flips))
        return move.x, move.y


class MinimaxPlayer:
    """Depth-limited minimax player with alpha-beta pruning."""

    WIN_SCORE = 1_000_000

    def __init__(self, depth: int = DEFAULT_MINIMAX_DEPTH) -> None:
        if depth < 1:
            raise ValueError("Minimax depth must be at least 1")
        self.depth = depth
        self.name = f"Minimax (depth {depth}, alpha-beta)"

    @staticmethod
    def _ordered_moves(legal_moves: Sequence[LegalMove]) -> list[LegalMove]:
        # Strong moves first make alpha-beta cutoffs happen earlier. The tuple
        # also provides a stable, deterministic tie-break order.
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

    @classmethod
    def _terminal_value(cls, game: HeadlessOthello, root_color: int) -> int:
        scores = game.score()
        difference = scores[root_color] - scores[opponent(root_color)]
        if difference > 0:
            return cls.WIN_SCORE + difference
        if difference < 0:
            return -cls.WIN_SCORE + difference
        return 0

    @staticmethod
    def _heuristic_value(game: HeadlessOthello, root_color: int) -> int:
        other = opponent(root_color)
        root_discs = 0
        other_discs = 0
        positional = 0

        for y, row in enumerate(game.board):
            for x, square in enumerate(row):
                if square == root_color:
                    root_discs += 1
                    positional += POSITION_WEIGHTS[y][x]
                elif square == other:
                    other_discs += 1
                    positional -= POSITION_WEIGHTS[y][x]

        occupied = root_discs + other_discs
        empty = BOARD_SIZE * BOARD_SIZE - occupied
        disc_difference = root_discs - other_discs
        mobility_difference = len(game.legal_moves(root_color)) - len(
            game.legal_moves(other)
        )

        # Piece count matters increasingly near the end. Earlier in the game,
        # mobility and stable positional advantages are better indicators.
        if empty > 20:
            disc_weight = 1
        elif empty > 10:
            disc_weight = 4
        else:
            disc_weight = 12

        return positional + 10 * mobility_difference + disc_weight * disc_difference

    def _alpha_beta(
        self,
        game: HeadlessOthello,
        color: int,
        depth: int,
        alpha: float,
        beta: float,
        root_color: int,
    ) -> int:
        legal_moves = game.legal_moves(color)

        if not legal_moves:
            if not game.legal_moves(opponent(color)):
                return self._terminal_value(game, root_color)
            if depth <= 0:
                return self._heuristic_value(game, root_color)
            # A pass changes the active color but does not consume search depth.
            return self._alpha_beta(
                game,
                opponent(color),
                depth,
                alpha,
                beta,
                root_color,
            )

        if depth <= 0:
            return self._heuristic_value(game, root_color)

        ordered_moves = self._ordered_moves(legal_moves)
        if color == root_color:
            value = -self.WIN_SCORE * 2
            for move in ordered_moves:
                game.play(color, move)
                try:
                    child_value = self._alpha_beta(
                        game,
                        opponent(color),
                        depth - 1,
                        alpha,
                        beta,
                        root_color,
                    )
                finally:
                    game.undo(color, move)
                value = max(value, child_value)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = self.WIN_SCORE * 2
        for move in ordered_moves:
            game.play(color, move)
            try:
                child_value = self._alpha_beta(
                    game,
                    opponent(color),
                    depth - 1,
                    alpha,
                    beta,
                    root_color,
                )
            finally:
                game.undo(color, move)
            value = min(value, child_value)
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
        del rng
        best_move = self._ordered_moves(legal_moves)[0]
        best_value = -self.WIN_SCORE * 2
        alpha = float("-inf")

        for move in self._ordered_moves(legal_moves):
            game.play(color, move)
            try:
                value = self._alpha_beta(
                    game,
                    opponent(color),
                    self.depth - 1,
                    alpha,
                    float("inf"),
                    color,
                )
            finally:
                game.undo(color, move)

            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)

        return best_move.x, best_move.y


class DQNPlayer:
    def __init__(self, checkpoint: Path) -> None:
        try:
            import torch
            from torch import nn
            import torch.nn.functional as functional
        except ImportError as exc:
            raise RuntimeError(
                "DQN checkpoints require PyTorch. Run the benchmark with the "
                "same Python environment used to run or train COSMOS."
            ) from exc

        class QNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer1 = nn.Linear(64, 128)
                self.layer2 = nn.Linear(128, 128)
                self.layer3 = nn.Linear(128, 128)
                self.layer4 = nn.Linear(128, 64)

            def forward(self, state):
                state = functional.relu(self.layer1(state))
                state = functional.relu(self.layer2(state))
                state = functional.relu(self.layer3(state))
                return self.layer4(state)

        self._torch = torch
        self.checkpoint = checkpoint.resolve()
        self.network = QNet()

        try:
            state_dict = torch.load(
                self.checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            # Compatibility with older PyTorch releases without weights_only.
            state_dict = torch.load(self.checkpoint, map_location="cpu")

        self.network.load_state_dict(state_dict)
        self.network.eval()

        try:
            relative_name = self.checkpoint.relative_to(PROJECT_ROOT)
        except ValueError:
            relative_name = self.checkpoint
        self.name = f"DQN ({relative_name})"

    @staticmethod
    def _encode_board(game: HeadlessOthello, color: int) -> list[int]:
        other = opponent(color)
        values = {EMPTY: 0, color: 1, other: -1}
        return [values[square] for row in game.board for square in row]

    def _get_q_values(self, game: HeadlessOthello, color: int):
        """Return raw Q-values tensor for the current position."""
        state = self._torch.tensor(self._encode_board(game, color), dtype=self._torch.float32)
        with self._torch.inference_mode():
            return self.network(state)

    def get_value_prediction(self, game: HeadlessOthello, color: int) -> float:
        """Return the estimated value (max Q over legal moves) for the current position."""
        q_values = self._get_q_values(game, color)
        legal_moves = game.legal_moves(color)
        if not legal_moves:
            return 0.0
        max_q = max(
            q_values[move.y * BOARD_SIZE + move.x].item()
            for move in legal_moves
        )
        return max_q

    def choose_move(
        self,
        game: HeadlessOthello,
        color: int,
        legal_moves: Sequence[LegalMove],
        rng: random.Random,
    ) -> tuple[int, int]:
        del rng
        q_values = self._get_q_values(game, color)

        # Restrict the argmax to legal board positions.
        move = max(
            legal_moves,
            key=lambda candidate: q_values[candidate.y * BOARD_SIZE + candidate.x].item(),
        )
        return move.x, move.y


@dataclass(frozen=True)
class ModelOption:
    spec: str
    label: str


@dataclass(frozen=True)
class GameResult:
    winner: int | None
    player_scores: tuple[int, int]
    player_colors: tuple[int, int]
    moves: int


@dataclass
class MatchStats:
    games: int = 0
    draws: int = 0
    wins: list[int] = field(default_factory=lambda: [0, 0])
    wins_as_black: list[int] = field(default_factory=lambda: [0, 0])
    wins_as_white: list[int] = field(default_factory=lambda: [0, 0])
    games_as_black: list[int] = field(default_factory=lambda: [0, 0])
    total_discs: list[int] = field(default_factory=lambda: [0, 0])
    total_moves: int = 0

    def record(self, result: GameResult) -> None:
        self.games += 1
        self.total_moves += result.moves
        for player_index, score in enumerate(result.player_scores):
            self.total_discs[player_index] += score
            if result.player_colors[player_index] == BLACK:
                self.games_as_black[player_index] += 1

        if result.winner is None:
            self.draws += 1
            return

        self.wins[result.winner] += 1
        if result.player_colors[result.winner] == BLACK:
            self.wins_as_black[result.winner] += 1
        else:
            self.wins_as_white[result.winner] += 1


def discover_models() -> list[ModelOption]:
    options = [
        ModelOption("random", "Random"),
        ModelOption("greedy", "Greedy"),
        ModelOption(
            "minimax",
            f"Minimax with alpha-beta pruning (depth {DEFAULT_MINIMAX_DEPTH})",
        ),
    ]
    if MODELS_DIRECTORY.exists():
        for checkpoint in sorted(MODELS_DIRECTORY.rglob("*.pth")):
            relative = checkpoint.relative_to(PROJECT_ROOT)
            options.append(ModelOption(str(relative), f"DQN: {relative}"))
    return options


def print_model_list(options: Sequence[ModelOption]) -> None:
    print("Available models:")
    for index, option in enumerate(options, start=1):
        print(f"  {index:>2}. {option.label}")


def prompt_for_model(title: str, options: Sequence[ModelOption]) -> str:
    while True:
        raw = input(f"{title} [number]: ").strip()
        try:
            index = int(raw)
        except ValueError:
            print("Please enter one of the model numbers shown above.")
            continue
        if 1 <= index <= len(options):
            return options[index - 1].spec
        print(f"Please enter a number from 1 to {len(options)}.")


def prompt_for_games(default: int = 100) -> int:
    while True:
        raw = input(f"Number of games [{default}]: ").strip()
        if not raw:
            return default
        try:
            games = int(raw)
        except ValueError:
            print("Please enter a positive whole number.")
            continue
        if games > 0:
            return games
        print("Please enter a positive whole number.")


def normalize_checkpoint(raw_spec: str) -> Path:
    spec = raw_spec[4:] if raw_spec.lower().startswith("dqn:") else raw_spec
    checkpoint = Path(spec).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        raise ValueError(f"DQN checkpoint not found: {checkpoint}")
    if checkpoint.suffix.lower() != ".pth":
        raise ValueError(f"DQN checkpoint must be a .pth file: {checkpoint}")
    return checkpoint


def build_player(spec: str) -> Player:
    normalized = spec.strip().lower()
    if normalized == "random":
        return RandomPlayer()
    if normalized == "greedy":
        return GreedyPlayer()
    if normalized == "minimax":
        return MinimaxPlayer()
    if normalized.startswith("minimax:"):
        raw_depth = normalized.partition(":")[2]
        try:
            depth = int(raw_depth)
        except ValueError as exc:
            raise ValueError(
                f"Invalid minimax depth {raw_depth!r}; use a positive whole number"
            ) from exc
        return MinimaxPlayer(depth)
    return DQNPlayer(normalize_checkpoint(spec.strip()))


def play_game(
    players: tuple[Player, Player],
    player_colors: tuple[int, int],
    rng: random.Random,
) -> GameResult:
    game = HeadlessOthello()
    player_for_color = {
        player_colors[0]: 0,
        player_colors[1]: 1,
    }
    color = BLACK
    consecutive_passes = 0
    moves_played = 0

    while consecutive_passes < 2:
        legal_moves = game.legal_moves(color)
        if not legal_moves:
            consecutive_passes += 1
            color = opponent(color)
            continue

        consecutive_passes = 0
        player_index = player_for_color[color]
        selected = players[player_index].choose_move(game, color, legal_moves, rng)
        legal_by_coordinate = {(move.x, move.y): move for move in legal_moves}
        if selected not in legal_by_coordinate:
            raise RuntimeError(
                f"{players[player_index].name} selected illegal move {selected}"
            )

        game.play(color, legal_by_coordinate[selected])
        moves_played += 1
        color = opponent(color)

    scores_by_color = game.score()
    player_scores = (
        scores_by_color[player_colors[0]],
        scores_by_color[player_colors[1]],
    )
    if player_scores[0] == player_scores[1]:
        winner = None
    else:
        winner = 0 if player_scores[0] > player_scores[1] else 1

    return GameResult(
        winner=winner,
        player_scores=player_scores,
        player_colors=player_colors,
        moves=moves_played,
    )


def run_match(
    players: tuple[Player, Player],
    games: int,
    seed: int,
    show_progress: bool,
) -> tuple[MatchStats, float]:
    rng = random.Random(seed)
    stats = MatchStats()
    progress_interval = max(1, games // 10)
    started = time.perf_counter()

    for game_index in range(games):
        # Alternating colors prevents either player from always receiving the
        # first-move advantage. Player 1 is Black in game 1.
        colors = (BLACK, WHITE) if game_index % 2 == 0 else (WHITE, BLACK)
        stats.record(play_game(players, colors, rng))

        completed = game_index + 1
        if show_progress and (completed % progress_interval == 0 or completed == games):
            print(f"\rCompleted {completed}/{games} games", flush=True)

    if show_progress:
        print()
    return stats, time.perf_counter() - started


def percentage(count: int, total: int) -> str:
    return f"{100.0 * count / total:.2f}%"


def print_results(
    players: tuple[Player, Player],
    stats: MatchStats,
    elapsed: float,
    seed: int,
) -> None:
    print("\nMatchup")
    print(f"  Player 1: {players[0].name}")
    print(f"  Player 2: {players[1].name}")
    print(f"  Games:    {stats.games} (colors alternated)")
    print(f"  Seed:     {seed}")

    print("\nResults")
    for index, player in enumerate(players):
        white_games = stats.games - stats.games_as_black[index]
        average_discs = stats.total_discs[index] / stats.games
        print(
            f"  Player {index + 1} wins: {stats.wins[index]:>5} "
            f"({percentage(stats.wins[index], stats.games)})  {player.name}"
        )
        print(
            f"    as Black: {stats.wins_as_black[index]:>5}/"
            f"{stats.games_as_black[index]:<5}  "
            f"as White: {stats.wins_as_white[index]:>5}/{white_games:<5}  "
            f"average discs: {average_discs:.2f}"
        )
    print(
        f"  Draws:         {stats.draws:>5} "
        f"({percentage(stats.draws, stats.games)})"
    )
    print(f"  Average moves: {stats.total_moves / stats.games:.2f}")
    print(f"  Elapsed time:  {elapsed:.2f} seconds")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare COSMOS random, greedy, minimax, and DQN Othello players.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Model specs can be 'random', 'greedy', 'minimax', "
            "'minimax:DEPTH', or a path to a .pth file.\n"
            "Omit both players to use the interactive model picker."
        ),
    )
    parser.add_argument("--player-1", help="First model spec")
    parser.add_argument("--player-2", help="Second model spec")
    parser.add_argument(
        "-n",
        "--games",
        type=int,
        help="Number of games (interactive default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible games (default: 0)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Do not print progress while games run",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model specs and exit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    options = discover_models()

    if args.list_models:
        print_model_list(options)
        return 0

    if bool(args.player_1) != bool(args.player_2):
        print("Error: provide both --player-1 and --player-2, or neither.", file=sys.stderr)
        return 2

    if args.player_1:
        player_specs = (args.player_1, args.player_2)
        games = args.games if args.games is not None else 100
    else:
        print_model_list(options)
        player_specs = (
            prompt_for_model("Choose Player 1", options),
            prompt_for_model("Choose Player 2", options),
        )
        games = args.games if args.games is not None else prompt_for_games()

    if games <= 0:
        print("Error: --games must be a positive whole number.", file=sys.stderr)
        return 2

    try:
        players = (build_player(player_specs[0]), build_player(player_specs[1]))
        stats, elapsed = run_match(
            players,
            games=games,
            seed=args.seed,
            show_progress=not args.no_progress,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_results(players, stats, elapsed, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
