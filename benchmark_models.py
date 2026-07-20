#!/usr/bin/env python3
"""Run headless matches between COSMOS Othello players.

Run without arguments for an interactive model picker, or pass model specs:

    python benchmark_models.py --player-1 random --player-2 greedy --games 500
    python benchmark_models.py --player-1 minimax:3 --player-2 greedy --games 100
    python benchmark_models.py --player-1 genetic:models/genetic/genetic_gen_0009.json \
        --player-2 minimax --games 100
    python benchmark_models.py --player-1 greedy \
        --player-2 models/v1/othello_100k.pth --games 100
    python benchmark_models.py --player-1 bard --player-2 greedy --games 100

The program is independent of the Pygame game loop, so benchmarks can run
without opening a window. PyTorch is imported only when a DQN is selected.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from computer import Computer as GreedyPlayer
from computer import RandomComputer as RandomPlayer
from genetic_model import GeneticPlayer
from minimax_model import DEFAULT_MINIMAX_DEPTH, MinimaxPlayer
from othello_engine import (
    BLACK,
    WHITE,
    HeadlessOthello,
    LegalMove,
    Player,
    opponent,
    play_game as play_engine_game,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIRECTORY = PROJECT_ROOT / "models"
GENETIC_MODELS_DIRECTORY = MODELS_DIRECTORY / "genetic"
DEFAULT_BARD_CHECKPOINT = MODELS_DIRECTORY / "supervised" / "wthor-kaggle.bard"
BENCHMARK_MINIMAX_DEPTHS = (1, 2, 3, 4)


class DQNPlayer:
    """Benchmark adapter using the original computerRL.py DQN implementation."""

    def __init__(self, checkpoint: Path) -> None:
        try:
            from computerRL import (
                encode_state,
                index_to_coord,
                legal_moves_to_np_arr,
                load_agent,
            )
        except ImportError as exc:
            raise RuntimeError(
                "DQN checkpoints require PyTorch. Run the benchmark with the "
                "same Python environment used to run or train COSMOS."
            ) from exc

        self.checkpoint = checkpoint.resolve()
        self.agent = load_agent(self.checkpoint)
        self._encode_state = encode_state
        self._index_to_coord = index_to_coord
        self._legal_moves_to_np_arr = legal_moves_to_np_arr
        hidden_size = self.agent.policyNet.layer1.out_features

        try:
            relative_name = self.checkpoint.relative_to(PROJECT_ROOT)
        except ValueError:
            relative_name = self.checkpoint
        self.name = f"DQN ({relative_name}, {hidden_size} hidden units)"

    def get_value_prediction(self, game: HeadlessOthello, color: int) -> float:
        legal = game.get_all_legal_moves(color)
        if not legal:
            return 0.0
        return self.agent.get_value_prediction(
            self._encode_state(game.board,color),
            self._legal_moves_to_np_arr(legal,self.agent.actionDim),
        )

    def choose_move(
        self,
        game: HeadlessOthello,
        color: int,
        legal_moves: Sequence[LegalMove],
        rng: random.Random,
    ) -> tuple[int, int]:
        del rng
        legal = [(move.x,move.y) for move in legal_moves]
        action = self.agent.select_action(
            self._encode_state(game.board,color),
            self._legal_moves_to_np_arr(legal,self.agent.actionDim),
            0.0,
        )
        y,x = self._index_to_coord(action)
        return x,y


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
    ]

    for depth in BENCHMARK_MINIMAX_DEPTHS:
        spec = "minimax" if depth == DEFAULT_MINIMAX_DEPTH else f"minimax:{depth}"
        default_label = " — default" if depth == DEFAULT_MINIMAX_DEPTH else ""
        options.append(
            ModelOption(
                spec,
                f"Minimax with alpha-beta pruning (depth {depth}){default_label}",
            )
        )

    if MODELS_DIRECTORY.exists():
        for checkpoint in sorted(MODELS_DIRECTORY.rglob("*.bard")):
            relative = checkpoint.relative_to(PROJECT_ROOT)
            is_default = checkpoint.resolve() == DEFAULT_BARD_CHECKPOINT.resolve()
            spec = "bard" if is_default else f"bard:{relative}"
            default_label = " — default" if is_default else ""
            options.append(
                ModelOption(
                    spec,
                    f"Bard supervised: {relative}{default_label}",
                )
            )

        for checkpoint in sorted(MODELS_DIRECTORY.rglob("*.pth")):
            relative = checkpoint.relative_to(PROJECT_ROOT)
            options.append(ModelOption(str(relative), f"DQN: {relative}"))

    if GENETIC_MODELS_DIRECTORY.exists():
        for checkpoint in sorted(
            GENETIC_MODELS_DIRECTORY.rglob("genetic_gen_*.json")
        ):
            relative = checkpoint.relative_to(PROJECT_ROOT)
            options.append(
                ModelOption(f"genetic:{relative}", f"Genetic: {relative}")
            )

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


def normalize_genetic_checkpoint(raw_spec: str) -> Path:
    lowered = raw_spec.lower()
    if lowered.startswith("genetic:"):
        spec = raw_spec[len("genetic:") :]
    elif lowered.startswith("ga:"):
        spec = raw_spec[len("ga:") :]
    else:
        spec = raw_spec

    checkpoint = Path(spec).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        raise ValueError(f"Genetic checkpoint not found: {checkpoint}")
    if checkpoint.suffix.lower() != ".json":
        raise ValueError(f"Genetic checkpoint must be a .json file: {checkpoint}")
    return checkpoint


def normalize_bard_checkpoint(raw_spec: str) -> Path:
    lowered = raw_spec.lower()
    if lowered == "bard":
        checkpoint = DEFAULT_BARD_CHECKPOINT
    else:
        spec = raw_spec[len("bard:") :] if lowered.startswith("bard:") else raw_spec
        checkpoint = Path(spec).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        raise ValueError(f"Bard checkpoint not found: {checkpoint}")
    if checkpoint.suffix.lower() != ".bard":
        raise ValueError(f"Bard checkpoint must be a .bard file: {checkpoint}")
    return checkpoint


def build_player(spec: str) -> Player:
    stripped = spec.strip()
    normalized = stripped.lower()
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
    if (
        normalized == "bard"
        or normalized.startswith("bard:")
        or normalized.endswith(".bard")
    ):
        try:
            from computer import ComputerSupervised
        except ImportError as exc:
            raise RuntimeError(
                "Bard checkpoints require NumPy and XGBoost. Run the benchmark "
                "with the same Python environment used to train Bard."
            ) from exc
        return ComputerSupervised(path=normalize_bard_checkpoint(stripped))
    if (
        normalized.startswith("genetic:")
        or normalized.startswith("ga:")
        or normalized.endswith(".json")
    ):
        return GeneticPlayer.from_checkpoint(normalize_genetic_checkpoint(stripped))
    return DQNPlayer(normalize_checkpoint(stripped))


def play_game(
    players: tuple[Player, Player],
    player_colors: tuple[int, int],
    rng: random.Random,
) -> GameResult:
    black_index = player_colors.index(BLACK)
    white_index = player_colors.index(WHITE)
    outcome = play_engine_game(players[black_index], players[white_index], rng)

    scores_by_color = {
        BLACK: outcome.black_score,
        WHITE: outcome.white_score,
    }
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
        moves=outcome.moves,
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
        if show_progress and (
            completed % progress_interval == 0 or completed == games
        ):
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
        description=(
            "Compare COSMOS random, greedy, minimax, Bard, genetic, and DQN "
            "Othello players."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Model specs can be 'random', 'greedy', 'minimax', "
            "'minimax:DEPTH', 'bard', 'bard:PATH.bard', "
            "'genetic:PATH.json', or a path to a .pth file.\n"
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
        print(
            "Error: provide both --player-1 and --player-2, or neither.",
            file=sys.stderr,
        )
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
