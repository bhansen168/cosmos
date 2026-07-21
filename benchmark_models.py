#!/usr/bin/env python3
"""Run headless matches between COSMOS Othello players.

Run without arguments for an interactive model picker, or pass model specs:

    python benchmark_models.py --player-1 random --player-2 greedy --games 500
    python benchmark_models.py --player-1 minimax:3 --player-2 greedy --games 100
    python benchmark_models.py --player-1 genetic --player-2 minimax --games 100
    python benchmark_models.py --player-1 bard --player-2 greedy --games 100
    python benchmark_models.py --player-1 ppo --player-2 dqn --games 100

The dqn, bard, genetic, and ppo names resolve to the newest available checkpoint
each time the program starts. PPO uses policy/value-guided search. Explicit
checkpoint paths remain supported when a benchmark must be reproducible against
an older model.

The program is independent of the Pygame game loop, so benchmarks can run
without opening a window. PyTorch is imported only when a DQN or PPO model is
selected.
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
    opponent,  # noqa: F401 - retained as a compatibility re-export
    play_game as play_engine_game,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIRECTORY = PROJECT_ROOT / "models"
GENETIC_MODELS_DIRECTORY = MODELS_DIRECTORY / "genetic"
PPO_MODELS_DIRECTORY = MODELS_DIRECTORY / "ppo"


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


def _newest_checkpoint(candidates: Sequence[Path], model_name: str) -> Path:
    existing: list[Path] = []
    for checkpoint in candidates:
        try:
            if checkpoint.is_file():
                existing.append(checkpoint)
        except OSError:
            continue
    if not existing:
        raise ValueError(
            f"No {model_name} checkpoints found under {MODELS_DIRECTORY}"
        )

    def freshness(checkpoint: Path) -> tuple[int, str]:
        try:
            modified = checkpoint.stat().st_mtime_ns
        except OSError:
            modified = 0
        return modified, checkpoint.as_posix().casefold()

    return max(existing, key=freshness)


def latest_dqn_checkpoint() -> Path:
    completed = [
        checkpoint
        for checkpoint in MODELS_DIRECTORY.rglob("*.pth")
        if "aborted" not in checkpoint.name.casefold()
    ]
    return _newest_checkpoint(completed, "DQN")


def latest_bard_checkpoint() -> Path:
    return _newest_checkpoint(
        list(MODELS_DIRECTORY.rglob("*.bard")),
        "Bard",
    )


def latest_genetic_checkpoint() -> Path:
    # Genetic v2 also has legacy runs directly under models/.
    search_directories = (GENETIC_MODELS_DIRECTORY, MODELS_DIRECTORY)
    latest_files = [
        checkpoint
        for directory in search_directories
        for checkpoint in directory.glob("latest*.json")
    ]
    candidates = latest_files or [
        checkpoint
        for directory in search_directories
        for checkpoint in directory.glob("genetic_gen_*.json")
    ]
    return _newest_checkpoint(candidates, "genetic")


def latest_ppo_checkpoint() -> Path:
    # Older trainer versions wrote directly under models/. Keep those runs
    # discoverable while preferring whichever latest checkpoint is newest.
    search_directories = (PPO_MODELS_DIRECTORY, MODELS_DIRECTORY)
    latest_files = [
        checkpoint
        for directory in search_directories
        for checkpoint in directory.glob("latest*.ppo")
    ]
    candidates = latest_files or [
        checkpoint
        for directory in search_directories
        for checkpoint in directory.glob("*.ppo")
        if checkpoint.name.casefold() != "best.ppo"
    ]
    return _newest_checkpoint(candidates, "PPO")


def _relative_label(checkpoint: Path) -> str:
    try:
        return str(checkpoint.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(checkpoint)


def discover_models() -> list[ModelOption]:
    options = [
        ModelOption("random", "Random"),
        ModelOption("greedy", "Greedy"),
        ModelOption(
            "minimax",
            f"Minimax with alpha-beta pruning (depth {DEFAULT_MINIMAX_DEPTH})",
        ),
    ]

    learned_models = (
        ("dqn", "DQN", latest_dqn_checkpoint),
        ("bard", "Bard supervised", latest_bard_checkpoint),
        ("genetic", "Genetic", latest_genetic_checkpoint),
        ("ppo", "PPO", latest_ppo_checkpoint),
    )
    for spec, label, resolver in learned_models:
        try:
            checkpoint = resolver()
        except ValueError:
            continue
        options.append(
            ModelOption(spec, f"{label} (latest: {_relative_label(checkpoint)})")
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


def _normalize_explicit_checkpoint(
    raw_path: str,
    model_name: str,
    extension: str,
) -> Path:
    checkpoint = Path(raw_path).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        raise ValueError(f"{model_name} checkpoint not found: {checkpoint}")
    if checkpoint.suffix.lower() != extension:
        raise ValueError(
            f"{model_name} checkpoint must be a {extension} file: {checkpoint}"
        )
    return checkpoint


def normalize_checkpoint(raw_spec: str) -> Path:
    stripped = raw_spec.strip()
    lowered = stripped.lower()
    if lowered == "dqn":
        return latest_dqn_checkpoint()
    spec = stripped[4:] if lowered.startswith("dqn:") else stripped
    return _normalize_explicit_checkpoint(spec, "DQN", ".pth")


def normalize_ppo_checkpoint(raw_spec: str) -> Path:
    stripped = raw_spec.strip()
    lowered = stripped.lower()
    if lowered == "ppo":
        return latest_ppo_checkpoint()
    spec = stripped[len("ppo:") :] if lowered.startswith("ppo:") else stripped
    return _normalize_explicit_checkpoint(spec, "PPO", ".ppo")


def normalize_genetic_checkpoint(raw_spec: str) -> Path:
    stripped = raw_spec.strip()
    lowered = stripped.lower()
    if lowered in ("genetic", "ga"):
        return latest_genetic_checkpoint()
    if lowered.startswith("genetic:"):
        spec = stripped[len("genetic:") :]
    elif lowered.startswith("ga:"):
        spec = stripped[len("ga:") :]
    else:
        spec = stripped
    return _normalize_explicit_checkpoint(spec, "Genetic", ".json")


def normalize_bard_checkpoint(raw_spec: str) -> Path:
    stripped = raw_spec.strip()
    lowered = stripped.lower()
    if lowered == "bard":
        return latest_bard_checkpoint()
    spec = stripped[len("bard:") :] if lowered.startswith("bard:") else stripped
    return _normalize_explicit_checkpoint(spec, "Bard", ".bard")


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
        normalized in ("genetic", "ga")
        or normalized.startswith("genetic:")
        or normalized.startswith("ga:")
        or normalized.endswith(".json")
    ):
        return GeneticPlayer.from_checkpoint(normalize_genetic_checkpoint(stripped))
    if (
        normalized == "ppo"
        or normalized.startswith("ppo:")
        or normalized.endswith(".ppo")
    ):
        try:
            from ppo_model import PPOPlayer
        except ImportError as exc:
            raise RuntimeError(
                "PPO checkpoints require PyTorch. Run the benchmark with the "
                "same Python environment used to train PPO."
            ) from exc
        return PPOPlayer(normalize_ppo_checkpoint(stripped))
    if (
        normalized == "dqn"
        or normalized.startswith("dqn:")
        or normalized.endswith(".pth")
    ):
        return DQNPlayer(normalize_checkpoint(stripped))
    raise ValueError(
        f"Unknown model {stripped!r}; use random, greedy, minimax, dqn, bard, "
        "genetic, ppo, or an explicit checkpoint path"
    )


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
            "Compare COSMOS random, greedy, minimax, DQN, Bard, genetic, and "
            "PPO Othello players."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Use 'random', 'greedy', 'minimax', 'dqn', 'bard', 'genetic', "
            "or 'ppo'. Learned-model names automatically use their latest "
            "checkpoint. Custom specs such as 'minimax:DEPTH', "
            "'bard:PATH.bard', 'genetic:PATH.json', 'ppo:PATH.ppo', and "
            "'dqn:PATH.pth' are also supported.\n"
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
