#!/usr/bin/env python3
"""Watch two COSMOS Othello models play with autoplay and step controls."""

from __future__ import annotations

import argparse
import os
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]

from benchmark_models import (
    build_player,
    discover_models,
    print_model_list,
    prompt_for_model,
    DQNPlayer,
)
from othello_engine import (
    BLACK,
    BOARD_SIZE,
    WHITE,
    HeadlessOthello,
    LegalMove,
    Player,
    opponent,
)


COLOR_NAMES = {BLACK: "Black", WHITE: "White"}


@dataclass(frozen=True)
class TurnRecord:
    turn: int
    color: int
    coordinate: tuple[int, int] | None
    black_score: int
    white_score: int

    @property
    def text(self) -> str:
        color_name = COLOR_NAMES[self.color]
        if self.coordinate is None:
            action = "passes"
        else:
            x, y = self.coordinate
            action = f"{chr(ord('A') + x)}{y + 1}"
        return (
            f"{self.turn:>2}. {color_name:<5} {action:<6} "
            f"({self.black_score}-{self.white_score})"
        )


class SpectatorMatch:
    """UI-independent match state used by the real-time viewer."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.game = HeadlessOthello()
        self.current_color = BLACK
        self.last_move: tuple[int, int] | None = None
        self.history: list[TurnRecord] = []
        self.game_over = False
        self.winner: int | None = None

    def position_key(self) -> tuple[int, tuple[tuple[int, ...], ...]]:
        return self.current_color, tuple(tuple(row) for row in self.game.board)

    def legal_moves(self) -> list[LegalMove]:
        if self.game_over:
            return []
        return self.game.legal_moves(self.current_color)

    def scores(self) -> dict[int, int]:
        return self.game.score()

    def _record(self, color: int, coordinate: tuple[int, int] | None) -> None:
        scores = self.scores()
        self.history.append(
            TurnRecord(
                turn=len(self.history) + 1,
                color=color,
                coordinate=coordinate,
                black_score=scores[BLACK],
                white_score=scores[WHITE],
            )
        )

    def _finish(self) -> None:
        scores = self.scores()
        self.game_over = True
        if scores[BLACK] == scores[WHITE]:
            self.winner = None
        else:
            self.winner = BLACK if scores[BLACK] > scores[WHITE] else WHITE

    def apply_move(self, coordinate: tuple[int, int]) -> None:
        if self.game_over:
            raise ValueError("The game is already over")
        legal_by_coordinate = {
            (move.x, move.y): move for move in self.game.legal_moves(self.current_color)
        }
        if coordinate not in legal_by_coordinate:
            raise ValueError(
                f"Illegal {COLOR_NAMES[self.current_color]} move: {coordinate}"
            )

        color = self.current_color
        self.game.play(color, legal_by_coordinate[coordinate])
        self.last_move = coordinate
        self._record(color, coordinate)

        next_color = opponent(color)
        if not self.game.legal_moves(next_color) and not self.game.legal_moves(color):
            self._finish()
        else:
            self.current_color = next_color

    def pass_turn(self) -> None:
        if self.game_over:
            raise ValueError("The game is already over")
        if self.game.legal_moves(self.current_color):
            raise ValueError(f"{COLOR_NAMES[self.current_color]} has a legal move")

        color = self.current_color
        self._record(color, None)
        next_color = opponent(color)
        if not self.game.legal_moves(next_color):
            self._finish()
        else:
            self.current_color = next_color


@dataclass
class PendingMove:
    result_queue: queue.Queue[tuple[bool, object]]
    thread: threading.Thread
    color: int
    position_key: tuple[int, tuple[tuple[int, ...], ...]]


class SpectatorApp:
    WINDOW_WIDTH = 1080
    WINDOW_HEIGHT = 760
    BOARD_LEFT = 40
    BOARD_TOP = 54
    SQUARE_SIZE = 76
    BOARD_PIXELS = SQUARE_SIZE * BOARD_SIZE
    PANEL_LEFT = BOARD_LEFT + BOARD_PIXELS + 35

    BACKGROUND = (237, 239, 242)
    BOARD_GREEN = (38, 139, 78)
    BOARD_LIGHT = (66, 166, 102)
    GRID = (18, 75, 45)
    BLACK_PIECE = (28, 31, 35)
    WHITE_PIECE = (246, 246, 242)
    GOLD = (244, 183, 64)
    TEXT = (35, 40, 47)
    MUTED = (103, 111, 122)
    PANEL = (255, 255, 255)
    RED = (187, 53, 53)

    def __init__(
        self,
        black_player: Player,
        white_player: Player,
        delay: float,
        seed: int,
        start_paused: bool,
    ) -> None:
        assert pygame is not None
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("COSMOS - Model Spectator")
        logo_path = Path(__file__).resolve().parent / "logo.png"
        if logo_path.is_file():
            pygame.display.set_icon(pygame.image.load(str(logo_path)))

        self.title_font = pygame.font.SysFont("Segoe UI", 28, bold=True)
        self.heading_font = pygame.font.SysFont("Segoe UI", 19, bold=True)
        self.body_font = pygame.font.SysFont("Segoe UI", 17)
        self.small_font = pygame.font.SysFont("Consolas", 14)
        self.piece_font = pygame.font.SysFont("Segoe UI", 17, bold=True)
        self.clock = pygame.time.Clock()

        self.players = {BLACK: black_player, WHITE: white_player}
        self.match = SpectatorMatch()
        self.rng = random.Random(seed)
        self.delay = max(0.0, delay)
        self.paused = start_paused
        self.step_requests = 0
        self.next_action_at = time.monotonic() + self.delay
        self.pending: PendingMove | None = None
        self.message = ""
        self.running = True

    @staticmethod
    def _fit_text(font: Any, text: str, width: int) -> str:
        if font.size(text)[0] <= width:
            return text
        shortened = text
        while shortened and font.size(shortened + "…")[0] > width:
            shortened = shortened[:-1]
        return shortened + "…"

    def _consume_step_if_needed(self) -> bool:
        if not self.paused:
            return True
        if self.step_requests <= 0:
            return False
        self.step_requests -= 1
        return True

    def _start_model_move(self, legal_moves: Sequence[LegalMove]) -> None:
        color = self.match.current_color
        snapshot = self.match.game.clone()
        snapshot_legal = snapshot.legal_moves(color)
        result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)
        move_rng = random.Random(self.rng.randrange(0, 2**63))
        player = self.players[color]

        def choose() -> None:
            try:
                coordinate = player.choose_move(
                    snapshot,
                    color,
                    snapshot_legal,
                    move_rng,
                )
                result_queue.put((True, coordinate))
            except Exception as exc:  # Report model failures in the GUI.
                result_queue.put((False, exc))

        thread = threading.Thread(target=choose, daemon=True)
        self.pending = PendingMove(
            result_queue=result_queue,
            thread=thread,
            color=color,
            position_key=self.match.position_key(),
        )
        thread.start()

    def _poll_pending_move(self) -> None:
        if self.pending is None:
            return
        try:
            succeeded, result = self.pending.result_queue.get_nowait()
        except queue.Empty:
            return

        pending = self.pending
        self.pending = None
        if not succeeded:
            self.message = f"Model error: {result}"
            self.paused = True
            return
        if (
            pending.color != self.match.current_color
            or pending.position_key != self.match.position_key()
        ):
            self.message = "Discarded a move calculated for an old position."
            return

        try:
            coordinate = tuple(result)  # type: ignore[arg-type]
            if len(coordinate) != 2:
                raise ValueError(f"invalid coordinate {result!r}")
            self.match.apply_move((int(coordinate[0]), int(coordinate[1])))
            self.message = ""
        except (TypeError, ValueError) as exc:
            self.message = f"Model error: {exc}"
            self.paused = True
        self.next_action_at = time.monotonic() + self.delay

    def _advance_when_ready(self) -> None:
        self._poll_pending_move()
        if self.pending is not None or self.match.game_over:
            return
        if not self.paused and time.monotonic() < self.next_action_at:
            return
        if not self._consume_step_if_needed():
            return

        legal_moves = self.match.legal_moves()
        if not legal_moves:
            self.match.pass_turn()
            self.next_action_at = time.monotonic() + self.delay
            return
        self._start_model_move(legal_moves)

    def _change_speed(self, amount: float) -> None:
        self.delay = _clamp_delay(self.delay + amount)
        self.next_action_at = min(
            self.next_action_at,
            time.monotonic() + self.delay,
        )

    def _handle_event(self, event: Any) -> None:
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_SPACE:
            self.paused = not self.paused
            self.step_requests = 0
            if not self.paused:
                self.next_action_at = time.monotonic()
        elif event.key in (pygame.K_RIGHT, pygame.K_n):
            self.paused = True
            self.step_requests += 1
        elif event.key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self._change_speed(-0.10)
        elif event.key in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
            self._change_speed(0.10)
        elif event.key == pygame.K_r:
            if self.pending is None:
                self.match.reset()
                self.step_requests = 0
                self.next_action_at = time.monotonic() + self.delay
                self.message = "New game started."
            else:
                self.message = "Wait for the current model calculation before restarting."

    def _draw_board(self) -> None:
        board_rect = pygame.Rect(
            self.BOARD_LEFT,
            self.BOARD_TOP,
            self.BOARD_PIXELS,
            self.BOARD_PIXELS,
        )
        pygame.draw.rect(self.screen, self.BOARD_GREEN, board_rect, border_radius=5)

        if self.match.last_move is not None:
            x, y = self.match.last_move
            highlight = pygame.Rect(
                self.BOARD_LEFT + x * self.SQUARE_SIZE,
                self.BOARD_TOP + y * self.SQUARE_SIZE,
                self.SQUARE_SIZE,
                self.SQUARE_SIZE,
            )
            pygame.draw.rect(self.screen, self.BOARD_LIGHT, highlight)

        if not self.match.game_over and self.pending is None:
            for move in self.match.legal_moves():
                center = (
                    self.BOARD_LEFT + move.x * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
                    self.BOARD_TOP + move.y * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
                )
                pygame.draw.circle(self.screen, (31, 108, 65), center, 6)

        for index in range(BOARD_SIZE + 1):
            x = self.BOARD_LEFT + index * self.SQUARE_SIZE
            y = self.BOARD_TOP + index * self.SQUARE_SIZE
            pygame.draw.line(
                self.screen,
                self.GRID,
                (x, self.BOARD_TOP),
                (x, self.BOARD_TOP + self.BOARD_PIXELS),
                2,
            )
            pygame.draw.line(
                self.screen,
                self.GRID,
                (self.BOARD_LEFT, y),
                (self.BOARD_LEFT + self.BOARD_PIXELS, y),
                2,
            )

        radius = int(self.SQUARE_SIZE * 0.36)
        for y, row in enumerate(self.match.game.board):
            for x, square in enumerate(row):
                if square not in (BLACK, WHITE):
                    continue
                center = (
                    self.BOARD_LEFT + x * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
                    self.BOARD_TOP + y * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
                )
                pygame.draw.circle(self.screen, (20, 70, 42), (center[0] + 2, center[1] + 3), radius)
                piece_color = self.BLACK_PIECE if square == BLACK else self.WHITE_PIECE
                pygame.draw.circle(self.screen, piece_color, center, radius)
                pygame.draw.circle(self.screen, self.GRID, center, radius, 1)

        for index in range(BOARD_SIZE):
            column = self.small_font.render(chr(ord("A") + index), True, self.MUTED)
            row = self.small_font.render(str(index + 1), True, self.MUTED)
            self.screen.blit(
                column,
                (
                    self.BOARD_LEFT + index * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 - column.get_width() // 2,
                    self.BOARD_TOP + self.BOARD_PIXELS + 7,
                ),
            )
            self.screen.blit(
                row,
                (
                    self.BOARD_LEFT - 19,
                    self.BOARD_TOP + index * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 - row.get_height() // 2,
                ),
            )

    def _status_text(self) -> tuple[str, tuple[int, int, int]]:
        if self.message.startswith("Model error"):
            return self.message, self.RED
        if self.match.game_over:
            if self.match.winner is None:
                return "Game over — draw", self.TEXT
            return f"Game over — {COLOR_NAMES[self.match.winner]} wins", self.TEXT
        if self.pending is not None:
            return f"{COLOR_NAMES[self.pending.color]} is thinking…", self.TEXT
        if self.paused:
            return "Paused — press → for one turn", self.TEXT
        return f"{COLOR_NAMES[self.match.current_color]} to move", self.TEXT

    def _draw_panel(self) -> None:
        panel_rect = pygame.Rect(self.PANEL_LEFT, 30, 365, 690)
        pygame.draw.rect(self.screen, self.PANEL, panel_rect, border_radius=12)
        x = self.PANEL_LEFT + 22
        width = panel_rect.width - 44
        y = 50

        title = self.title_font.render("Model Spectator", True, self.TEXT)
        self.screen.blit(title, (x, y))
        y += 50

        scores = self.match.scores()
        for color in (BLACK, WHITE):
            piece_color = self.BLACK_PIECE if color == BLACK else self.WHITE_PIECE
            pygame.draw.circle(self.screen, piece_color, (x + 12, y + 12), 11)
            pygame.draw.circle(self.screen, self.GRID, (x + 12, y + 12), 11, 1)
            label = f"{COLOR_NAMES[color]} — {scores[color]}"
            surface = self.heading_font.render(label, True, self.TEXT)
            self.screen.blit(surface, (x + 34, y))
            y += 28
            model_name = self._fit_text(self.body_font, self.players[color].name, width)
            self.screen.blit(self.body_font.render(model_name, True, self.MUTED), (x, y))
            y += 22
            # Show DQN value prediction for the current player if applicable
            if color == self.match.current_color and isinstance(self.players[color], DQNPlayer):
                try:
                    value = self.players[color].get_value_prediction(self.match.game, color)
                    val_text = f"Value: {value:+.3f}"
                    val_color = self.GOLD if value > 0 else (self.RED if value < 0 else self.MUTED)
                    self.screen.blit(self.small_font.render(val_text, True, val_color), (x, y))
                    y += 20
                except Exception:
                    pass
            y += 20

        status, status_color = self._status_text()
        status = self._fit_text(self.heading_font, status, width)
        self.screen.blit(self.heading_font.render(status, True, status_color), (x, y))
        y += 35
        speed = "fastest" if self.delay == 0 else f"{self.delay:.2f}s between turns"
        self.screen.blit(self.body_font.render(f"Speed: {speed}", True, self.MUTED), (x, y))
        y += 37

        self.screen.blit(self.heading_font.render("Recent turns", True, self.TEXT), (x, y))
        y += 28
        recent = self.match.history[-12:]
        if not recent:
            self.screen.blit(self.body_font.render("No moves yet", True, self.MUTED), (x, y))
            y += 22
        else:
            for record in recent:
                line = self.small_font.render(record.text, True, self.TEXT)
                self.screen.blit(line, (x, y))
                y += 21

        controls_y = 585
        self.screen.blit(self.heading_font.render("Controls", True, self.TEXT), (x, controls_y))
        controls = (
            "Space   Pause / resume",
            "→ or N  Play one turn",
            "↑ / +   Faster",
            "↓ / −   Slower",
            "R       Restart",
            "Esc     Close",
        )
        for index, text in enumerate(controls):
            line = self.small_font.render(text, True, self.MUTED)
            self.screen.blit(line, (x, controls_y + 29 + index * 18))

        if self.message and not self.message.startswith("Model error"):
            message = self._fit_text(self.small_font, self.message, width)
            self.screen.blit(self.small_font.render(message, True, self.MUTED), (x, 700))

    def draw(self) -> None:
        self.screen.fill(self.BACKGROUND)
        self._draw_board()
        self._draw_panel()
        pygame.display.flip()

    def run(self) -> None:
        assert pygame is not None
        while self.running:
            for event in pygame.event.get():
                self._handle_event(event)
            self._advance_when_ready()
            self.draw()
            self.clock.tick(60)
        pygame.quit()


def _clamp_delay(delay: float) -> float:
    return round(max(0.0, min(10.0, delay)), 2)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch two COSMOS Othello models play in real time.",
    )
    parser.add_argument("--black", "--player-1", dest="black", help="Black model spec")
    parser.add_argument("--white", "--player-2", dest="white", help="White model spec")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.75,
        help="Seconds between automatic turns (default: 0.75)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--step",
        "--start-paused",
        action="store_true",
        help="Open paused and advance with Right Arrow or N",
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
    if args.delay < 0:
        print("Error: --delay cannot be negative.", file=sys.stderr)
        return 2
    if bool(args.black) != bool(args.white):
        print("Error: provide both --black and --white, or neither.", file=sys.stderr)
        return 2

    if args.black:
        black_spec, white_spec = args.black, args.white
    else:
        print_model_list(options)
        black_spec = prompt_for_model("Choose Black", options)
        white_spec = prompt_for_model("Choose White", options)

    if pygame is None:
        print(
            "Error: Pygame is required for the spectator window. Install pygame "
            "in the Python environment used to run COSMOS.",
            file=sys.stderr,
        )
        return 1

    try:
        black_player = build_player(black_spec)
        white_player = build_player(white_spec)
        app = SpectatorApp(
            black_player,
            white_player,
            delay=_clamp_delay(args.delay),
            seed=args.seed,
            start_paused=args.step,
        )
        app.run()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
