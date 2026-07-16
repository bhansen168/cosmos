#!/usr/bin/env python3
"""Watch two COSMOS Othello models play using the original game.py engine."""

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
    ModelOption,
)
from game import Game, LegalMove
from othello_engine import (
    Player,
    opponent,
)


BLACK = Game.BLACK
WHITE = Game.WHITE
BOARD_SIZE = 8
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


@dataclass(frozen=True)
class PositionSnapshot:
    board: tuple[tuple[int, ...], ...]
    current_color: int
    last_move: tuple[int, int] | None
    game_over: bool
    winner: int | None


class SpectatorMatch:
    """UI-independent match state used by the real-time viewer."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.game = Game()
        self.current_color = BLACK
        self.last_move: tuple[int, int] | None = None
        self.history: list[TurnRecord] = []
        self.game_over = False
        self.winner: int | None = None
        self.timeline: list[PositionSnapshot] = [self._snapshot()]
        self.timeline_index = 0

    @property
    def at_latest(self) -> bool:
        return self.timeline_index == len(self.timeline) - 1

    @property
    def visible_history(self) -> list[TurnRecord]:
        return self.history[: self.timeline_index]

    def _snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(
            board=tuple(tuple(row) for row in self.game.board),
            current_color=self.current_color,
            last_move=self.last_move,
            game_over=self.game_over,
            winner=self.winner,
        )

    def _restore(self, snapshot: PositionSnapshot) -> None:
        self.game = Game()
        self.game.board = [list(row) for row in snapshot.board]
        self.game.last = (
            None if snapshot.last_move is None else list(snapshot.last_move)
        )
        self.current_color = snapshot.current_color
        self.last_move = snapshot.last_move
        self.game_over = snapshot.game_over
        self.winner = snapshot.winner

    def seek(self, timeline_index: int) -> bool:
        destination = max(0, min(timeline_index, len(self.timeline) - 1))
        if destination == self.timeline_index:
            return False
        self.timeline_index = destination
        self._restore(self.timeline[destination])
        return True

    def seek_relative(self, amount: int) -> bool:
        return self.seek(self.timeline_index + amount)

    def _prepare_branch(self) -> None:
        if self.at_latest:
            return
        self.history = self.history[: self.timeline_index]
        self.timeline = self.timeline[: self.timeline_index + 1]

    def _append_snapshot(self) -> None:
        self.timeline.append(self._snapshot())
        self.timeline_index = len(self.timeline) - 1

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
        self._prepare_branch()
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
        self._append_snapshot()

    def pass_turn(self) -> None:
        if self.game_over:
            raise ValueError("The game is already over")
        self._prepare_branch()
        if self.game.legal_moves(self.current_color):
            raise ValueError(f"{COLOR_NAMES[self.current_color]} has a legal move")

        color = self.current_color
        self._record(color, None)
        next_color = opponent(color)
        if not self.game.legal_moves(next_color):
            self._finish()
        else:
            self.current_color = next_color
        self._append_snapshot()


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
        model_options: Sequence[ModelOption],
        black_spec: str,
        white_spec: str,
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
        self.player_specs = {BLACK: black_spec, WHITE: white_spec}
        self.model_options = list(model_options)
        self.picker_indices = {
            BLACK: self._model_index(black_spec),
            WHITE: self._model_index(white_spec),
        }
        self.picker_color = BLACK
        self.picker_open = False
        self.picker_message = ""
        self.match = SpectatorMatch()
        self.rng = random.Random(seed)
        self.delay = max(0.0, delay)
        self.previous_delay = self.delay or 0.75
        self.paused = start_paused
        self.step_requests = 0
        self.next_action_at = time.monotonic() + self.delay
        self.pending: PendingMove | None = None
        self.message = ""
        self.running = True

    def _model_index(self, spec: str) -> int:
        normalized = spec.strip().lower()
        for index, option in enumerate(self.model_options):
            if option.spec.lower() == normalized:
                return index
        return 0

    @staticmethod
    def _fit_text(font: Any, text: str, width: int) -> str:
        if font.size(text)[0] <= width:
            return text
        shortened = text
        while shortened and font.size(shortened + "...")[0] > width:
            shortened = shortened[:-1]
        return shortened + "..."

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
        if (
            self.pending is not None
            or self.match.game_over
            or not self.match.at_latest
            or self.picker_open
        ):
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
        if self.delay > 0:
            self.previous_delay = self.delay
        self.next_action_at = min(
            self.next_action_at,
            time.monotonic() + self.delay,
        )

    def _toggle_fastest(self) -> None:
        if self.delay == 0:
            self.delay = self.previous_delay
        else:
            self.previous_delay = self.delay
            self.delay = 0.0
        self.next_action_at = time.monotonic() + self.delay

    def _restart_match(self, message: str) -> None:
        self.match.reset()
        self.step_requests = 0
        self.next_action_at = time.monotonic() + self.delay
        self.message = message

    def _seek_history(self, destination: int) -> None:
        if self.pending is not None:
            self.message = "Wait for the current model calculation before reviewing."
            return
        if self.match.seek(destination):
            self.paused = True
            self.step_requests = 0
            self.message = (
                f"Reviewing turn {self.match.timeline_index}/"
                f"{len(self.match.timeline) - 1}."
            )

    def _open_model_picker(self) -> None:
        if self.pending is not None:
            self.message = "Wait for the current model calculation before changing models."
            return
        if not self.model_options:
            self.message = "No models are available."
            return
        self.paused = True
        self.step_requests = 0
        self.picker_indices = {
            BLACK: self._model_index(self.player_specs[BLACK]),
            WHITE: self._model_index(self.player_specs[WHITE]),
        }
        self.picker_color = BLACK
        self.picker_message = ""
        self.picker_open = True

    def _start_picker_match(self) -> None:
        selected = {
            color: self.model_options[self.picker_indices[color]]
            for color in (BLACK, WHITE)
        }
        self.picker_message = "Loading selected models..."
        self.draw()
        try:
            players = {
                color: build_player(selected[color].spec)
                for color in (BLACK, WHITE)
            }
        except (OSError, RuntimeError, ValueError) as exc:
            self.picker_message = f"Could not load model: {exc}"
            return

        self.players = players
        self.player_specs = {
            color: selected[color].spec for color in (BLACK, WHITE)
        }
        self.picker_open = False
        self.paused = True
        self._restart_match("New matchup loaded. Press Space to start or N to step.")

    def _handle_picker_event(self, event: Any) -> None:
        if event.type == pygame.MOUSEWHEEL:
            count = len(self.model_options)
            self.picker_indices[self.picker_color] = (
                self.picker_indices[self.picker_color] - event.y
            ) % count
            self.picker_message = ""
            return
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.picker_open = False
            self.picker_message = ""
        elif event.key in (pygame.K_TAB, pygame.K_LEFT, pygame.K_RIGHT):
            self.picker_color = opponent(self.picker_color)
            self.picker_message = ""
        elif event.key in (pygame.K_UP, pygame.K_DOWN):
            amount = -1 if event.key == pygame.K_UP else 1
            count = len(self.model_options)
            self.picker_indices[self.picker_color] = (
                self.picker_indices[self.picker_color] + amount
            ) % count
            self.picker_message = ""
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._start_picker_match()

    def _handle_event(self, event: Any) -> None:
        if event.type == pygame.QUIT:
            self.running = False
            return
        if self.picker_open:
            self._handle_picker_event(event)
            return
        if event.type == pygame.MOUSEWHEEL:
            amount = -event.y * (5 if pygame.key.get_mods() & pygame.KMOD_SHIFT else 1)
            self._seek_history(self.match.timeline_index + amount)
            return
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_SPACE:
            if not self.match.at_latest:
                self.match.seek(len(self.match.timeline) - 1)
                self.message = "Returned to the live position."
            self.paused = not self.paused
            self.step_requests = 0
            if not self.paused:
                self.next_action_at = time.monotonic()
        elif event.key == pygame.K_LEFT:
            self._seek_history(self.match.timeline_index - 1)
        elif event.key == pygame.K_RIGHT:
            if self.match.at_latest:
                self.paused = True
                self.step_requests += 1
            else:
                self._seek_history(self.match.timeline_index + 1)
        elif event.key == pygame.K_n:
            if not self.match.at_latest:
                self.match.seek(len(self.match.timeline) - 1)
            self.paused = True
            self.step_requests += 1
        elif event.key == pygame.K_HOME:
            self._seek_history(0)
        elif event.key == pygame.K_END:
            self._seek_history(len(self.match.timeline) - 1)
        elif event.key == pygame.K_PAGEUP:
            self._seek_history(self.match.timeline_index - 5)
        elif event.key == pygame.K_PAGEDOWN:
            self._seek_history(self.match.timeline_index + 5)
        elif event.key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self._change_speed(-0.10)
        elif event.key in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
            self._change_speed(0.10)
        elif event.key == pygame.K_f:
            self._toggle_fastest()
        elif event.key == pygame.K_r:
            if self.pending is None:
                self._restart_match("New game started with the same models.")
            else:
                self.message = "Wait for the current model calculation before restarting."
        elif event.key == pygame.K_m:
            self._open_model_picker()

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
        if not self.match.at_latest:
            return (
                f"Reviewing turn {self.match.timeline_index}/"
                f"{len(self.match.timeline) - 1}",
                self.GOLD,
            )
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
            if (
                color == self.match.current_color
                and self.match.at_latest
                and self.pending is None
                and isinstance(self.players[color], DQNPlayer)
            ):
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
        y += 23
        timeline_state = "LIVE" if self.match.at_latest else "REVIEW"
        timeline_text = (
            f"Turn: {self.match.timeline_index}/{len(self.match.timeline) - 1} "
            f"[{timeline_state}]"
        )
        self.screen.blit(self.small_font.render(timeline_text, True, self.MUTED), (x, y))
        y += 32

        self.screen.blit(self.heading_font.render("Recent turns", True, self.TEXT), (x, y))
        y += 28
        recent = self.match.visible_history[-5:]
        if not recent:
            self.screen.blit(self.body_font.render("No moves yet", True, self.MUTED), (x, y))
            y += 22
        else:
            for record in recent:
                line_color = (
                    self.GOLD
                    if record.turn == self.match.timeline_index
                    else self.TEXT
                )
                line = self.small_font.render(record.text, True, line_color)
                self.screen.blit(line, (x, y))
                y += 21

        controls_y = 510
        self.screen.blit(self.heading_font.render("Controls", True, self.TEXT), (x, controls_y))
        controls = (
            "Space       Pause / return live",
            "Left/Right  Review previous / next",
            "Home/End    First / live position",
            "Wheel/PgUp  Review 1 / 5 turns",
            "N           Play one live turn",
            "Up/Down     Faster / slower",
            "F           Toggle fastest speed",
            "R           New game, same models",
            "M           Choose models + new game",
            "Esc         Close",
        )
        for index, text in enumerate(controls):
            line = self.small_font.render(text, True, self.MUTED)
            self.screen.blit(line, (x, controls_y + 27 + index * 17))

        if self.message and not self.message.startswith("Model error"):
            message = self._fit_text(self.small_font, self.message, width)
            self.screen.blit(self.small_font.render(message, True, self.MUTED), (x, 700))

    def _draw_model_picker(self) -> None:
        shade = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 155))
        self.screen.blit(shade, (0, 0))

        dialog = pygame.Rect(145, 145, 790, 450)
        pygame.draw.rect(self.screen, self.PANEL, dialog, border_radius=14)
        pygame.draw.rect(self.screen, self.GRID, dialog, width=2, border_radius=14)
        title = self.title_font.render("Choose a new matchup", True, self.TEXT)
        self.screen.blit(title, (dialog.x + 30, dialog.y + 24))

        y = dialog.y + 90
        for color in (BLACK, WHITE):
            active = color == self.picker_color
            card = pygame.Rect(dialog.x + 30, y, dialog.width - 60, 105)
            pygame.draw.rect(
                self.screen,
                (255, 248, 224) if active else (248, 249, 250),
                card,
                border_radius=9,
            )
            pygame.draw.rect(
                self.screen,
                self.GOLD if active else (205, 210, 216),
                card,
                width=3 if active else 1,
                border_radius=9,
            )
            self.screen.blit(
                self.heading_font.render(
                    f"{COLOR_NAMES[color]} model",
                    True,
                    self.TEXT,
                ),
                (card.x + 18, card.y + 14),
            )
            option = self.model_options[self.picker_indices[color]]
            option_label = self._fit_text(
                self.body_font,
                option.label,
                card.width - 36,
            )
            self.screen.blit(
                self.body_font.render(option_label, True, self.MUTED),
                (card.x + 18, card.y + 55),
            )
            y += 120

        instructions = (
            "Tab or Left/Right: switch color    Up/Down or wheel: choose model",
            "Enter: load models and start paused    Esc: cancel",
        )
        for index, line in enumerate(instructions):
            self.screen.blit(
                self.small_font.render(line, True, self.MUTED),
                (dialog.x + 30, dialog.y + 345 + index * 22),
            )
        if self.picker_message:
            message = self._fit_text(
                self.small_font,
                self.picker_message,
                dialog.width - 60,
            )
            message_color = (
                self.RED
                if self.picker_message.startswith("Could not")
                else self.TEXT
            )
            self.screen.blit(
                self.small_font.render(message, True, message_color),
                (dialog.x + 30, dialog.y + 407),
            )

    def draw(self) -> None:
        self.screen.fill(self.BACKGROUND)
        self._draw_board()
        self._draw_panel()
        if self.picker_open:
            self._draw_model_picker()
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
            model_options=options,
            black_spec=black_spec,
            white_spec=white_spec,
        )
        app.run()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
