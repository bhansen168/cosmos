"""
Adapters to use minimax_model.py and genetic_model.py with the legacy
computerRL.py training loop (which uses game.py's Game class).
"""

import random
from typing import Optional

from othello_engine import (
    BLACK,
    WHITE,
    EMPTY,
    HeadlessOthello,
    LegalMove,
    opponent,
)
from minimax_model import MinimaxPlayer
from genetic_model import GeneticPlayer, load_checkpoint


class GameToHeadlessAdapter:
    """Wrap a game.py Game instance to look like a HeadlessOthello."""
    
    def __init__(self, game):
        self._game = game
        self.board = [row[:] for row in game.board]
    
    def _sync_from_game(self):
        self.board = [row[:] for row in self._game.board]
    
    def _sync_to_game(self):
        for y in range(8):
            for x in range(8):
                self._game.board[y][x] = self.board[y][x]
    
    def legal_moves(self, color: int) -> list[LegalMove]:
        self._sync_from_game()
        legal = self._game.get_all_legal_moves(color, returnJump=True)
        moves = []
        for (x, y), jumped in legal:
            flips = tuple((jx, jy) for jx, jy in jumped)
            moves.append(LegalMove(x=x, y=y, flips=flips))
        return moves
    
    def play(self, color: int, move: LegalMove) -> None:
        self._sync_from_game()
        self._game.place_piece(color, move.x, move.y)
        self._sync_to_game()
    
    def undo(self, color: int, move: LegalMove) -> None:
        self._sync_from_game()
        other = opponent(color)
        self._game.board[move.y][move.x] = EMPTY
        for x, y in move.flips:
            self._game.board[y][x] = other
        self._sync_to_game()
    
    def score(self) -> dict[int, int]:
        return self._game.get_score()
    
    def clone(self) -> "GameToHeadlessAdapter":
        new_adapter = GameToHeadlessAdapter(self._game)
        new_adapter.board = [row[:] for row in self.board]
        return new_adapter


class HeadlessToGameAdapter:
    """Wrap a HeadlessOthello to look like a game.py Game for Computer interface."""
    
    def __init__(self, headless: HeadlessOthello):
        self._headless = headless
        self.side = 8
        self.board = [row[:] for row in headless.board]
        self.no_legal_moves = False
        self.last = None
    
    def get_all_legal_moves(self, color: int, returnJump: bool = False):
        moves = self._headless.legal_moves(color)
        if returnJump:
            return [((move.x, move.y), [(fx, fy) for fx, fy in move.flips]) for move in moves]
        return [(move.x, move.y) for move in moves]
    
    def place_piece(self, color: int, x: int, y: int) -> bool:
        legal = self._headless.legal_moves(color)
        legal_dict = {(move.x, move.y): move for move in legal}
        if (x, y) not in legal_dict:
            return False
        self._headless.play(color, legal_dict[(x, y)])
        self.board = [row[:] for row in self._headless.board]
        self.last = (x, y)
        return True
    
    def get_score(self) -> dict[int, int]:
        return self._headless.score()
    
    def check_game_over(self) -> bool:
        return not self._headless.legal_moves(BLACK) and not self._headless.legal_moves(WHITE)
    
    def valid(self, x: int, y: int) -> bool:
        return 0 <= x < 8 and 0 <= y < 8
    
    def _set_middle(self):
        pass


class MinimaxComputer:
    """Computer adapter using MinimaxPlayer for computerRL.py training loop."""
    
    def __init__(self, game, color: int, depth: int = 2):
        self.color = color
        self._adapter = GameToHeadlessAdapter(game)
        self._player = MinimaxPlayer(depth=depth)
        self._rng = random.Random()
    
    def pick_greedy(self, color: Optional[int] = None, place: bool = True):
        if color is None:
            color = self.color
        legal_moves = self._adapter.legal_moves(color)
        if not legal_moves:
            return None
        x, y = self._player.choose_move(self._adapter, color, legal_moves, self._rng)
        if place:
            self._adapter.play(color, next(m for m in legal_moves if m.x == x and m.y == y))
        return (x, y)
    
    def pick_random(self, color: Optional[int] = None, place: bool = True):
        if color is None:
            color = self.color
        legal_moves = self._adapter.legal_moves(color)
        if not legal_moves:
            return None
        move = random.choice(legal_moves)
        if place:
            self._adapter.play(color, move)
        return (move.x, move.y)
    
    def pick_minimax(self, color: Optional[int] = None, place: bool = True):
        return self.pick_greedy(color, place)
    
    def pick(self):
        return self.pick_greedy()


class GeneticComputer:
    """Computer adapter using GeneticPlayer for computerRL.py training loop."""
    
    def __init__(self, game, color: int, checkpoint_path: str):
        self.color = color
        self._adapter = GameToHeadlessAdapter(game)
        self._player = GeneticPlayer.from_checkpoint(checkpoint_path)
        self._rng = random.Random()
    
    def pick_greedy(self, color: Optional[int] = None, place: bool = True):
        if color is None:
            color = self.color
        legal_moves = self._adapter.legal_moves(color)
        if not legal_moves:
            return None
        x, y = self._player.choose_move(self._adapter, color, legal_moves, self._rng)
        if place:
            self._adapter.play(color, next(m for m in legal_moves if m.x == x and m.y == y))
        return (x, y)
    
    def pick_random(self, color: Optional[int] = None, place: bool = True):
        if color is None:
            color = self.color
        legal_moves = self._adapter.legal_moves(color)
        if not legal_moves:
            return None
        move = random.choice(legal_moves)
        if place:
            self._adapter.play(color, move)
        return (move.x, move.y)
    
    def pick_minimax(self, color: Optional[int] = None, place: bool = True):
        return self.pick_greedy(color, place)
    
    def pick(self):
        return self.pick_greedy()


def create_minimax_computer(game, color: int, depth: int = 2) -> MinimaxComputer:
    """Factory to create a MinimaxComputer for computerRL.py."""
    return MinimaxComputer(game, color, depth)


def create_genetic_computer(game, color: int, checkpoint_path: str) -> GeneticComputer:
    """Factory to create a GeneticComputer for computerRL.py."""
    return GeneticComputer(game, color, checkpoint_path)