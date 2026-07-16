from __future__ import annotations

import os,warnings,sys
from dataclasses import dataclass

warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
pygame = None


@dataclass(frozen=True)
class LegalMove:
    """Move representation shared by the original and headless interfaces."""
    x: int
    y: int
    flips: tuple[tuple[int, int], ...]

class Game:
    EMPTY = 0
    WHITE = 2 #piece
    BLACK = 1
    
    C_GREEN = (34,139,34)
    C_BLACK = (0,0,0)
    C_WHITE = (255,255,255)
    C_LIGREEN = "#5CED73"

    SQUARE = 60
    RADIUS = int(SQUARE * 0.3)

    TOP_LEFT = (20,20)

    DIRS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
    
    def __init__(self,side=8): #side is number of squares per side
        self.side = side
        self.board = [[0 for _ in range(side)] for _ in range(side)]
        #set middle squares

        self.no_legal_moves = False
        self.last = None
        self._move_history = []

        self._set_middle()

    def _set_middle(self):
        self.board[3][3] = Game.WHITE
        self.board[4][4] = Game.WHITE
        self.board[3][4] = Game.BLACK
        self.board[4][3] = Game.BLACK

    def get_square_clicked(self,mx,my):
        TX,TY = Game.TOP_LEFT
        outX =(mx-TX)//Game.SQUARE
        outY =(my-TY)//Game.SQUARE

        if self.valid(outX,outY):
            return (outX,outY)
        return None

    def check_game_over(self):
        found = {Game.EMPTY:False,Game.WHITE:False,Game.BLACK:False}
        for y in range(self.side):
            for x in range(self.side):
                found[self.board[y][x]] = True
        return (False in list(found.values()) or self.no_legal_moves)

    def draw_board(self,screen):
        global pygame
        if pygame is None:
            try:
                import pygame as pygame_module
            except ImportError as exc:
                raise RuntimeError("Drawing the board requires Pygame") from exc
            pygame = pygame_module
        TX,TY = Game.TOP_LEFT

        pygame.draw.rect(screen,Game.C_GREEN,pygame.Rect(-12,-12,Game.SQUARE * self.side + TX * 2 + 12, Game.SQUARE * self.side + TY * 2 + 12),border_radius = 12)

        if self.last is not None:
            xl,yl = self.last
            pygame.draw.rect(screen,Game.C_LIGREEN,pygame.Rect(TX+Game.SQUARE*xl,TY+Game.SQUARE*yl,Game.SQUARE,Game.SQUARE))

            
        for yb in range(self.side+1):
            pygame.draw.line(screen,Game.C_BLACK,(TX,TY + yb * Game.SQUARE),(TX+Game.SQUARE * self.side,TY + yb * Game.SQUARE),width=2)
            
        for xb in range(self.side+1):
            pygame.draw.line(screen,Game.C_BLACK,(TX+xb * Game.SQUARE,TY),(TX+xb * Game.SQUARE,TY + self.side * Game.SQUARE),width=2)
        
        for y in range(self.side):
            for x in range(self.side):
                if self.board[y][x]!=Game.EMPTY:
                    pygame.draw.circle(screen,(Game.C_BLACK if self.board[y][x] == Game.BLACK else Game.C_WHITE),(TX + Game.SQUARE * (x+0.5),TY + Game.SQUARE* (y+0.5)),Game.RADIUS)

    def valid(self,x,y):
        return 0 <= y < self.side and 0 <= x < self.side #is on board

    def is_oppo(self,x,y,color):
        return self.board[y][x] != Game.EMPTY and self.board[y][x] != color #is opponent's

    def is_yours(self,x,y,color):
        return self.board[y][x] == color

    def get_jumped(self,color,x,y):
        flip = []
        #returns the pieces to flip
        for dx,dy in Game.DIRS:
            if  self.valid(x+dx,y+dy):
                sq = self.board[y+dy][x+dx]
                if self.is_oppo(x+dx,y+dy,color):
                    visited = []
                    x1 = x+dx
                    y1 = y+dy
                    #iterate until is yours again or is off board
                    while self.valid(x1,y1) and self.is_oppo(x1,y1,color):
                        visited.append((x1,y1))
                        x1 += dx
                        y1 += dy

                    if self.valid(x1,y1) and self.is_yours(x1,y1,color): #not out of bounds
                        flip.extend(visited)
        return flip

    def flip_piece(self,x,y):
        if self.board[y][x] == Game.BLACK:
            self.board[y][x] = Game.WHITE
        elif self.board[y][x] == Game.WHITE:
            self.board[y][x] = Game.BLACK

    def get_score(self):
        out = {Game.BLACK:0,Game.WHITE:0}
        for y in range(self.side):
            for x in range(self.side):
                if self.board[y][x]!= Game.EMPTY:
                    out[self.board[y][x]]+=1
        return out
                        
                    
        
    def is_legal_move(self,color,x,y,returnJump = False):
        jumped = self.get_jumped(color,x,y)
        if returnJump:
            return [self.board[y][x] == Game.EMPTY and len(jumped)>0,jumped]
        return self.board[y][x] == Game.EMPTY and len(jumped)>0

    def get_all_legal_moves(self,color,returnJump = False):
        moves = self.legal_moves(color)
        if returnJump:
            return [[(move.x,move.y),list(move.flips)] for move in moves]
        return [(move.x,move.y) for move in moves]

    def legal_moves(self,color):
        """Return legal moves using the interface shared by newer models."""
        board = self.board
        other = Game.WHITE if color == Game.BLACK else Game.BLACK
        moves = []
        for y in range(self.side):
            for x in range(self.side):
                if board[y][x] != Game.EMPTY:
                    continue
                flips = []
                for dx,dy in Game.DIRS:
                    scanX = x+dx
                    scanY = y+dy
                    if not (
                        0 <= scanX < self.side
                        and 0 <= scanY < self.side
                        and board[scanY][scanX] == other
                    ):
                        continue

                    lineStart = len(flips)
                    while (
                        0 <= scanX < self.side
                        and 0 <= scanY < self.side
                        and board[scanY][scanX] == other
                    ):
                        flips.append((scanX,scanY))
                        scanX += dx
                        scanY += dy

                    if not (
                        0 <= scanX < self.side
                        and 0 <= scanY < self.side
                        and board[scanY][scanX] == color
                    ):
                        del flips[lineStart:]
                if flips:
                    moves.append(LegalMove(x=x,y=y,flips=tuple(flips)))
        return moves

    def place_piece(self,color,x,y):
        legal,toChange = self.is_legal_move(color,x,y,returnJump = True)
        if legal:
            self.board[y][x] = color
            for x1,y1 in toChange:
                self.flip_piece(x1,y1)

            self.last = [x,y]
            return True
        return False

    def play(self,color,move):
        """Apply a prevalidated LegalMove for search and headless games."""
        if self.board[move.y][move.x] != Game.EMPTY or not move.flips:
            raise ValueError(f"Illegal move ({move.x}, {move.y})")

        previousLast = None if self.last is None else list(self.last)
        self._move_history.append((color,move,previousLast))
        self.board[move.y][move.x] = color
        for x,y in move.flips:
            self.board[y][x] = color
        self.last = [move.x,move.y]

    def undo(self,color,move):
        """Undo a move previously applied with play()."""
        previousLast = None
        if self._move_history:
            savedColor,savedMove,previousLast = self._move_history.pop()
            if savedColor != color or savedMove != move:
                raise ValueError("Moves must be undone in reverse order")

        self.board[move.y][move.x] = Game.EMPTY
        other = Game.WHITE if color == Game.BLACK else Game.BLACK
        for x,y in move.flips:
            self.board[y][x] = other
        self.last = previousLast

    def score(self):
        return self.get_score()

    def clone(self):
        """Return an independent copy for model search or spectator threads."""
        copied = Game(self.side)
        copied.board = [row[:] for row in self.board]
        copied.no_legal_moves = self.no_legal_moves
        copied.last = None if self.last is None else list(self.last)
        return copied
            

        
        
        
