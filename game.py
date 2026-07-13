import os,warnings,sys
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

class Game:
    EMPTY = 0
    WHITE = 2 #piece
    BLACK = 1
    
    C_GREEN = (34,139,34)
    C_BLACK = (0,0,0)
    C_WHITE = (255,255,255)
    C_LIGREEN = "#5CED73"

    SQUARE = 60

    TOP_LEFT = (20,20)

    DIRS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
    
    def __init__(self,side): #side is number of squares per side
        self.side = side
        self.board = [[0 for _ in range(side)] for _ in range(side)]
        #set middle squares

        self.no_legal_moves = False
        self.last = None

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
                    pygame.draw.circle(screen,(Game.C_BLACK if self.board[y][x] == Game.BLACK else Game.C_WHITE),(TX + Game.SQUARE * (x+0.5),TY + Game.SQUARE* (y+0.5)),int(Game.SQUARE * 0.3))

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
        out = []
        for y in range(self.side):
            for x in range(self.side):
                if self.board[y][x] == Game.EMPTY:
                    if returnJump:
                        legal,jumped = self.is_legal_move(color,x,y,returnJump = True)
                        if legal:
                            out.append([(x,y),jumped])
                    else:
                        if self.is_legal_move(color,x,y):
                            out.append((x,y))

        return out

    def place_piece(self,color,x,y):
        legal,toChange = self.is_legal_move(color,x,y,returnJump = True)
        if legal:
            self.board[y][x] = color
            for x1,y1 in toChange:
                self.flip_piece(x1,y1)

            self.last = [x,y]
            return True
        return False
            

        
        
        
