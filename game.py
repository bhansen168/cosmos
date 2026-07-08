import os,warnings,sys
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

class Game:
    EMPTY = 0
    WHITE = 1 #piece
    BLACK = 2

    C_BLACK = (0,0,0)
    C_WHITE = (255,255,255)

    SQUARE = 60

    TOP_LEFT = (20,20)
    
    def __init__(self,side): #side is number of squares per side
        self.side = side
        self.board = [[0 for _ in range(side)] for _ in range(side)]
        #set middle squares

        self._set_middle()

    def _set_middle(self):
        self.board[3][3] = Game.WHITE
        self.board[4][4] = Game.WHITE
        self.board[3][4] = Game.BLACK
        self.board[4][3] = Game.BLACK

    def draw_board(self,screen):
        TX,TY = Game.TOP_LEFT
        
        for yb in range(self.side+1):
            pygame.draw.line(screen,Game.C_BLACK,(TX,TY + yb * Game.SQUARE),(TX+Game.SQUARE * self.side,TY + yb * Game.SQUARE),width=2)
            
        for xb in range(self.side+1):
            pygame.draw.line(screen,Game.C_BLACK,(TX+xb * Game.SQUARE,TY),(TX+xb * Game.SQUARE,TY + self.side * Game.SQUARE),width=2)

        
        for y in range(self.side):
            for x in range(self.side):
                if self.board[y][x]!=Game.EMPTY:
                    pygame.draw.circle(screen,(Game.C_BLACK if self.board[y][x] == Game.BLACK else Game.C_WHITE),(TX + Game.SQUARE * (x+0.5),TY + Game.SQUARE* (y+0.5)),int(Game.SQUARE * 0.3))
        
        


        
        
        
