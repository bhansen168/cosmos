import os,warnings,sys
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

class Game:
    EMPTY = 0
    WHITE = 1
    BLACK = 2
    
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
        pass
        


        
        
        
