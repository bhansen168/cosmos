"""
GUI for the game
"""
import sys,os
sys.path.append(os.getcwd())
from game import Game

class Main:
    def __init__(self,side=8):
        self.game = Game(side)
