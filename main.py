"""
GUI for the game
"""

#generic pygame template

import os,warnings,sys
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
pygame.init()

sys.path.append(os.getcwd())
from game import Game

class Main:
    GREEN = (34,139,34)
    BLACK = (0,0,0)
    WHITE = (255,255,255)
    
    def __init__(self,side=8):
        self.running = True

        self.width = 1200
        self.height = 800

        self.game = Game(side)

    def draw(self,screen):
        self.game.draw_board(screen)

    def main(self):
        screen = pygame.display.set_mode((self.width,self.height))

        pygame.display.set_caption("Generic Pygame Window")

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    #if event.key == pygame.K_SPACE:
                    pass
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx,my = pygame.mouse.get_pos()
            
            screen.fill(Main.GREEN)
            self.draw(screen)
            pygame.display.flip()

        pygame.quit()
        
if __name__ == "__main__":
    m = Main()
    m.main()



