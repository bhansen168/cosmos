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
 
        self.activePlayerIndex = 0 #for active player index, 0 is black and 1 is white; add 1 to get real value

        self.font = pygame.font.SysFont("Comic Sans",20)


    def blit_turn(self,screen):
        text = self.font.render(("Black's" if self.activePlayerIndex+1 == Game.BLACK else "White's")+" Turn",True,Main.BLACK)

        screen.blit(text,(self.width-150,25))

    def draw_score(self,screen,x,y): #top left
        score = self.game.get_score()


        texts = ["Scores:",f"Black: {score[Game.BLACK]}",f"White: {score[Game.WHITE]}"]
        for i in range(len(texts)):#text in texts:
            surf = self.font.render(texts[i],True,Main.BLACK)
            screen.blit(surf,(x,y + i * 30))
        

    def draw(self,screen):
        self.game.draw_board(screen)

        self.blit_turn(screen)

        self.draw_score(screen,self.width-150,80)

        

    def main(self):
        screen = pygame.display.set_mode((self.width,self.height))

        pygame.display.set_caption("COSMOS - Othello")

        while not self.game.check_game_over():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    #if event.key == pygame.K_SPACE:
                    pass
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx,my = pygame.mouse.get_pos()
                    sq = self.game.get_square_clicked(mx,my)
                    if sq is not None:
                        x,y = sq
                        successful = self.game.place_piece(self.activePlayerIndex+1,x,y)
                        if successful:
                            self.activePlayerIndex = (self.activePlayerIndex+1)%2
            
            screen.fill(Main.WHITE)
            self.draw(screen)
            pygame.display.flip()

        pygame.quit()

        score = self.game.get_score()

        print(f"Game over!\nFinal scores: Black: {score[Game.BLACK]}",f"White: {score[Game.WHITE]}")
        
if __name__ == "__main__":
    m = Main()
    m.main()



