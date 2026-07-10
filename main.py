"""
GUI for the game
"""

#generic pygame template

import os,warnings,sys
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
pygame.init()

from datetime import datetime,timedelta

sys.path.append(os.getcwd())
from game import Game
from computer import Computer

class Main:
    GREEN = (34,139,34)
    BLACK = (0,0,0)
    WHITE = (255,255,255)
    PINK =  "#FF8DA1"
    
    def __init__(self,side=8,mode = "computer"):
        #mode is computer: pvcom
        #mode is player: pvp
        self.running = True

        self.width = 800
        self.height = 600

        self.game = Game(side)
 
        self.activePlayerIndex = 0 #for active player index, 0 is black and 1 is white; add 1 to get real value

        self.font = pygame.font.SysFont("Comic Sans",20)
        self.bigFont = pygame.font.SysFont("Comic Sans",40)

        self.close_timeout = None
        self.mode = mode

        self.computer = None
        if mode == "computer":
            self.computer = Computer(self.game,Game.WHITE)


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

        if self.close_timeout is not None:
            text = self.bigFont.render("GAME OVER",True,Main.PINK)
            rect = text.get_rect()
            rect.center = (self.width/2,self.height/2)
            screen.blit(text,rect)

    def next_turn(self):
        self.activePlayerIndex = (self.activePlayerIndex+1)%2
        if len(self.game.get_all_legal_moves(self.activePlayerIndex+1))== 0: #forfeit turn
            self.activePlayerIndex = (self.activePlayerIndex+1)%2


    def main(self):
        screen = pygame.display.set_mode((self.width,self.height))

        icon_image = pygame.image.load('logo.png')  # Relative path to your 32x32 image
        pygame.display.set_icon(icon_image)

        pygame.display.set_caption("COSMOS - Othello")

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    pass
        
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.close_timeout is None:
                        mx,my = pygame.mouse.get_pos()
                        sq = self.game.get_square_clicked(mx,my)
                        if self.activePlayerIndex+1 == Game.BLACK or self.mode!="computer":
                            if sq is not None:
                                x,y = sq
                                successful = self.game.place_piece(self.activePlayerIndex+1,x,y)
                                if successful:
                                    if self.computer is not None:
                                        self.computer.cooldown = datetime.now()
                                        
                                    self.next_turn()
                                    if len(self.game.get_all_legal_moves(self.activePlayerIndex+1)) == 0:
                                        #game over
                                        self.game.no_legal_moves = True

                                    if self.game.check_game_over():
                                        self.close_timeout = datetime.now()

                            
            
            screen.fill(Main.WHITE)
            self.draw(screen)
            pygame.display.flip()

            if self.close_timeout is not None:
                if (datetime.now() - self.close_timeout).total_seconds()>=5:
                    self.running = False
            elif self.activePlayerIndex+1 == Game.WHITE and self.mode == "computer":
                if (datetime.now()-self.computer.cooldown).total_seconds() > 1.5:
                    self.computer.pick_greedy()
                    self.next_turn()

                    if self.game.check_game_over():
                        self.close_timeout = datetime.now()

        pygame.quit()

        score = self.game.get_score()

        print(f"Game over!\nFinal scores: ")
        for key in score:
            print(f"{('White' if key == Game.WHITE else 'Black')}: {score[key]}")
        
if __name__ == "__main__":
    GAME_MODE = "computer"
    #GAME_MODE = "player"
    
    m = Main(mode=GAME_MODE)
    m.main()



