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
#from computer import Computer
#from computer2 import Computer2 as Computer
from computer2 import Computer3 as Computer #supervised bot; really bad

class Main:
    GREEN = (34,139,34)
    BLACK = (0,0,0)
    WHITE = (255,255,255)
    PINK =  "#FF8DA1"
    PALE_RED = "#FFDFE4"
    LIGHT_GREEN = (162,255,184)
    LIGHT_RED = (255,151,151)
    LIME = (50,205,50)
    
    def __init__(self,side=8,mode = "computer"):
        #mode is computer: pvcom
        #mode is player: pvp
        self.running = True

        self.width = 800
        self.height = 600
 
        self.font = pygame.font.SysFont("Comic Sans",20)
        self.bigFont = pygame.font.SysFont("Comic Sans",40)

        self.mode = mode
        self.computer = None
        self.side = side

        self.showLegal = False
        self.clickDict = {}

        self.reset()

        

    def reset(self):
        self.game = Game(self.side)
        self.activePlayerIndex = 0
        if self.mode == "computer":
            self.computer = Computer(self.game,Game.WHITE)
            print(f"Playing \"{Computer.PATH}\"")
        self.close_timeout = None
        


    def blit_turn(self,screen):
        text = self.font.render(("Black's" if self.activePlayerIndex+1 == Game.BLACK else "White's")+" Turn",True,Main.BLACK)

        screen.blit(text,(self.width-150,25))

    def draw_score(self,screen,x,y): #top left
        score = self.game.get_score()


        texts = ["Scores:",f"Black: {score[Game.BLACK]}",f"White: {score[Game.WHITE]}"]
        for i in range(len(texts)):#text in texts:
            surf = self.font.render(texts[i],True,Main.BLACK)
            screen.blit(surf,(x,y + i * 30))

    def draw_toggle_bar(self,screen,x,y): #center
        RADIUS = 15

        text = self.font.render("Show legal moves",True,Main.BLACK)
        text_rect = text.get_rect()
        text_rect.centerx = x
        text_rect.centery = y
        screen.blit(text,text_rect)

        if self.showLegal == False:
            color1 = Main.PALE_RED#PINK#WHITE
            color2 = Main.LIGHT_RED
            xMod = 0
        else:
            color1 = Main.LIGHT_GREEN
            color2 = Main.LIME
            xMod=40
            
        toggle = pygame.Rect(text_rect.width + text_rect.x + 15, y-RADIUS,60, RADIUS*2)
        pygame.draw.rect(screen, color1, toggle,  0, RADIUS*2)
        
        pygame.draw.circle(screen,color2,(text_rect.width + text_rect.x + 25 + xMod,y),RADIUS)

        out = pygame.Rect(text_rect.x,min(text_rect.y,toggle.y),(toggle.x-text_rect.x)+toggle.width,max(text_rect.bottom,toggle.bottom)-min(text_rect.y,toggle.y))

        #pygame.draw.rect(screen,Main.BLACK,out,width=1)

        return out

    def draw_legal(self,screen):
        TX,TY = Game.TOP_LEFT
        legal = self.game.get_all_legal_moves(self.activePlayerIndex+1)
        if not self.computer_active():
            for lx,ly in legal:
                color = (Game.C_WHITE if self.activePlayerIndex+1 == Game.WHITE else Game.C_BLACK)
                center = (TX + Game.SQUARE * (lx+0.5),TY + Game.SQUARE* (ly+0.5))
                pygame.draw.circle(screen,color,center,Game.RADIUS,width=1)


    def computer_active(self):
        return (self.mode == "computer" and self.activePlayerIndex+1 == self.computer.color)

    def draw(self,screen):
        self.clickDict = {}
        self.game.draw_board(screen)

        self.blit_turn(screen)

        self.draw_score(screen,self.width-180,80)

        if self.close_timeout is not None:
            text = self.bigFont.render("GAME OVER",True,Main.PINK)
            rect = text.get_rect()
            rect.center = (self.width/2,self.height/2)
            screen.blit(text,rect)

        self.clickDict["toggle"] = self.draw_toggle_bar(screen,self.width-180,self.height/2)

        if self.showLegal:
            self.draw_legal(screen)

        # Show DQN value prediction when computer is thinking or it's computer's turn
        if self.computer_active() and hasattr(self.computer, 'get_value_prediction'):
            try:
                value = self.computer.get_value_prediction()
                val_text = f"DQN Value: {value:+.3f}"
                val_color = Main.LIGHT_GREEN if value > 0 else (Main.LIGHT_RED if value < 0 else Main.BLACK)
                surf = self.font.render(val_text, True, val_color)
                screen.blit(surf, (self.width-180, 180))
            except Exception:
                pass

    def next_turn(self):
        self.activePlayerIndex = (self.activePlayerIndex+1)%2
        if len(self.game.get_all_legal_moves(self.activePlayerIndex+1))== 0: #forfeit turn
            self.activePlayerIndex = (self.activePlayerIndex+1)%2

        if len(self.game.get_all_legal_moves(self.activePlayerIndex+1)) == 0:
            #game over
            self.game.no_legal_moves = True

        if self.game.check_game_over():
            self.close_timeout = datetime.now()

        if self.computer is not None:
            self.computer.cooldown = datetime.now()


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
                        if sq is not None:
                            if self.activePlayerIndex+1 == Game.BLACK or self.mode!="computer":
                                x,y = sq
                                successful = self.game.place_piece(self.activePlayerIndex+1,x,y)
                                if successful:
                                    self.next_turn()

                        else: #look in clickDict
                            for key in self.clickDict:
                                if self.clickDict[key].collidepoint((mx,my)):
                                    #true
                                    if key == "toggle":
                                        self.showLegal = not self.showLegal
                                    break

                    

                            
            
            screen.fill(Main.WHITE)
            self.draw(screen)
            pygame.display.flip()

            if self.close_timeout is not None:
                if (datetime.now() - self.close_timeout).total_seconds()>=15:
                    self.reset()
                    #self.running = False
            elif self.computer_active():
                if (datetime.now()-self.computer.cooldown).total_seconds() > 1.5:
                    self.computer.pick()
                    self.next_turn()

        pygame.quit()

        
if __name__ == "__main__":
    GAME_MODE = "computer"
    #GAME_MODE = "player"
    
    m = Main(mode=GAME_MODE)
    m.main()



