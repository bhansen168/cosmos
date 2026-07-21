"""
GUI for the game
"""


import os,warnings,sys,threading,asyncio
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
pygame.init()

from datetime import datetime,timedelta
from copy import deepcopy

sys.path.append(os.getcwd())
from game import Game
from computer import ComputerDQN,ComputerSupervised as SupervisedComputer,ComputerGen as GeneticComputer,ComputerGen25 as GeneticComputer25,create_minimax_computer, Computer50D5 as GeneticComputer5, Computer25D5 as GeneticComputer25_5


class Main:
    GREEN = (34,139,34)
    BLACK = (0,0,0)
    WHITE = (255,255,255)
    PINK =  "#FF8DA1"
    PALE_RED = "#FFDFE4"
    LIGHT_GREEN = (162,255,184)
    LIGHT_RED = (255,151,151)
    LIME = (50,205,50)
    GRAY = (210,210,210)
    DARK_GREEN = (0,100,0)
    
    AI_MODES = {
        "dqn": ("Hamlet (DQN)", ComputerDQN),
        "genetic": ("Prospero (G50-2)", GeneticComputer),
        "genetic_25": ("Ariel (G25-2)", GeneticComputer25),
        "genetic_d5": ("Caliban (G50-5)", GeneticComputer5),
        "genetic_25_d5": ("Stephano (G25-5)", GeneticComputer25_5),
        "supervised": ("Horatio (SL)", SupervisedComputer),
        "minimax-2": ("Hotspur (MM-2)", lambda g, c: create_minimax_computer(g, c, depth=2)),
        "minimax-4": ("Henry V (MM-4)", lambda g, c: create_minimax_computer(g, c, depth=4)),
        "minimax-6": ("Octavius (MM-6)", lambda g, c: create_minimax_computer(g, c, depth=6)),
    }

    TESTING_ML = False
    FPS = 60
    
    def __init__(self,side=8,mode = "dqn",compColor = "W"):
        #mode is computer: pvcom
        #mode is player: pvp
        self.running = True

        self.width = 800
        self.height = 600
  
        self.subtitle = pygame.font.Font("williamshakespearewf.ttf",45)
        self.modeFont = pygame.font.Font("Augusta.ttf",35)
        #self.modeFont =pygame.font.SysFont("Comic Sans",20)
        #self.subtitle = pygame.font.Font("Shakespeare-First-Folio.ttf",15)
        self.bigFont = pygame.font.Font("Shakespeare-First-Folio.ttf",40)#pygame.font.SysFont("Comic Sans",40)
        self.font = pygame.font.SysFont("Comic Sans",20) #for gameplay
        
        self.mode = mode
        self.computer = None
        self.computer_name = ""
        self.compClass = None
        self.side = side

        self.compColor = (Game.WHITE if compColor == "W" else Game.BLACK)
        self.pickColor = (compColor not in ["W","B"])

        self.compLoc = None
        self.thread =  None

        self.showLegal = False
        self.printed = False
        self.clickDict = {}

        #feather for titles
        self.init_feathers()


        self.screen = "home"

        self.clock = pygame.time.Clock()

        self.switch_comp() #init necessary stuff
        self.reset()

    def init_feathers(self):
        self.featherSurfR = pygame.Surface((230,435),pygame.SRCALPHA)
        img = pygame.transform.scale(pygame.image.load("quill2b.png"),(226,435))
        img2 = pygame.transform.scale(pygame.image.load("quill2w.png"),(226,435))

        

        self.featherSurfR.blit(img,(0,0))
        self.featherSurfR.blit(pygame.transform.flip(img2,True,False),(0,0))
        self.featherRectR = self.featherSurfR.get_rect()
        self.featherRectR.center = (5*self.width/6,self.height/2)

        self.featherSurfL = pygame.transform.flip(self.featherSurfR,True,False)
        self.featherRectL = self.featherSurfL.get_rect()
        self.featherRectL.center = (self.width/6,self.height/2)

    def comp_pick(self):
        #save previous state
        oldBoard = deepcopy(self.game.board)
        lastBefore = (self.game.last.copy() if self.game.last is not None else None)

        #save pick
        self.computer.pick()
        self.compLoc = self.game.last.copy()

        #revert state
        self.game.last = lastBefore
        self.game.board = oldBoard
        

    def switch_comp(self):
        if self.mode in Main.AI_MODES: #NOT PVP
            ai_name, ai_class = Main.AI_MODES[self.mode]
            self.compClass = ai_class
            self.computer_name = ai_name

        else:
            self.compClass = None
            self.computer_name = "The Players (PvP)"

    def begin_game(self):
        if self.mode in Main.AI_MODES:
            self.computer = self.compClass(self.game,self.compColor)
        else:
            self.computer = None
        self.screen = "game"
        
            
    def reset(self):
        self.game = Game(self.side)#never make save=True because then saves empty list
        self.activePlayerIndex = 0

        '''
        if self.mode in Main.AI_MODES:
            ai_name, ai_class = Main.AI_MODES[self.mode]
            self.computer = ai_class(self.game, self.compColor)
            self.computer_name = ai_name
            print(f"Switched to {self.mode}: {self.computer.name if hasattr(self.computer, 'name') else self.computer.__class__.__name__}")
        else:
            self.computer = None
            print(f"Switched to {self.mode} mode (human vs human)")
        '''
        self.close_timeout = None
        self.screen = "home"
        self.computer = None
        


    def blit_turn(self,screen):
        text = self.font.render(("Black's" if self.activePlayerIndex+1 == Game.BLACK else "White's")+" Turn",True,Main.BLACK)

        screen.blit(text,(self.width-150,25))

    @staticmethod
    def _fit_text(font, text, width):
        if font.size(text)[0] <= width:
            return text
        shortened = text
        while shortened and font.size(shortened + "…")[0] > width:
            shortened = shortened[:-1]
        return shortened + "…"

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
        return (self.computer is not None and self.activePlayerIndex+1 == self.computer.color)

    def draw_ai_val(self,screen):
        if self.computer_active() and hasattr(self.computer, 'get_value_prediction') and Main.TESTING_ML:
            try:
                value = self.computer.get_value_prediction()
                val_text = f"{value:+.3f}"
                val_color = Main.LIGHT_GREEN if value > 0 else (Main.LIGHT_RED if value < 0 else Main.BLACK)
                surf = self.font.render(val_text, True, val_color)
                screen.blit(surf, (self.width-180, 180))
            except Exception as e:
                # Print error to console for debugging
                print(f"Value display error: {e}")

    def draw_title(self,screen):
        text = self.bigFont.render("Tempest",True,Main.WHITE)
        rect = text.get_rect()
        rect.center = (self.width/2,self.height/8)
        screen.blit(text,rect)

        subtitle = self.subtitle.render("Automated Othello Bot",True,Main.WHITE)
        rect = subtitle.get_rect()
        rect.center = (self.width/2,self.height/8 + 55)
        screen.blit(subtitle,rect)
        

        text2 = self.modeFont.render(f"Mode{('l' if self.computer_name.lower()!='The Players (PvP)' else '')}: "+str(self.computer_name),True,Main.WHITE)
        rect = text2.get_rect()
        rect.center = (self.width/2,self.height/4 + 45)
        screen.blit(text2,rect)

        if self.pickColor and "pvp" not in self.computer_name.lower():
            #draw boxes
            text3 = self.modeFont.render("Bot plays as:",True,Main.WHITE)
            rect = text3.get_rect()
            rect.centery = self.height/2-30
            rect.x = self.width/2-30-rect.width/2-5

            box  = pygame.Rect(rect.x + rect.width+ 10,self.height/2-55,50,50)
            #pygame.draw.rect(screen,(Main.WHITE if self.compColor == "W" else Main.BLACK),box)
            pygame.draw.circle(screen,(Main.WHITE if self.compColor == Game.WHITE else Main.BLACK),box.center,25)
            screen.blit(text3,rect)
            self.clickDict["color"] = box
            


        
        button = pygame.Rect(self.width/2 - 60, self.height*5/8 - 30,120,60)
        pygame.draw.rect(screen,Main.GRAY,button,border_radius = 5)
        self.clickDict["begin"] = button
        buttonText = self.subtitle.render("Begin Game",True,Main.BLACK)
        rect = buttonText.get_rect()
        rect.center = button.center
        screen.blit(buttonText,rect)

        screen.blit(self.featherSurfR,self.featherRectR)
        screen.blit(self.featherSurfL,self.featherRectL)
        
        """
        #Alternate style
        button = pygame.Rect(self.width/2 - 60, self.height/2 - 30,120,60)
        pygame.draw.rect(screen,Main.WHITE,button,width=2,border_radius = 5)
        self.clickDict["begin"] = button
        buttonText = self.subtitle.render("Begin Game",True,Main.WHITE)
        rect = buttonText.get_rect()
        rect.center = button.center
        screen.blit(buttonText,rect)
        """


    def draw(self,screen):
        if self.screen == "game":
            self.clickDict = {}
            self.game.draw_board(screen)

            self.blit_turn(screen)

            self.draw_score(screen,self.width-180,80)

            if self.close_timeout is not None:
                text = self.bigFont.render("GAME OVER",True,Main.PINK)
                rect = text.get_rect()
                rect.centerx = self.width/2
                rect.bottom = self.height - 10
                screen.blit(text,rect)

            self.clickDict["toggle"] = self.draw_toggle_bar(screen,self.width-180,self.height/2)

            if self.showLegal:
                self.draw_legal(screen)

            # Show AI value prediction when computer is thinking or it's computer's turn
            self.draw_ai_val(screen)
        else:
            screen.fill(Main.DARK_GREEN)
            self.draw_title(screen)
            

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


    async def main(self):
        screen = pygame.display.set_mode((self.width,self.height))

        icon_image = pygame.image.load('logo.png')  # Relative path to your 32x32 image
        pygame.display.set_icon(icon_image)

        pygame.display.set_caption("Tempest Othello Environment")

        while self.running:
            mode_switched = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if self.screen == "home":
                        if event.key == pygame.K_F1 or event.key == pygame.K_RIGHT:
                            # Cycle through AI modes
                            modes = list(Main.AI_MODES.keys()) + ["player"]
                            idx = modes.index(self.mode) if self.mode in modes else 0
                            self.mode = modes[(idx + 1) % len(modes)]
                            self.switch_comp()
                            #self.reset()
                            #mode_switched = True
                            #break
                        elif event.key == pygame.K_LEFT:
                            modes = list(Main.AI_MODES.keys()) + ["player"]
                            idx = modes.index(self.mode) if self.mode in modes else 0
                            self.mode = modes[(idx - 1) % len(modes)]
                            self.switch_comp()
                            #self.reset()
                            #mode_switched = True
                            #break
                            
                        
                        elif event.key == pygame.K_RETURN: #begin game
                            self.begin_game()
                        elif event.key == pygame.K_c:
                            self.compColor = (Game.BLACK if self.compColor == Game.WHITE else Game.WHITE)
                        

                
                    
                    # Allow other keydowns to pass through (though we don't handle them)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.screen == "game":
                        if self.close_timeout is None:
                            mx,my = pygame.mouse.get_pos()
                            sq = self.game.get_square_clicked(mx,my)
                            if sq is not None:
                                if self.computer is None or self.activePlayerIndex+1 != self.computer.color:
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
                    else:
                        mx,my = pygame.mouse.get_pos()
                        for key in self.clickDict:
                            if self.clickDict[key].collidepoint((mx,my)):
                                #true
                                if key == "begin":
                                    self.begin_game()
                                elif key == "color":
                                    self.compColor = (Game.BLACK if self.compColor == Game.WHITE else Game.WHITE)
            
            #if mode_switched:
            #    continue

                    

                            
            
            screen.fill(Main.WHITE)
            self.draw(screen)
            pygame.display.flip()

            if self.screen == "game":
                if self.close_timeout is not None:
                    if (datetime.now() - self.close_timeout).total_seconds()>=15:
                        self.reset()
                        #self.running = False
                elif self.computer_active():
                    if self.thread is None:
                        self.thread = threading.Thread(target = self.comp_pick)
                        self.thread.start()

                    if self.compLoc is not None:
                        if (datetime.now()-self.computer.cooldown).total_seconds() > 1.5:
                            #    self.comp_pick()
                            x,y = self.compLoc
                            self.game.place_piece(self.computer.color,x,y)
                            self.next_turn()
                            self.compLoc = None
                            self.thread = None

            self.clock.tick(Main.FPS)

            await asyncio.sleep(0) 

        pygame.quit()

        
if __name__ == "__main__":
    GAME_MODE = "genetic"  # Options: dqn, genetic, supervised, minimax, player

    AI_COLOR = ""#"B" #choices: "B","W",[anything else]
    """
    if GAME_MODE != "player" and AI_COLOR not in ["B","W"]:
        while True:
            try:
                col = int(input("Type 1 to play as Black, or 2 to play as White: "))
                if col == 1 or col == 2:
                    AI_COLOR = ("B" if col == 2 else "W")
                    break
            except ValueError:
                print("Please try again!")
    """
    m = Main(mode=GAME_MODE,compColor = AI_COLOR)
    asyncio.run(m.main())


