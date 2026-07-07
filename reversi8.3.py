import random,sys,os,math,time,copy,subprocess,warnings
warnings.filterwarnings("ignore")
sys.path.extend(["D:/importPython","C:/Users/synco/AppData/Roaming/Python/Python310/site-packages"])

def read(fileName):
    file=open(fileName,"r")
    output=file.readlines()
    file.close()
    return output

def write(fileName,string,printStuff=True,mode="w",newLine=True):
    path= os.path.abspath(fileName)
    path=path.replace("\\","/")
    with open(path,mode) as inFile:
        if newLine==True:
            inFile.write(str(string)+"\n")
        else:
            inFile.write(str(string))

def integer(prompt,maxError="",minError="",errorMessage="Invalid course of action--Please try again.",maxAns=1000000000000,minAns=0,errors=[ValueError,KeyboardInterrupt],enter=False):
    if ValueError not in errors:
        errors.append(ValueError)
    while True:
        try:
            x=int(str(input(str(prompt))).replace(",",""))
            if x>maxAns:
                if len(maxError)==0:
                    print("Too high! The maximum value allowed as your answer is "+str(maxAns)+". Please try again.")
                else:
                    print(maxError)
            elif x<minAns:
                if len(minError)==0:
                    print ("Too low! The minimum value allowed as your answer is "+str(minAns)+".Please try again.")
                else:
                    print(minError)
            else:
                break
        except tuple(errors):
            print (errorMessage)

    if enter==True:
        print()
    return x

def input_float_with_comma_removal(prompt,errorMessage="Invalid course of action--Please try again.",maxAns=1000000000000,minAns=0,errors=[ValueError,KeyboardInterrupt],enter=False):
    if ValueError not in errors:
        errors.append(ValueError)
    while True:
        try:
            x=float(str(input(str(prompt))).replace(",",""))
            if x>maxAns:
                print("Too high! The maximum value allowed as your answer is "+str(maxAns)+". Please try again.")
            elif x<minAns:
                print ("Too low! The minimum value allowed as your answer is "+str(minAns)+".Please try again.")
            else:
                break
        except tuple(errors):
            print (errorMessage)

    if enter==True:
        print()
    return x

def string(prompt,errorMessage="Invalid course of action--Please try again.",errors=[ValueError,KeyboardInterrupt],enter=False):
    if ValueError not in errors:
        errors.append(ValueError)
    while True:
        try:
            x=str(input(str(prompt)))
            break
        except tuple(errors):
            print (errorMessage)

    if enter==True:
        print()
    return x


def is_module_installed(module_name):
    try:
        # Check if the module is already installed
        subprocess.check_output([sys.executable, '-m', 'pip', 'show', module_name])
        return True
    except subprocess.CalledProcessError:
        return False

def install_module(module_name,printWorks=False):
    if is_module_installed(module_name) and printWorks==True:
        print(f"{module_name} is already installed.")
    else:
        try:
            # Run the pip install command
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', module_name])
            if printWorks==True:
                print(f"{module_name} installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {module_name}: {e}")

moduleList=["pygame","tabulate"]
for module in moduleList:
    install_module(module)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
#saveFile="D:/reversi/unlocked.txt"
import pygame
from tabulate import tabulate



def check_if_file_exists(fileName):
    fileExists=True
    try:
        test=open(fileName,"r")
    except FileNotFoundError:
        fileExists=False

    if fileExists==True:
        test.close()
    return fileExists


def initiation(lockedList,locks=True):
    names=["EASY","MEDIUM","HARD","TALENTED","EXPERT","GENIUS","DEMIGOD-LIKE","GOD-LIKE"]
    if locks==True:
        sizeOptions=[]
        for x in range(4,27,2):
            sizeOptions.append(x)
        firstOne=sizeOptions.pop(0)
        difficulties=""
        count=0
        maxAns=-1
        for x in range(0,len(names)):
            status=lockedList[x]
            name=names[x]
            string="\n\t"+str(x+1)+" --> "+str(status)+str(name.upper())
            if count==1 and "🔒" in status:
                break
            elif "🔒" in status:
                count=1
            maxAns=maxAns+1
            
            difficulties=difficulties+string
        
        maxAns=maxAns+1
        #print(maxAns)
            
    else:
        sizeOptions=[]
        for x in range(4,27,2):
            sizeOptions.append(x)
        firstOne=sizeOptions.pop(0)
        difficulties=""
        count=0
        maxAns=len(names)
        for x in range(0,len(names)):
            name=names[x]
            string="\n\t"+str(x+1)+" --> "+str(name.upper())
            
            difficulties=difficulties+string
    #help(integer)
    for z in range(maxAns,len(names)):
        string2="\n\t"+str(z+1)+" --> "+str(lockedList[z])
    levelError="Sorry, that level is either locked or nonexistent. Please try again.\n\n"
    #difficulties="\n\t1 --> EASY\n\t2 --> MEDIUM\n\t3 --> HARD\n\t4 --> TALENTED\n\t5 --> EXPERT\n\t6 --> GENIUS\n\t7 --> DEMIGOD\n\t8 --> GOD"
    level=integer("Please enter your desired opponent difficulty level:"+difficulties+"\nDesired computer level: ",maxAns=maxAns,minAns=1,maxError=levelError,minError=levelError)
    string1="\n\nType 1 for a board side length of "+str(firstOne)+" ("+str(firstOne**2)+" squares) , \n\t"
    lastOne=sizeOptions.pop(-1)
    for y in range(0,len(sizeOptions)):
        string1=string1+str(y+2)+" for a board side length of "+str(sizeOptions[y])+" ("+str(sizeOptions[y]**2)+" squares), \n\t"
    string1=string1+"or type "+str(len(sizeOptions)+2)+" for a board size length of "+str(lastOne)+" ("+str(lastOne**2)+" squares). \nYour choice: "
    maxAns=len(sizeOptions)+2
    sizes=[firstOne]+sizeOptions+[lastOne]
    size=integer(string1,maxAns=len(sizeOptions)+2,minAns=1)
    side=sizes[size-1]

    #side=24
    boardSize=(side,side)#width, height
    return (level,boardSize)

class Save_Game:
    def __init__(self):
         self.saveFile="unlocked.txt"
         self.totalLevels=8
         f=check_if_file_exists(self.saveFile)
         if f==False:
             write(self.saveFile,"",printStuff=False,mode="w",newLine=False)
    def get_saves(self):
        lines=read(self.saveFile)
        saves={}
        for line in lines:
            line=line.strip()
            lineList=line.split()
            saveName,levelAchieved=lineList
            saveName=saveName.replace("="," ")
            levelAchieved=int(levelAchieved)
            saves[saveName]=levelAchieved
        return saves
    def find_and_import_save(self):
        saves=self.get_saves()
        saveNames=list(saves.keys())
        fullString="Saves you can import: "
        for x in range(0,len(saveNames)):
            saveName=saveNames[x]
            string="\n\t"+str(x+1)+") "+str(saveName)
            fullString=fullString+string
        print(fullString)
        choice=integer("Please select a save to import, or type 0 to cancel: ",minAns=0,maxAns=len(saveNames))
        if choice==0:
            print("Import cancelled.")
            return []
        else:
            lockedList=[]
            saveName=saveNames[choice-1]
            level=saves[saveName]
            for x in range(0,self.totalLevels):
                if x<level:
                
                    lockedList.append("")
                else:
                    lockedList.append("🔒")
                
                    
            print("Save successfully imported!")     
            return lockedList
        
    
        
    def find_last_index_in_list(self,item,list1):
        last_index = len(list1) - list1[::-1].index(item) - 1
        return last_index
    def get_max_level_available(self,unlocked):
        index=self.find_last_index_in_list("",unlocked)
        return index+1
        
        #presuming it's ordered left-->right means lowest-->highest level.
    def export_a_save(self,unlocked):
        level=self.get_max_level_available(unlocked)
        saveName=string("Please enter a name for your save to be saved under: ")
        write(self.saveFile,str(saveName.replace(" ","="))+" "+str(level),printStuff=False,mode="a",newLine=True)
        print("Save complete.")
class GUI_Interface:
    def __init__(self,boardW,boardH,playerNum,windowW=800,windowH=800,margin=30):
        pygame.init()
        self.ROWS, self.COLS = boardW,boardH
        self.SQUARE_SIZE = windowW // self.COLS

        self.GREEN = (34,139,34)
        self.BLACK = (0,0,0)
        self.WHITE = (255,255,255)
        
        self.RED = (255,0,0)#NOT USING CURRENTLY
        self.ORANGE = (255,110,0)
        self.BLUE = (0,0,255)
        self.SILVER= (85, 85, 85)#(179, 169, 173)
        self.CYAN=(0, 255, 255)
        self.MAGENTA=(255, 0, 255)
        self.PINK=(199, 21, 133)
        self.LIME=(0, 255, 0)
        self.INDIGO=(75, 0, 130)
        self.TURQUOISE=(64, 224, 208)
        self.LAVENDER=(230, 230, 250)
        self.GOLD=(255, 215, 0)
        
        self.PURPLE = (128,0,128)
        self.TEAL=(0,128,128)
        
        #self.BOTTOMMARGIN=margin*2
        #margin=int(round(self.SQUARE_SIZE/,0))
        self.SCOREBOARD_WIDTH = 200
        self.WIDTH = windowW+(2*margin)+self.SCOREBOARD_WIDTH
        self.HEIGHT = windowH+int(round((3.5*margin),0))
        self.MARGIN = margin
        self.LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.BUTTONWIDTH = self.SQUARE_SIZE*2
        self.BUTTONHEIGHT = self.SQUARE_SIZE//3 * 2
        self.BUTTONCOLOR = self.WHITE
        self.BUTTONTEXTCOLOR = self.BLACK
        self.BOTTOMMARGIN=int(round(margin*2.5,0))
        if playerNum==2:
            self.textColors=[self.BLACK,self.WHITE]
        else:
            self.textColors=[self.ORANGE,self.BLUE,self.PURPLE,self.SILVER,self.GOLD,self.PINK,self.LIME,self.CYAN,self.MAGENTA,self.INDIGO,
                             self.TURQUOISE,self.LAVENDER]
        
        
        #we want an entirely green board
    def import_player_list(self,playerList):
        self.playerList=playerList
        self.playerNum=len(playerList)
    def draw_checker(self,screen, row, col,color):
        SQUARE_SIZE = self.SQUARE_SIZE
        # Calculate the center of the specified square
        center_x = self.MARGIN + col * SQUARE_SIZE + SQUARE_SIZE // 2
        center_y = self.MARGIN + row * SQUARE_SIZE + SQUARE_SIZE // 2
        colors=self.textColors
        c=colors[color-1]
        pygame.draw.circle(screen, c, (center_x, center_y), SQUARE_SIZE // 3)
    def draw_button(self,screen, text, x, y):
        height=self.BUTTONHEIGHT
        width=self.BUTTONWIDTH
        # Draw button rectangle
        pygame.draw.rect(screen,self.BUTTONCOLOR, (x, y, width, height))
        # Draw button text
         
        text_surface =  font.render(text, True, self.BUTTONTEXTCOLOR)
        text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
        screen.blit(text_surface, text_rect)
    
    def draw_board(self,screen):
        LETTERS=self.LETTERS
        pygame.init()
        font = pygame.font.SysFont(None, 24)
        SQUARE_SIZE = self.SQUARE_SIZE
        
        margin = self.MARGIN 
        color=self.GREEN
        for row in range(self.ROWS):
            for col in range(self.COLS):
                pygame.draw.rect(screen, self.BLACK, (margin + col * SQUARE_SIZE, margin + row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 1)  # Outline each square in black
                pygame.draw.rect(screen, color, (margin + col * SQUARE_SIZE + 1, margin + row * SQUARE_SIZE + 1, SQUARE_SIZE - 1, SQUARE_SIZE - 1))  # Fill the square with the appropriate color       
        for i in range(self.ROWS):
             
            text =  font.render(self.LETTERS[i], True, self.WHITE)
            screen.blit(text, (5, self.MARGIN + i * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 - text.get_height() // 2))

        # Label columns with letters
        for i in range(self.COLS):
             
            #num=self.ROWS - i
            num=i+1
            text =  font.render(str(num), True, self.WHITE)
            screen.blit(text, (self.MARGIN + i * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 - text.get_width() // 2, 5))

    def draw_board_from_board_list(self,screen,board):
        self.draw_board(screen)
        for x in range(0,len(board)):
            list1=board[x]
            ##finish by placing all checkers on board
            for y in range(0,len(list1)):
                item=list1[y]
                if item>0:
                    self.draw_checker(screen,x,y,item)
            
    def draw_input_prompt(self,screen, prompt):
        pygame.init()
        # Draw input prompt
        font = pygame.font.SysFont(None, 24)
        text =  font.render(prompt, True,self.WHITE)
        MARGIN=self.MARGIN
        screen.blit(text, (MARGIN, self.HEIGHT - self.BOTTOMMARGIN + 10))
        pygame.display.flip()
    
    def blit_input(self,screen,inputText,textColor):

        pygame.init()
        font = pygame.font.SysFont(None, 24)
        pygame.draw.rect(screen, self.BLACK, (self.MARGIN + 150, self.HEIGHT - self.MARGIN, 100, 30))
        text_surface = font.render(inputText, True, textColor)
        screen.blit(text_surface, (self.MARGIN + 150 + 5, self.HEIGHT - self.MARGIN + 5))

        pygame.display.flip()

    def draw_scoreboard(self,screen,scoreboard_data):
        # Draw the scoreboard area using dictionary data
        pygame.init()
        font=pygame.font.SysFont(None, 24)
        pygame.draw.rect(screen, self.BLACK, (self.SQUARE_SIZE * self.COLS + self.MARGIN, 0, self.SCOREBOARD_WIDTH, self.HEIGHT))
        # Render and display scoreboard text
        count=-2
        for i, (label, value) in enumerate(scoreboard_data.items()):
            coLor=self.WHITE
            text_surface = font.render(f"{label}: {value}", True, coLor)
            text_rect = text_surface.get_rect(x=self.COLS*self.SQUARE_SIZE + self.MARGIN + 20, y=50 + i * 40)
            screen.blit(text_surface, text_rect)
            
            count=count+1

    def main(self,prompt,player,board,playerList,playerColor):
        pygame.init()
        font = pygame.font.SysFont(None, 24)
        g=Game((self.ROWS,self.COLS))
        g.import_board(board)
        g.import_playerList(playerList)
        
        b=Board((self.ROWS,self.COLS))
        b.import_board(board)
        playerScores=b.get_score()
        playerList = ["Empty"] + playerList
        inputText=""
        MARGIN=self.MARGIN
        HEIGHT=self.HEIGHT
        WIDTH=self.WIDTH
        BLACK=self.BLACK
        WHITE=self.WHITE
        pString=str(player)+" ("+str(playerColor)+")"
        scoreboard_data = {"Scoreboard":""}
        for x in range(0,len(playerList)):
            player1=playerList[x]
            scoreboard_data[player1]=playerScores[x]
                          
            
            
        scoreboard_data["Turn"]=pString
        
        xPos= self.ROWS * self.SQUARE_SIZE // 2
        yPos=self.COLS * self.SQUARE_SIZE + int(round(self.MARGIN*1.25,0))
        pygame.init()
        screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("REVERSI/OTHELLO v. 8.3")
        response_font = pygame.font.SysFont(None, 24)
    
        clock = pygame.time.Clock()
        input_rect = pygame.Rect(MARGIN, HEIGHT - MARGIN + 40, WIDTH - MARGIN * 2, 40)


        screen.fill(self.BLACK)
        self.draw_board_from_board_list(screen, board)
        self.draw_input_prompt(screen, prompt)
        pygame.draw.rect(screen, self.BLACK, input_rect, 2)
        response_surface = response_font.render("Your move: ", True, self.WHITE)
        # Calculate the y-coordinate based on the height of the window and the margin
        text_height = response_surface.get_height()
        active = True

        while active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if g.check_if_input_is_valid(inputText, player) == True:
                            active = False
                            break
                        else:
                            print("INVALID INPUT")
                    elif event.key == pygame.K_BACKSPACE:
                        inputText = inputText[:-1]
                    else:
                        inputText = inputText + event.unicode
            """
            screen.fill(self.BLACK)
            self.draw_board_from_board_list(screen, board)
            self.draw_input_prompt(screen, prompt)
            pygame.draw.rect(screen, self.BLACK, input_rect, 2)
            """
            
            self.blit_input(screen,inputText,self.WHITE)
            # Issue is here
            #response_surface = response_font.render("Your move: " + inputText, True, self.WHITE)
            # Calculate the y-coordinate based on the height of the window and the margin
            text_height = response_surface.get_height()
            y_coordinate = self.HEIGHT - self.MARGIN - text_height
            #screen.blit(response_surface, (self.MARGIN, y_coordinate))
            self.draw_scoreboard(screen,scoreboard_data)
            
            pygame.display.flip()
            clock.tick(30)
            if active==False:
                break

        #pygame.display.update(rectangle) #this line, though not in correct place, will just update some things
        return inputText
    def show_board(self,board):
        def draw_board(screen):
            ROWS=self.ROWS
            COLS=self.COLS
            WHITE=self.WHITE
            BLACK=self.BLACK
            MARGIN=self.MARGIN
            SQUARE_SIZE=self.SQUARE_SIZE
            BUTTON_COLOR=self.BUTTONCOLOR
            GREEN=self.GREEN
            # Draw the white and black squares with black outlines and a margin
            for row in range(ROWS):
                for col in range(COLS):
                    color = GREEN
                    pygame.draw.rect(screen, BLACK, (MARGIN + col * SQUARE_SIZE, MARGIN + row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 1)  # Outline each square in black
                    pygame.draw.rect(screen, color, (MARGIN + col * SQUARE_SIZE + 1, MARGIN + row * SQUARE_SIZE + 1, SQUARE_SIZE - 1, SQUARE_SIZE - 1))  # Fill the square with the appropriate color

            # Label rows with numbers
            for i in range(ROWS):
                font = pygame.font.SysFont(None, 24)
                text = font.render(self.LETTERS[i], True, WHITE)
                screen.blit(text, (5, MARGIN + i * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_height() // 2))

            # Label columns with letters
            for i in range(COLS):
                font = pygame.font.SysFont(None, 24)
                #text = font.render(str(ROWS-i), True, WHITE)
                text=font.render(str(i+1),True,WHITE)
                screen.blit(text, (MARGIN + i * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_width() // 2, 5))

        def draw_checker(screen, row, col,color):
            SQUARE_SIZE = self.SQUARE_SIZE
            # Calculate the center of the specified square
            center_x = self.MARGIN + col * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = self.MARGIN + row * SQUARE_SIZE + SQUARE_SIZE // 2
            colors=self.textColors
            c=colors[color-1]
            pygame.draw.circle(screen, c, (center_x, center_y), SQUARE_SIZE // 3)

        def draw_button(screen, text, x, y, width, height):
            BUTTON_COLOR=self.BUTTONCOLOR
            BUTTON_TEXT_COLOR = self.BUTTONTEXTCOLOR
            # Draw button rectangle
            pygame.draw.rect(screen, BUTTON_COLOR, (x, y, width, height))
            # Draw button text
            font = pygame.font.SysFont(None, 24)
            text_surface = font.render(text, True, BUTTON_TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
            screen.blit(text_surface, text_rect)
        def draw_scoreboard(screen,scoreboard_data):
            # Draw the scoreboard area using dictionary data
            pygame.init()
            font=pygame.font.SysFont(None, 24)
            pygame.draw.rect(screen, self.BLACK, (self.SQUARE_SIZE * self.COLS + self.MARGIN, 0, self.SCOREBOARD_WIDTH, self.HEIGHT))
            # Render and display scoreboard text
            count=-2
            for i, (label, value) in enumerate(scoreboard_data.items()):
                coLor=self.WHITE
                text_surface = font.render(f"{label}: {value}", True, coLor)
                text_rect = text_surface.get_rect(x=self.COLS*self.SQUARE_SIZE + self.MARGIN + 20, y=50 + i * 40)
                screen.blit(text_surface, text_rect)
                
                count=count+1

        def main(board):
            pygame.init()
            screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("REVERSI/OTHELLO")

            clock = pygame.time.Clock()
            screen.fill(self.BLACK)
            draw_board(screen)

            b=Board((self.ROWS,self.COLS))
            b.import_board(board)
            playerScores=b.get_score()
            e=playerScores.pop(0)
            #playerList=["Empty squares remaining"]
            scoreboard_data = {"Scoreboard":"","Empty":e}
            for x in range(0,len(self.playerList)):
                player1=self.playerList[x]
                scoreboard_data[player1]=playerScores[x]
                              
                
                
            scoreboard_data["Turn"]="[Computer]"

            # Draw a checker on the square at row 3, column 'd' (0-indexed)
            for x in range(0,len(board)):
                list1=board[x]
                ##finish by placing all checkers on board
                for y in range(0,len(list1)):
                    item=list1[y]
                    if item>0:
                        self.draw_checker(screen,x,y,item)

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        # Check if the button is clicked
                        if self.WIDTH // 2 - 50 <= event.pos[0] <= self.WIDTH // 2 + 50 and self.HEIGHT - 70 <= event.pos[1] <= self.HEIGHT - 30:
                            running = False  # Close the window
                    elif event.type == pygame.KEYDOWN: #pressing enter works too
                        if event.key == pygame.K_RETURN:
                            running=False

                draw_scoreboard(screen,scoreboard_data)
                draw_button(screen, "Computer's turn!", self.WIDTH//2 - 75, self.HEIGHT - 70, 150, 40)
                pygame.display.flip()
                clock.tick(20)
                
            pygame.quit()
        main(board)
        

class Board:
    def __init__(self,boardSize):
        self.boardSize=boardSize
        x,y=boardSize
        self.x=x-1
        self.y=y-1
        
        self.build_board_lists()
    def build_board_lists(self):
        """
        build_board_lists(boardSize-->tupl(width,height))
        Builds a list based on dimensions in tuple
        Sample:

        build_board_lists((4,4))

        OUTPUT:
        [
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]]
        """
        board=[]
        x1,y1=self.boardSize
        for x in range(0,x1+1):
            list1=[]
            for y in range(0,y1):
                list1.append(0)
            board.append(list1)
        self.board=board
        
    def import_board(self,board):
        self.board=board
    def export_board(self):
        return self.board

    def translate_coordinates(self,coord):
        #letter, then number
        NUMS="1234567890"
        l=""
        n=""
        coord=str(coord).upper()
        for digit in coord:
            if digit in NUMS:
                n = n+digit
            else:
                l = l+digit

        try:
            x=ord(l)-65
            y=int(n)-1
        except ValueError:
            #print("Sorry, invalid coordinates")
            return "INVALID COORDINATES"
        return (x,y)
    
    def find_value_at_space(self,x,y):
        """
        Uses x and y coordinates derived with translate_coordinates function to determine value
        """
        try:
            value=self.board[x][y]
        except IndexError:
            return ""
        return value
    def find_bordering_spaces(self,x,y):
        neighbors=[]
        board=self.board
        
        for xadd in range(0,3):
            coordx=x+xadd
            for yadd in range(0,3):
                coordy=y+yadd
                if (x,y)==(coordx,coordy):
                    pass
                elif coordx==0 or coordy==0:
                    pass
                elif coordx>self.x or coordy>self.y:
                    pass
                else:
                    neighbors.append((coordx,coordy))
        return neighbors
    def update_item(self,x,y,replaceWith):
        """
        Uses x and y coordinates derived with translate_coordinates function to determine value
        """
        if x<=self.x and y<=self.y:
            board=self.board
            board[x][y]=replaceWith
            
        else:
            print("NOT POSSIBLE!")
    def set_middle_four_squares(self):
        yColumns=[math.floor(self.y/2),math.ceil((self.y+1)/2)]
        xRows=[math.floor(self.x/2),math.ceil((self.x+1)/2)]
        squares=[]
        for x in range(0,2):
            for y in range(0,2):
                squares.append((xRows[x],yColumns[y]))
        
        p1S=[squares.pop(-1),squares.pop(0)]
        p2S=squares
        for tupl in p1S:
            xC,yC=tupl
            self.update_item(xC,yC,1)
        for tupl in p2S:
            xC,yC=tupl
            self.update_item(xC,yC,2)
    def get_score(self):
        list1=[0,0,0] #empty, p1, p2
        for listA in self.board:
            for x in range(0,len(list1)):
                item=listA.count(x)
                list1[x]=list1[x]+item
        list1[0]=list1[0]-self.x-1
                
        return list1
    def get_edges(self):
        edges=[]
        board=self.board
        y=len(board[0])
        x=len(board)
        for x1 in range(0,x):
            for y1 in range(0,y):
                if x1==0 or x1==(x-1) or y1==0 or y1==(y-1):
                    edges.append((x1,y1))
        return edges
    def get_corners(self):
        board=self.board
        y=len(board[0])
        x=len(board)
        corners=[(0,y-1),(0,0),(x-1,0),(x-1,y-1)]
        return corners
        
    def print_board(self,characterList):

        board=self.board
        board2=[]
        for list2 in board:
            list3=[]
            for item in list2:
                if item==0:
                    list3.append(" ")
                elif item==1:
                    list3.append(characterList[0])
                else:
                    list3.append(characterList[1])
            board2.append(list3)
                
        boardToPrint=[]
        x,y=self.x,self.y
        h=["[]"]
        c=[]
        rows=[]
        ALPHA="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for x in range(0,self.x+1):
            c.append(ALPHA[x])
        for y in range(0,self.y+1):
            h.append(str(y+1))
        for z in range(0,len(board2)-1):
            
            list1=board2[z]
            columnHeader=c[z]
            row=[columnHeader]+list1
            
            rows.append(row)
        from texttable import Texttable

        # Create a Texttable object
        table = Texttable()

        # Set gridlines
        table.set_chars(['-', '|', '+', '-'])

        # Set maximum width to prevent newline between digits
        table.set_max_width(100)


        # Add columns
        table.header(h)

        # Add rows
        for row in rows:
            table.add_row(row)

        # Print the table
        return table.draw()
            
class Computer:
    def __init__(self,board):
        self.EMPTY = 0
        self.BLACK = 1
        self.WHITE = 2
        self.board = board
        self.x=len(board)
        self.y=len(board[0])
    def get_legal_moves(self,board, color):
        def opposite_color(color):
            return 1 if color ==2 else 2

        # Function to check if a move is within the bounds of the board
        def is_valid_move(board, x, y):
            return 0 <= x < len(board) and 0 <= y < len(board[0])
        def is_valid_capture(board, x, y, color):
            if board[x][y] != 0:
                return False
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = x + dx, y + dy
                found_opponent = False
                while is_valid_move(board, nx, ny) and board[nx][ny] == opposite_color(color):
                    nx, ny = nx + dx, ny + dy
                    found_opponent = True
                if found_opponent and is_valid_move(board, nx, ny) and board[nx][ny] == color:
                    return True
            return False
        
        moves = []
        for x in range(len(board)):
            for y in range(len(board[0])):
                if is_valid_capture(board, x, y, color):
                    moves.append((x, y))
        return moves
    def get_pieces_to_reverse(self,move):
        board=self.board
        x, y = move
        color = board[x][y]
        opponent_color = 1 if color == 2 else 2
        pieces_to_reverse = []

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            x_dir, y_dir = dx, dy
            temp_pieces = []
            x, y = move
            x = x + dx
            y = y + dy
            while 0 <= x < self.x and 0 <= y < self.y and board[x][y] == opponent_color:
                temp_pieces.append((x, y))
                x = x + dx
                y = y + dy
            if 0 <= x < self.x and 0 <= y < self.y and board[x][y] == color:
                pieces_to_reverse.extend(temp_pieces)

        return pieces_to_reverse
    
    def select_move(self,color, computerLevel,selecting):
        def opposite_color(color):
            return 1 if color ==2 else 2

        # Function to check if a move is within the bounds of the board
        def is_valid_move(board, x, y):
            return 0 <= x < len(board) and 0 <= y < len(board[0])

        # Function to check if a move is valid and captures opponent pieces
        def is_valid_capture(board, x, y, color):
            if board[x][y] != 0:
                return False
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = x + dx, y + dy
                found_opponent = False
                while is_valid_move(board, nx, ny) and board[nx][ny] == opposite_color(color):
                    nx, ny = nx + dx, ny + dy
                    found_opponent = True
                if found_opponent and is_valid_move(board, nx, ny) and board[nx][ny] == color:
                    return True
            return False

        # Function to make a move on the board and capture opponent pieces
        def make_move(board, x, y, color):
            board[x][y] = color
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = x + dx, y + dy
                if is_valid_move(board, nx, ny) and board[nx][ny] == opposite_color(color):
                    while is_valid_move(board, nx, ny) and board[nx][ny] == opposite_color(color):
                        nx, ny = nx + dx, ny + dy
                    if is_valid_move(board, nx, ny) and board[nx][ny] == color:
                        while (nx, ny) != (x, y):
                            nx, ny = nx - dx, ny - dy
                            board[nx][ny] = color

        # Function to evaluate the score of the board for a given color
        def evaluate_board(board, color):
            score = 0
            for row in board:
                for cell in row:
                    if cell == color:
                        score += 1
            return score

        # Function to determine all legal moves for a given color
        def get_legal_moves(board, color):
            moves = []
            for x in range(len(board)):
                for y in range(len(board[0])):
                    if is_valid_capture(board, x, y, color):
                        moves.append((x, y))
            return moves

        # Function for the computer to select the best move using minimax algorithm with alpha-beta pruning
        
        def select_move(board, color, depth,loopNum,modifier):
            import random

            def minimax(board, depth, alpha, beta, maximizing_player, color, variation=0):
                if depth == 0 or len(get_legal_moves(board, 1)) == 0 or len(get_legal_moves(board, 2)) == 0:
                    return evaluate_board(board, color)
                
                if maximizing_player:
                    max_eval = float('-inf')
                    best_moves = []
                    for move in get_legal_moves(board, color):
                        new_board = copy.deepcopy(board)
                        make_move(new_board, *move, color)
                        eval = minimax(new_board, depth - 1, alpha, beta, False, color)
                        max_eval = max(max_eval, eval)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
                        best_moves.append((eval, move))
                    if variation > 0 and len(best_moves) > 1:
                        max_eval -= random.uniform(0, variation)
                    return max_eval
                else:
                    min_eval = float('inf')
                    best_moves = []
                    for move in get_legal_moves(board, opposite_color(color)):
                        new_board = copy.deepcopy(board)
                        make_move(new_board, *move, opposite_color(color))
                        eval = minimax(new_board, depth - 1, alpha, beta, True, color)
                        min_eval = min(min_eval, eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
                        best_moves.append((eval, move))
                    if variation > 0 and len(best_moves) > 1:
                        min_eval += random.uniform(0, variation)
                    return min_eval
                
            best_move = None

            if loopNum>15:
                 best_move=random.choice(get_legal_moves(board, color))
            else:
                max_eval = float('-inf')
                for move in get_legal_moves(board, color):
                    new_board = copy.deepcopy(board)
                    make_move(new_board, *move, color)
                    eval = minimax(new_board, depth - 1, float('-inf'), float('inf'), False, color,variation=modifier)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
            #print(max_eval)
            return best_move

        # Example usage
        legals=self.get_legal_moves(self.board,color+1)
        bo=Board((self.x,self.y))
        edges=bo.get_edges()
        corners=bo.get_corners()
        modifier=0
        if computerLevel>6:
            avgBoardWidth=(self.x+self.y)/2
            cornerBonus=math.ceil(1.75*avgBoardWidth)
            edgeBonus=math.ceil(0.75*avgBoardWidth)
            for legal in legals:
                if legal in corners:
                    modifier=modifier+cornerBonus
                    #other one not necessary because you can't jump corners.
            for legal in legals:
                r=self.get_pieces_to_reverse(legal)
                for item in r:
                    if item in edges:
                        modifier=modifier+edgeBonus
                if legal in edges:
                    modifier=modifier+edgeBonus
        depth = computerLevel-3
        selected_move = select_move(self.board,color+1, depth,selecting,modifier)
        return selected_move

class Game:
    def __init__(self,boardSize):
        self.bsz=boardSize
        self.w,self.h=boardSize
        b=Board(boardSize)
        b.set_middle_four_squares()
        self.board=b.export_board()
        self.consecutivePasses=0
        
    def determine_legal_moves(self,color):
        board=self.board
        legal_moves = []
        opponent_color = 1 if color == 2 else 2
        """
        for row in range(len(board)-1):
            for col in range(len(board[row])-1):
        """
        for row in range(len(board)-1):
            for col in range(len(board[row])):
                if board[row][col] == 0:
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                        x, y = row + dx, col + dy
                        if 0 <= x < self.w and 0 <= y < self.h and board[x][y] == opponent_color:
                            while 0 <= x <= self.w and 0 <= y < self.h and board[x][y] == opponent_color:
                                x += dx
                                y += dy
                            if 0 <= x < self.w and 0 <= y < self.h and board[x][y] == color:
                                legal_moves.append((row, col))
                                break

        return legal_moves
    def get_pieces_to_reverse(self,move):
        board=self.board
        x, y = move
        color = board[x][y]
        opponent_color = 1 if color == 2 else 2
        pieces_to_reverse = []

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            x_dir, y_dir = dx, dy
            temp_pieces = []
            x, y = move
            x = x + dx
            y = y + dy
            while 0 <= x < self.w and 0 <= y < self.h and board[x][y] == opponent_color:
                temp_pieces.append((x, y))
                x = x + dx
                y = y + dy
            if 0 <= x < self.w and 0 <= y < self.h and board[x][y] == color:
                pieces_to_reverse.extend(temp_pieces)

        return pieces_to_reverse

    def reverse_pieces(self,move,color):
        x1,y1=move
        toReverse=self.get_pieces_to_reverse(move)
        #print(toReverse)
        bored=Board(boardSize)
        bored.import_board(self.board)
        for x,y in toReverse:
            bored.update_item(x,y,color)
        self.board=bored.export_board()
        
    def print_board(self,characters=["●","○"]):
        boarding=Board(boardSize)
        boarding.import_board(self.board)
        print("\n\n")
        print(boarding.print_board(characters))
    def find_number_of_characters_that_are_in_list(self,string,list1,lowerAll=False):
        total=0
        if lowerAll==True:
            string=string.lower()
            list2=[]
            for item in list1:
                list2.append(str(item).lower())
        else:
            list2=list1
        for item in string:
            if item in list2:
                total=total+1
        return total

    def translate_coordinates(self,coord):
        #letter, then number
        NUMS="1234567890"
        l=""
        n=""
        coord=str(coord).upper()
        for digit in coord:
            if digit in NUMS:
                n = n+digit
            else:
                l = l+digit

        try:
            x=ord(l)-65
            y=int(n)-1
        except ValueError:
            #print("Sorry, invalid coordinates")
            return "INVALID COORDINATES"
        return (x,y)
    def import_board(self,board):
        self.board=board
    def export_board(self):
        return self.board
    def import_playerList(self,playerList):
        self.playerList=playerList
        
    def check_if_input_is_valid(self,inputText,player,printStuff=True):
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        playerList=self.playerList
        color=playerList.index(player)
        legalMoves=self.determine_legal_moves(color+1)
        string2=alphabet+"~!@#$%^&*()_+`-=[]{}[]\\|;':\",./<>?"
        
        if inputText.lower()=="pass":
            return True
        else:
            colorz=["black","white"]
            num1=self.find_number_of_characters_that_are_in_list(inputText,list(str(34567890)),lowerAll=True)
            num11=self.find_number_of_characters_that_are_in_list(inputText,list(str(1)),lowerAll=True)
            num12=self.find_number_of_characters_that_are_in_list(inputText,list(str(2)),lowerAll=True)
            num2=self.find_number_of_characters_that_are_in_list(inputText,string2,lowerAll=True)
            num3=len(inputText)
            #print(num1,num11,num12,num2,num3)
            if num1<=1 and num11+num12<=2 and num2==1 and num3>1 and num3<4:
                #print(num11+num12)
                movex,movey=self.translate_coordinates(inputText)
                #valid space
                if (movex,movey) in legalMoves:
                    #legal move
                    if printStuff==True:
                        print("Move accepted.")
                    return True
                else:
                    if printStuff==True:
                        print("Illegal move!")
                        override=integer("OVERRIDE: Is "+str(inputText)+" a legal move? \n\t1 --> YES\n\t2 --> NO\n\t",maxAns=2,minAns=1)
                        if override==1:
                            print("TEST: Only legal moves were ")
                            for coords in legalMoves:
                                x,y=coords
                                possMove=chr(x+65)+str(y+1)
                                print(possMove)
                            return True
                        else:
                            inputText=""
                            return False
                        
                    return False
            else:
                if printStuff==True:
                    print("Can't comprehend your entry!")
                return False
                
    def determine_point_gain(self,x,y):
        points=1+len(self.get_pieces_to_reverse((x,y)))
        return points
    def find_outliers_in_dict(self,dictionary,largestOrSmallest=2):
        #2 is max in dict, 1 is min in dict
        keys=[]
        output=[]
        values=list(dictionary.values())
        if largestOrSmallest==2:
            value=max(values)
        else:
            value=min(values)
        for key in list(dictionary.keys()):
            if dictionary[key]==value:
                keys.append(key)
        for k in keys:
            v=dictionary[k]
            output.append((k,v))
        return output
        
    def one_turn(self,level):
        #easy -- goes for fewest point gain each turn
        #medium -- goes for most point gain each turn
        #hard -- picks randomly
        gu=GUI_Interface(self.w,self.h,len(self.playerList))
        playerList=self.playerList
        gu.import_player_list(playerList)
        board=self.board
        boardSize=self.bsz
        gameboard=Board(boardSize)
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        a,b=boardSize
        empty=a*b
        for color in range(0,len(playerList)):
            player=playerList[color]
            #self.print_board()
            gameboard.import_board(board)
            legalMoves=self.determine_legal_moves(color+1)
            if "[" in player:
                selecting=0
                failCheck=False
                print("\n\n\nIt's the computer's turn!")
                c=Computer(board)
                while failCheck==False:
                    if selecting==0:
                        print("Selecting move...")
                    selecting=selecting+1
                    move1=c.select_move(color, level,selecting)
                    x,y=move1
                    move=str(chr(x+65))+str(y+1)
                    
                    failCheck=self.check_if_input_is_valid(move,player,printStuff=False)
                    
                    if failCheck==True:
                        print("The computer's move: "+move.upper())
                        break
                    
                        
                            
            else:
                #gu.show_board(self.board)
                print("\n\n\nIt's your turn!")
                
                failCheck=False
                passTurn=False
                #move="pass"
                colorz=["black","white"]
                if len(legalMoves)>0:
                    while failCheck==False:
                        crashed=True
                        prompt="Please enter your move: "
                        while crashed==True:
                            try:
                                move=gu.main(prompt,player,board,playerList,colorz[color])
                                crashed=False
                                break
                            except Exception as e:
                                print("ERROR: "+str(e)+"-- please try again.")
                        #move=string(str(player)+" ("+colorz[y]+" checker), please enter the coordinates to which you'd like to place your marker this turn, or type PASS to pass: ")
                        failCheck=self.check_if_input_is_valid(move,player,printStuff=False)
                        if move=="pass":
                            passTurn=True
                            break
                        if failCheck==True:
                            break
                    print("Your move: "+str(move.upper()))
                    
            #results now
            if passTurn==True:
                print(str(player)+" chose to pass their turn!")
                self.consecutivePasses=self.consecutivePasses+1
            else:
                if len(legalMoves)==0:
                    print(str(player)+" had to pass, as they had no legal moves!")
                    self.consecutivePasses=self.consecutivePasses+1
                else:
                    movex,movey=self.translate_coordinates(move)
                    self.consecutivePasses=0
                    gameboard.update_item(movex,movey,color+1)
                    self.reverse_pieces((movex,movey),color+1)
                    self.board = gameboard.export_board()
                empty,p1,p2=gameboard.get_score()
                print("SCORE: \n\t"+str(playerList[0])+": "+str(p1)+"\n\t"+str(playerList[1])+": "+str(p2))
                if "[" not in player:
                    crashed=True
                    while crashed==True:
                        try:
                            gu.show_board(self.board)
                            crashed=False
                            break
                        except Exception as e:
                            print("ERROR: "+str(e)+"-- please try again")
                if empty==0:
                    break
                
        return (empty,p1,p2)    
    def setup(self):
        
        
        name="You"
        players=[name,"[Computer]"]
        print(str(name)+" will use the black checker. \n")
        print("The computer will use the white checker. \n")

        self.playerList=players
        time.sleep(1)
        print("-------------------------------------------")

    def full_game(self,level):
        turn=1
        self.setup()
        
        b=Board(self.bsz)
        b.import_board(self.board)
        empty,p1,p2 = b.get_score()
        while self.consecutivePasses<2 and empty>0:
            empty,p1,p2 = b.get_score()
            if empty<1:
                print("The gameboard is full, so the game is over.")
                break
            elif self.consecutivePasses>=2:
                print("Both players each passed once in a row, so the game is over.")
                break
            elif p1==0:
                print(self.playerList[0]+" was eliminated, so the game is over.")
                break
            elif p2==0:
                print(self.playerList[1]+" was eliminated, so the game is over.")
                break
            else:
                print("TURN: "+str(turn))
                oldP1=p1
                oldP2=p2
                empty,p1,p2=self.one_turn(level)
                p1Increase=p1-oldP1
                p2Increase=p2-oldP2
                print("INCREASES: P1: "+str(p1Increase)+", P2: "+str(p2Increase))
                turn=turn+1
                if empty<1:
                    print("The gameboard is full, so the game is over.")
                    break
                elif self.consecutivePasses>=2:
                    print("Both players each passed once in a row, so the game is over.")
                    break

        print("GAME OVER!")
        b=Board(self.bsz)
        b.import_board(self.board)
        
        print ("Final stats: ")
        e,p1,p2 = b.get_score()
        list1=[p1,p2]
        list1.sort()
        list1.reverse()
        if list1[0]==p1:
            print("YOU WON")
            won=True
        else:
            print("YOU LOST")
            won=False
        print("SPACES: \n\tEmpty: "+str(e)+"\n\tOccupied by "+self.playerList[0]+":"+str(p1)+"\n\tOccupied by "+self.playerList[1]+":"+str(p2))
        return won
        
                ###finish function
            
            
            
lockedList=["","","","","🔒","🔒","🔒","🔒"]   

save=Save_Game()
while 5>4:
    MENU=integer("Type 1 to import a save, 2 to export the current game, or 3 to play a game: ",minAns=1,maxAns=3)
    if MENU==1:
        l=save.find_and_import_save()
        if len(l)>0:
            lockedList=l
    elif MENU==2:
        save.export_a_save(lockedList)
    else:
        level,boardSize = initiation(lockedList,locks=True)
        g=Game(boardSize)
        won=g.full_game(level)
        if won==True:
            try:
                lockedList[level]=""
            except IndexError:
                print("Congratulations! You've unlocked and defeated all the levels!")


"""
gu=GUI_Interface(side,side)
gu.main(board)
"""
    
        
    
