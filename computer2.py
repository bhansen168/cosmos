import os,sys
sys.path.append(os.getcwd())
from computer import Computer
from comptuerRL import load_agent

PATH = os.getcwd()+"/models/othello_70.pth"

class Computer2(Computer): #incorporates AI model
    def __init__(self,game,color):
        global PATH
        super().__init__(game,color)
        self.agent = load_agent(PATH)

    def pick_move(self):
        pass
