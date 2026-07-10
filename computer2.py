import os,sys
sys.path.append(os.getcwd())
from computer import Computer
from computerRL import load_agent,encode_state,legal_moves_to_np_arr,index_to_coord

#PATH = os.getcwd()+"/models/othello_10k.pth"
PATH = os.getcwd()+"/models/othello_20.pth"

class Computer2(Computer): #incorporates AI model
    def __init__(self,game,color):
        global PATH
        super().__init__(game,color)
        self.agent = load_agent(PATH)

    def pick(self):
        ind = self.agent.select_action(encode_state(self.game.board,self.color), legal_moves_to_np_arr(self.game.get_all_legal_moves(self.color),self.agent.actionDim), 0.0)

        y, x = index_to_coord(ind)
        legality = self.game.place_piece(self.color,x,y)

        #print(f"TESTlegal: {legality}")
