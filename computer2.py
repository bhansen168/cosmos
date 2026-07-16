import os,sys
sys.path.append(os.getcwd())
from computer import Computer
from computerRL import load_agent,encode_state,legal_moves_to_np_arr,index_to_coord,coord_to_index
import torch
from computer_supervised import load_agent as load_agent_sup


class Computer2(Computer): #incorporates AI model -- use PTH extension
    PATH = os.getcwd()+"/models/checkpoints/othello_v02_2.0k-sav.pth"
    def __init__(self,game,color):
        super().__init__(game,color)
        self.agent = load_agent(Computer2.PATH)

    def pick(self):
        ind = self.agent.select_action(encode_state(self.game.board,self.color), legal_moves_to_np_arr(self.game.get_all_legal_moves(self.color),self.agent.actionDim), 0.0)

        y, x = index_to_coord(ind)
        legality = self.game.place_piece(self.color,x,y)

    def get_value_prediction(self):
        """Return the DQN's estimated value (win probability) for the current position."""
        state = encode_state(self.game.board, self.color)
        legal_moves = legal_moves_to_np_arr(self.game.get_all_legal_moves(self.color), self.agent.actionDim)
        return self.agent.get_value_prediction(state, legal_moves)

class Computer3(Computer):
    PATH = os.getcwd()+"/models/supervised/wthor-kaggle.bard"
    #dataset sourced from Kaggle (CSV) and WTHOR (French Othello Federation)
    def __init__(self,game,color,path=None):
        super().__init__(game,color)

        if path is None:
            path = Computer3.PATH
        self.agent = load_agent_sup(path)

    def pick(self):
        legal = self.game.get_all_legal_moves(self.color)
        x,y = self.agent.pick([coord_to_index(pair[1],pair[0]) for pair in legal],encode_state(self.game.board,self.color))
        legality = self.game.place_piece(self.color,x,y)

