import os,sys
sys.path.append(os.getcwd())
from computer import Computer
from computerRL import load_agent,encode_state,legal_moves_to_np_arr,index_to_coord
import torch

PATH = os.getcwd()+"/models/checkpoints/othello_v02_1.0k-sav.pth"
#PATH = os.getcwd()+"/models/othello_v02_2k.pth"

class Computer2(Computer): #incorporates AI model
    def __init__(self,game,color):
        global PATH
        super().__init__(game,color)
        self.agent = load_agent(PATH)

    def pick(self):
        ind = self.agent.select_action(encode_state(self.game.board,self.color), legal_moves_to_np_arr(self.game.get_all_legal_moves(self.color),self.agent.actionDim), 0.0)

        y, x = index_to_coord(ind)
        legality = self.game.place_piece(self.color,x,y)

    def get_value_prediction(self):
        """Return the DQN's estimated value (max Q) for the current position."""
        state = encode_state(self.game.board, self.color)
        legal_moves = legal_moves_to_np_arr(self.game.get_all_legal_moves(self.color), self.agent.actionDim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.agent.policyNet(state_t).squeeze(0)
            # Mask illegal moves
            q_values[legal_moves == 0] = -float('inf')
            return q_values.max().item()

    def get_value_prediction(self):
        """Return the DQN's estimated value (win probability) for the current position."""
        state = encode_state(self.game.board, self.color)
        legal_moves = legal_moves_to_np_arr(self.game.get_all_legal_moves(self.color), self.agent.actionDim)
        return self.agent.get_value_prediction(state, legal_moves)

