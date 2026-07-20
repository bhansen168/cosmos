import os,sys
sys.path.append(os.getcwd())
from computer import Computer
from computer_supervised import (
    coord_to_index as coord_to_index_sup,
    encode_state as encode_state_sup,
    load_agent as load_agent_sup,
)
from computer import create_genetic_computer


class Computer2(Computer): #incorporates AI model -- use PTH extension
    PATH = os.getcwd()+"/models/checkpoints/othello_v02_1.0k-sav.pth"
    name = "DQN"
    def __init__(self,game,color):
        super().__init__(game,color)
        from computerRL import load_agent

        self.agent = load_agent(Computer2.PATH)

    def pick(self):
        from computerRL import encode_state,index_to_coord,legal_moves_to_np_arr

        ind = self.agent.select_action(encode_state(self.game.board,self.color), legal_moves_to_np_arr(self.game.get_all_legal_moves(self.color),self.agent.actionDim), 0.0)

        y, x = index_to_coord(ind)
        legality = self.game.place_piece(self.color,x,y)

    def get_value_prediction(self):
        """Return the DQN's estimated value (win probability) for the current position."""
        from computerRL import encode_state,legal_moves_to_np_arr

        state = encode_state(self.game.board, self.color)
        legal_moves = legal_moves_to_np_arr(self.game.get_all_legal_moves(self.color), self.agent.actionDim)
        return self.agent.get_value_prediction(state, legal_moves)

class Computer3(Computer):
    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models","supervised","wthor-kaggle.bard")
    #dataset sourced from Kaggle (CSV) and WTHOR (French Othello Federation)
    def __init__(self,game=None,color=None,path=None):
        super().__init__(game,color)

        if path is None:
            path = Computer3.PATH
        self.path = os.path.abspath(path)
        self.agent = load_agent_sup(self.path)
        self.name = f"Bard supervised ({os.path.basename(self.path)})"

    def choose_move(self,game,color,legal_moves,rng):
        del rng
        indexed_legal_moves = [
            coord_to_index_sup(move.y,move.x) for move in legal_moves
        ]
        selected = self.agent.pick(
            indexed_legal_moves,
            encode_state_sup(game.board,color),
        )
        if selected is None:
            raise RuntimeError("Bard did not select a move")
        coordinate = int(selected[0]),int(selected[1])
        if coordinate not in {(move.x,move.y) for move in legal_moves}:
            raise RuntimeError(f"Bard selected illegal move {coordinate}")
        return coordinate

    def pick(self):
        legal = self.game.legal_moves(self.color)
        if not legal:
            return
        x,y = self.choose_move(self.game,self.color,legal,None)
        self.game.place_piece(self.color,x,y)


class Computer4(Computer): # Genetic algorithm model - latest
    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models","genetic","latest.json")
    def __init__(self,game=None,color=None,path=None):
        super().__init__(game,color)

        if path is None:
            path = Computer4.PATH
        self.path = os.path.abspath(path)
        self.computer = create_genetic_computer(game, color, self.path)
        self.name = f"Genetic ({os.path.basename(self.path)})"

    def pick(self):
        self.computer.pick()

    def get_value_prediction(self):
        """Return the genetic model's estimated value for the current position."""
        return self.computer.get_value_prediction()


class Computer5(Computer): # Genetic algorithm model - 25th generation
    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models","genetic","genetic_gen_0024.json")
    def __init__(self,game=None,color=None,path=None):
        super().__init__(game,color)

        if path is None:
            path = Computer5.PATH
        self.path = os.path.abspath(path)
        self.computer = create_genetic_computer(game, color, self.path)
        self.name = f"Genetic 25th Gen ({os.path.basename(self.path)})"

    def pick(self):
        self.computer.pick()

    def get_value_prediction(self):
        """Return the genetic model's estimated value for the current position."""
        return self.computer.get_value_prediction()


class Computer6(Computer): # Genetic algorithm model - latest depth 5
    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models","genetic","latest.json")
    def __init__(self,game=None,color=None,path=None):
        super().__init__(game,color)

        if path is None:
            path = Computer6.PATH
        self.path = os.path.abspath(path)
        self.computer = create_genetic_computer(game, color, self.path, search_depth=5)
        self.name = f"Genetic Latest Depth 5 ({os.path.basename(self.path)})"

    def pick(self):
        self.computer.pick()

    def get_value_prediction(self):
        """Return the genetic model's estimated value for the current position."""
        return self.computer.get_value_prediction()


class Computer7(Computer): # Genetic algorithm model - 25th gen depth 5
    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models","genetic","genetic_gen_0024.json")
    def __init__(self,game=None,color=None,path=None):
        super().__init__(game,color)

        if path is None:
            path = Computer7.PATH
        self.path = os.path.abspath(path)
        self.computer = create_genetic_computer(game, color, self.path, search_depth=5)
        self.name = f"Genetic 25th Gen Depth 5 ({os.path.basename(self.path)})"

    def pick(self):
        self.computer.pick()

    def get_value_prediction(self):
        """Return the genetic model's estimated value for the current position."""
        return self.computer.get_value_prediction()

