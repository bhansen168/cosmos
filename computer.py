import os,sys,random
sys.path.append(os.getcwd())
from computer_supervised import (
    coord_to_index as coord_to_index_sup,
    encode_state as encode_state_sup,
    load_agent as load_agent_sup,
)

from datetime import datetime


class Computer: 
    name = "Greedy"

    def __init__(self,game=None,color=None):
        self.game = game
        self.color = color
        self.cooldown = datetime.now()

    def choose_move(self,game,color,legal_moves,rng):
        """Choose a move without placing it, for benchmarks and model training."""
        del game,color,rng
        move = max(legal_moves,key=lambda candidate: len(candidate.flips))
        return (move.x,move.y)

    def pick_greedy(self,color = None,place = True):#maximizes short-term (one turn) score gain
        if color == None:
            color = self.color
        legal = self.game.get_all_legal_moves(color,returnJump = True)

        if not legal:
            return None


        maxPair = None
        maxQty = 0

        for pair,jumped in legal:
            if len(jumped)>maxQty:
                maxQty = len(jumped)
                maxPair = pair

        mx,my = maxPair
        if place:
            self.game.place_piece(color,mx,my)
        else:
            return (mx,my)

    def pick_random(self, color = None, place = True): #selects randomly
        if color == None:
            color = self.color
        legal = self.game.get_all_legal_moves(color,returnJump = False)
        if not legal:
            return None
        mx,my = random.choice(legal)
        if place:
            self.game.place_piece(color,mx,my)
        else:
            return (mx,my)

    def pick(self): #call in main.py
        return self.pick_greedy()


class RandomComputer(Computer):
    """Original Computer interface using random rather than greedy moves."""
    name = "Random"

    def choose_move(self,game,color,legal_moves,rng):
        del game,color
        move = rng.choice(legal_moves)
        return (move.x,move.y)

    def pick(self):
        return self.pick_random()


class ModelComputer(Computer):
    """Expose a newer Player through the original bound Computer interface."""
    def __init__(self,game,color,player):
        super().__init__(game,color)
        self.player = player
        self.name = player.name
        self.rng = random.Random()

    def choose_move(self,game,color,legal_moves,rng):
        return self.player.choose_move(game,color,legal_moves,rng)

    def pick_model(self,color=None,place=True):
        if color == None:
            color = self.color
        legal = self.game.legal_moves(color)
        if not legal:
            return None
        x,y = self.player.choose_move(self.game,color,legal,self.rng)
        if place:
            selected = next(move for move in legal if move.x == x and move.y == y)
            self.game.play(color,selected)
        else:
            return (x,y)

    def pick_greedy(self,color=None,place=True):
        return self.pick_model(color,place)

    def pick_minimax(self,color=None,place=True):
        return self.pick_model(color,place)

    def pick(self):
        return self.pick_model()

    def get_value_prediction(self):
        """Return the model's estimated value for the current position."""
        if hasattr(self.player, 'evaluate'):
            # Genetic player uses evaluate(game, color)
            return self.player.evaluate(self.game, self.color)
        elif hasattr(self.player, 'get_value_prediction'):
            # DQN agent has get_value_prediction(state, legal_moves)
            # This is for the agent case, not used via ModelComputer
            pass
        return 0.0


def create_minimax_computer(game,color,depth=2):
    from minimax_model import MinimaxPlayer
    return ModelComputer(game,color,MinimaxPlayer(depth=depth))


def create_genetic_computer(game,color,checkpoint_path, search_depth=None):
    from genetic_model import GeneticPlayer
    player = GeneticPlayer.from_checkpoint(checkpoint_path)
    if search_depth is not None:
        player.search_depth = search_depth
    return ModelComputer(game,color,player)





class ComputerDQN(Computer): #incorporates AI model -- use PTH extension
    #formerly known as Computer2
    PATH = os.getcwd()+"/models/checkpoints/othello_v02_2.0k-sav.pth"
    def __init__(self,game,color,path=None):
        super().__init__(game,color)
        from computerRL import load_agent


        if path is None:
            path = ComputerDQN.PATH
        self.agent = load_agent(path)

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

class ComputerSupervised(Computer):
    #formerly known as Computer3
    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models","supervised","synth-20260720133316.bard")
    #dataset sourced from Kaggle (CSV) and WTHOR (French Othello Federation)
    def __init__(self,game=None,color=None,path=None):
        super().__init__(game,color)

        if path is None:
            path = ComputerSupervised.PATH
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

class ComputerGen(Computer): # Genetic algorithm model - latest
    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models","genetic","latest.json")
    def __init__(self,game=None,color=None,path=None):
        super().__init__(game,color)

        if path is None:
            path = ComputerGen.PATH
        self.path = os.path.abspath(path)
        self.computer = create_genetic_computer(game, color, self.path)
        self.name = f"Genetic ({os.path.basename(self.path)})"

    def pick(self):
        self.computer.pick()

    def get_value_prediction(self):
        """Return the genetic model's estimated value for the current position."""
        return self.computer.get_value_prediction()


class ComputerGen25(Computer): # Genetic algorithm model - 25th generation
    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models","genetic","genetic_gen_0024.json")
    def __init__(self,game=None,color=None,path=None):
        super().__init__(game,color)

        if path is None:
            path = ComputerGen25.PATH
        self.path = os.path.abspath(path)
        self.computer = create_genetic_computer(game, color, self.path)
        self.name = f"Genetic 25th Gen ({os.path.basename(self.path)})"

    def pick(self):
        self.computer.pick()

    def get_value_prediction(self):
        """Return the genetic model's estimated value for the current position."""
        return self.computer.get_value_prediction()


