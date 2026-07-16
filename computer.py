from datetime import datetime
import random


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


def create_genetic_computer(game,color,checkpoint_path):
    from genetic_model import GeneticPlayer
    return ModelComputer(game,color,GeneticPlayer.from_checkpoint(checkpoint_path))

