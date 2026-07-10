from datetime import datetime



class Computer: 
    def __init__(self,game,color):
        self.game = game
        self.color = color
        self.cooldown = datetime.now()

    def pick_greedy(self):#maximizes short-term (one turn) score gain
        legal = self.game.get_all_legal_moves(self.color,returnJump = True)


        maxPair = None
        maxQty = 0

        for pair,jumped in legal:
            if len(jumped)>maxQty:
                maxQty = len(jumped)
                maxPair = pair

        mx,my = maxPair 
        self.game.place_piece(self.color,mx,my)

    def pick_random(self): #selects randomly
        legal = self.game.get_all_legal_moves(self.color,returnJump = True)
        mx,my = random.choice(legal)
        self.game.place_piece(self.color,mx,my)

