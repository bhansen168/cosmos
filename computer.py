from datetime import datetime



class Computer: 
    def __init__(self,game,color):
        self.game = game
        self.color = color
        self.cooldown = datetime.now()

    def pick_greedy(self,color = None,place = True):#maximizes short-term (one turn) score gain
        if color == None:
            color = self.color
        legal = self.game.get_all_legal_moves(color,returnJump = True)


        maxPair = None
        maxQty = 0

        for pair,jumped in legal:
            if len(jumped)>maxQty:
                maxQty = len(jumped)
                maxPair = pair

        mx,my = maxPair
        if place:
            self.game.place_piece(self.color,mx,my)
        else:
            return (mx,my)

    def pick_random(self): #selects randomly
        legal = self.game.get_all_legal_moves(self.color,returnJump = True)
        mx,my = random.choice(legal)
        self.game.place_piece(self.color,mx,my)

    def pick(self): #call in main.py
        return self.pick_greedy()

