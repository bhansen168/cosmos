from datetime import datetime

class Computer: #maximizes score gain
    def __init__(self,game,color):
        self.game = game
        self.color = color
        self.cooldown = datetime.now()

    def pick(self):#precondition: legal is at least len=1
        legal = self.game.get_all_legal_moves(self.color,returnJump = True)


        maxPair = None
        maxQty = 0

        for pair,jumped in legal:
            if len(jumped)>maxQty:
                maxQty = len(jumped)
                maxPair = pair

        #move is maxPair
        mx,my = maxPair 
        self.game.place_piece(self.color,mx,my)

    
