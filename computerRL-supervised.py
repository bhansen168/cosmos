import os,sys,pickle
sys.path.append(os.getcwd())
from readWTB import parse_wtb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def predict_finish(start,amtCompleted):
    #start is datetime from start, amtCompleted is 0-1 decimal
    secsElapsed = (datetime.now()-start).total_seconds()
    totalSecs = int((1/amtCompleted)*secsElapsed)

    finish = datetime.now() + timedelta(seconds = totalSecs - secsElapsed)
    return str(finish).split(".")[0]

def index_to_coord(action_idx):
    if isinstance(action_idx,tuple): #already formatted correctly
        return action_idx
    
    y,x = divmod(action_idx,8)
    
    #y = action_idx // 8
    #x = action_idx % 8
    return y, x

def coord_to_index(y, x):
    return y * 8 + x

def encode_state(board,activePlayer):
    side = len(board)

    boardCopy = [[0 for _ in range(side)] for _ in range(side)]
    mapDict = {activePlayer:1,(1 if activePlayer==2 else 2):-1}
    for y in range(side):
        for x in range(side):
            if board[y][x] != 0:
                boardCopy[y][x] = mapDict[board[y][x]]

    return np.array(boardCopy).flatten()

def legal_moves_to_np_arr(legal,actionDim):
    global MUTE_PRINTS
    """
    Returns:
        legal_moves: A binary mask (list or NumPy array of 0s and 1s) 
                     matching the length of action_dim. 
                     1 indicates a legal position, 0 indicates a blocked/invalid move.
    """

    if not MUTE_PRINTS:
        print("TESTlegal:",legal)

    arr = np.zeros(actionDim)
    for mx,my in legal:
        arr[coord_to_index(my, mx)] = 1

    return arr

class CompSupervised:
    WTHOR = os.getcwd()+"/wthor"
    
    def __init__(self):
        self.files = [file for file in os.listdir(CompSupervised.WTHOR) if file.endswith(".wtb")]
        self.games = []
        for file in self.files:
            new = parse_wtb(CompSupervised.WTHOR+"/"+file)
            #print(f"{file}: {len(new)} games")
            new2 = []
            for game in new:
                new2.append([coord_to_index(tupl[1],tupl[0]) for tupl in game])
            
            self.games.extend(new2)

        #print(f"GAMES: {len(self.games)}")

    #still misnterpreting data storage
    '''
    def train(self,savePath = "model.shakespeare"):
        X = np.array([parse_board(b) for b in df['board_state']])
        y = df['next_move'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        model = XGBClassifier(
            objective='multi:softprob', 
            num_class=64, 
            n_estimators=100, 
            max_depth=6, 
            random_state=42
        )

        # Train the model to map board states to moves
        model.fit(X_train, y_train)

        with open(savePath,"wb") as file1:
            pickle.dump(model,file1)

class Agent:
    def __init__(self,model):
        self.model = model
        
    def pick(self,legal_moves):
        # Crucial Othello Rule: Filter out illegal moves
        # 'get_legal_moves' is a helper function based on standard Othello rules

        move_probabilities = model.predict_proba(encode_state())[0]

        # Find the highest probability move that is actually legal
        best_move = None
        best_prob = -1

        for move in legal_moves:
            if move_probabilities[move] > best_prob:
                best_prob = move_probabilities[move]
                best_move = move
    
def load_agent(file):
    with open(file,"rb") as file1:
        model = pickle.load(file1)

    a = Agent(model)
    return a

        

        
cs = CompSupervised()
'''
