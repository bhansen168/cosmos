"""
An attempt to train a supervised learning bot on historical data
"""

import os,sys,pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.append(os.getcwd())
from readWTB import parse_wtb
from computerRL import OthelloEnv

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
        games = []
        for file in self.files:
            new = parse_wtb(CompSupervised.WTHOR+"/"+file)
            games.extend(new)

        '''
        sample game:
            [(5, 6), (6, 4), (3, 3), (3, 4), (4, 3), (4, 6), (6, 6), (5, 7), (3, 5), (3, 6), (4, 7), (2, 5),
             (2, 4), (3, 8), (6, 5), (3, 7), (6, 3), (7, 6), (6, 7), (5, 3), (4, 2), (7, 5), (4, 8), (6, 8),
             (5, 8), (1, 3), (7, 4), (2, 3), (8, 6), (8, 5), (1, 5), (8, 7), (1, 6), (8, 4), (2, 6), (6, 2),
             (7, 3), (5, 2), (8, 3), (8, 2), (4, 1), (5, 1), (2, 7), (3, 1), (7, 7), (1, 4), (2, 8), (8, 8),
             (7, 8), (3, 2), (2, 2), (1, 8), (1, 7), (2, 1), (6, 1), (7, 1), (1, 1), (1, 2), (8, 1), (7, 2)]
        '''

        self.format_data(games)

    def format_data(self,games):
        #formats into 

        self.games = []
        env = OthelloEnv()
        for game in games:
            #example:
            gameFormatted = []
            board,_ = env.reset()
            for move in game:
                action = coord_to_index(move[1]-1, move[0]-1)#convert to computer indexing
                gameFormatted.append([board,action])
                board,_,_,_,_ = env.step(action)

            self.games.append(gameFormatted)

        print("Formatted data")       
        #print(f"GAMES: {len(games)}")

    def train(self, savePath="model.shk"):
        X_list = []
        y_list = []

        # 1. Unpack the nested structure from self.games
        # self.games is [[ [board1, act1], [board2, act2] ], [ [board1, act1], ... ]]
        for game in self.games:
            # We track the turn sequence per game to align perspective
            # Standard Othello: Black (1) starts first, White is (-1)
            current_player = 1 
            
            for board_state, action in game:
                # Ensure the board state is a flat numpy array
                flat_board = np.array(board_state).flatten()
                
                # PERSPECTIVE ALIGNMENT: 
                # Multiply by current_player so the model views its own pieces as +1.
                # This ensures the bot can play both Black and White using one model.
                aligned_board = flat_board * current_player
                
                X_list.append(aligned_board)
                y_list.append(action)
                
                # Alternate the player turn for the next move in this game
                current_player *= -1

        # 2. Convert lists to standard flat NumPy arrays for XGBoost
        X = np.array(X_list)
        y = np.array(y_list)

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
cs.train(savePath="demo.bard")
