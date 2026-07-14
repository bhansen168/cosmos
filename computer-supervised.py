"""
An attempt to train a supervised learning bot on historical data
"""

import os,sys,pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
    
    out = y * 8 + x
    if out<0:
        print(f"TEST: x={x}, y={y}, out={out}")
    return out


def index_to_xgb(idx):
    ct = 0
    missing = [27,28,35,36]
    for val in missing:
        if idx>val:
            ct+=1
    return idx-ct

XGB_DICT = {index_to_xgb(i):i for i in (list(range(27)) + list(range(29,35)) + list(range(37,65)))}

def xgb_to_index(xgb): #in process of determining; placeholder
    global XGB_DICT
    return XGB_DICT[xgb]
    
        
    

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

        #for y in range(8):
        #    for x in range(8):
        #        print(f"TEST: x={x}, y={y}, act={coord_to_index(y,x)}")

        self.format_data(games)

    def format_data(self, games):
        env = OthelloEnv()
        self.games = []
        
        for game in games:
            gameFormatted = []
            board, _ = env.reset() # Assuming env.reset() returns (initial_board, info)
            
            for move in game:
                mx, my = move
                
                # Check for Othello WTHOR Pass notation (often 0,0 or specifically flagged)
                # Adjust these coordinates based on how your parse_wtb labels a pass:
                if mx == 0 and my == 0: 
                    # It's a pass! Assign it the 64th index.
                    action = 64 
                else:
                    mx -= 1
                    my -= 1
                    action = coord_to_index(my, mx)
                
                # CRITICAL: Grab the actual active player color from your environment 
                # BEFORE taking the step, so perspective alignment is 100% correct.
                # Adjust 'env.current_player' to match your actual OthelloEnv attribute name.
                active_player = env.current_player 
                
                gameFormatted.append([board, action, active_player])
                
                # Advance the environment
                if action < 64: #64 is pass
                    board, _, _, _, _ = env.step(action)
                
            self.games.append(gameFormatted)
        #print("Formatted data!")
    def train(self, savePath="model.bard"):
        X_list = []
        y_list = []
        
        # 1. Unpack the nested structure from self.games
        for game in self.games:
            for board_state, action,_ in game:
                flat_board = np.array(board_state).flatten()
                aligned_board = flat_board
                X_list.append(aligned_board)
                y_list.append(action)

        X = np.array(X_list)
        
        # 2. CONVERT LABELS TO THE DENSE 0-59 RANGE USING YOUR FUNCTION
        y_dense = np.array([index_to_xgb(act) for act in y_list])

        # 3. STRATIFIED SPLIT
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_dense, 
            test_size=0.1, 
            random_state=42, 
            stratify=y_dense  # Now safe because there are no missing gaps
        )

        # 4. INITIALIZE XGBOOST WITH EXACTLY 60 CLASSES
        model = XGBClassifier(
            objective='multi:softprob', 
            num_class=60,                 # Locked down to your 60 dense classes
            n_estimators=100, 
            max_depth=6, 
            random_state=42,
            eval_metric='mlogloss'
        )

        # 5. TRAIN AND SAVE
        model.fit(X_train, y_train)

        with open(savePath, "wb") as file1:
            pickle.dump(model, file1)
            
        print(f"Model successfully saved to {savePath}!")

class Agent:
    def __init__(self,model):
        self.model = model
        
    def pick_xgb(self, legal_moves, board_state):
        # Fallback if the environment forces a turn check when no moves exist
        if not legal_moves:
            return 64 # Assuming 64 represents a pass in your game controller
            
        # Get raw probabilities (array length 60)
        # Note: Ensure encode_state() passes the active board array to your model
        move_probabilities = self.model.predict_proba(board_state)[0] 
        
        best_move = None
        best_prob = -1
        
        for move in legal_moves:
            # Skip the pass move if it's mixed into legal moves, or handle it explicitly
            if move == 64: 
                continue 
                
            # Convert Othello coordinate (0-63) to XGBoost coordinate (0-59)
            xgb_slot = index_to_xgb(move)
            
            # Safely query the probability from the 60-element output array
            prob = move_probabilities[xgb_slot]
            
            if prob > best_prob:
                best_prob = prob
                best_move = move
                
        # If no grid moves were selected, default to your pass action index
        return best_move if best_move is not None else 64

    def pick(self,legal,board,asCoord = False):
        #game should automatically pass, ignore case 64
        sel = self.pick_xgb(legal,board)
        
        if sel == 64: #PASS
            return

        othelloIdx = xgb_to_index(sel)
        if asCoord:
            y,x = index_to_coord(othelloIdx)
            return (x,y)
        return othelloIdx
        
    
def load_agent(file):
    with open(file,"rb") as file1:
        model = pickle.load(file1)

    a = Agent(model)
    return a

        
if __name__ == "__main__":
    cs = CompSupervised()
    cs.train(savePath=os.getcwd()+"/models/demo.bard")
