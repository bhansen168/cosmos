import os,sys
sys.path.append(os.getcwd())
from readWTB import parse_wtb

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
            print(f"{file}: {len(new)} games")
            self.games.extend(parse_wtb(file))

        print(f"GAMES: {len(self.games)}")


cs = CompSupervised()
