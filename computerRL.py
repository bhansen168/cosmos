import warnings,random,torch,sys,os
warnings.filterwarnings("ignore")

from torch import nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
import numpy as np
from datetime import datetime,timedelta

sys.path.append(os.getcwd())
from game import Game
from computer import Computer

MUTE_PRINTS = True
EPOCHS = 25000
VERSION = "v02"

def predict_finish(start,amtCompleted):
    #start is datetime from start, amtCompleted is 0-1 decimal
    secsElapsed = (datetime.now()-start).total_seconds()
    totalSecs = int((1/amtCompleted)*secsElapsed)

    finish = datetime.now() + timedelta(seconds = totalSecs - secsElapsed)
    return str(finish).split(".")[0]
    


class QNet(nn.Module):
    def __init__(self,params=64,actions=64):
        super().__init__()
        
        self.layer1 = nn.Linear(in_features=params, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=10)
        self.layer4 = nn.Linear(in_features=10, out_features=actions)

    def forward(self, x):
        x = F.relu(self.layer1(x)) #performs relu ops in between
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer =deque(maxlen=capacity)

    def push(self, state, action, reward, nextState, done):
        self.buffer.append((state,action,reward,nextState,done))

    def sample(self,batchSize):
        state,action,reward,nextState,done = zip(*random.sample(self.buffer,batchSize))
        return (torch.FloatTensor(state),torch.LongTensor(action),torch.FloatTensor(reward),torch.FloatTensor(nextState),torch.FloatTensor(done))
        
    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, stateDim, actionDim, lr=1e-3, gamma=0.99):
        self.actionDim = actionDim
        self.gamma = gamma
        
        self.policyNet = QNet(stateDim, actionDim)
        self.targetNet = QNet(stateDim, actionDim)
        self.targetNet.load_state_dict(self.policyNet.state_dict())

        self.id = 1 
        
        # FIX: Added () to .parameters() so PyTorch can register the weights
        self.optimizer = optim.Adam(self.policyNet.parameters(), lr=lr) 
        
    def select_action(self, state, legal_moves, epsilon):
        """
        legal_moves: binary mask list or numpy array of length actionDim (64)
        """
        # Convert to numpy array for fast masking operations
        legal_moves = np.array(legal_moves)
        
        if random.random() < epsilon:
            # EXPLORE: Find all indices where legal_moves == 1
            valid_indices = np.where(legal_moves == 1)[0]
            return random.choice(valid_indices)
            
        else:
            # EXPLOIT: Force network away from illegal positions
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policyNet(state_t).squeeze(0)
                
                # Clone values to avoid mutating live gradients
                masked_q = q_values.clone()

                # Find all indices where legal_moves == 0
                illegal_indices = np.where(legal_moves == 0)[0]
                
                # Force illegal moves to negative infinity so argmax never picks them
                masked_q[illegal_indices] = -float('inf')
                
                return masked_q.argmax().item()

def optimize(agent,memory,batchSize):
    if len(memory) < batchSize:
        return

    states,actions,rewards,nextStates,dones = memory.sample(batchSize)

    currentQ = agent.policyNet(states).gather(1,actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        nextQ = agent.targetNet(nextStates).max(1)[0]
        targetQ = rewards + (agent.gamma * nextQ * (1-dones))

    loss = F.mse_loss(currentQ,targetQ)
    agent.optimizer.zero_grad()
    loss.backward()#backpropagate?
    agent.optimizer.step()


class OpponentPool:
    def __init__(self, greedy_bot):
        self.greedy_bot = greedy_bot
        self.past_versions = []  # Stores saved state_dicts of your agent
        
    def add_checkpoint(self, agent_state_dict):
        # Save historical snapshots to prevent regression
        self.past_versions.append(agent_state_dict)
        
    def select_opponent(self, episode, total_episodes):
        # Phase 1: Early training heavily favors the greedy baseline
        if episode < (total_episodes * 0.15) or not self.past_versions:
            return self.greedy_bot
            
        # Phase 2: Self-play with a mix of past versions to prevent forgetting
        roll = random.random()
        if roll < 0.30:
            if len(self.past_versions)>0:
                return random.choice(self.past_versions) # 30% chance to play older selves

        if roll < 0.40:#baseline (10% chance)
            return self.greedy_bot
        else:
            return "LATEST_SELF"  # 60% chance to play against its most recent self

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




class OthelloEnv(Game):
    def __init__(self,side=8):
        super().__init__(side = side)
        self.current_player = 1 #player; 2 is player 2

        self.state_dim = 64
        self.action_dim = 64 #not really, but it makes it easier
        self.wins = 0
        """
    def _format_board(self):
        #formats board in favor of current player
        boardCopy = [[0 for _ in range(self.side)] for _ in range(self.side)]
        mapDict = {self.current_player:1,(1 if self.current_player==2 else 2):-1}
        for y in range(self.side):
            for x in range(self.side):
                if self.board[y][x] != 0:
                    boardCopy[y][x] = mapDict[self.board[y][x]]

        return boardCopy
        """

    def _flatten(self):
        # Flatten the nested list to a 1D NumPy array; perspective adjusted in _format_board
        '''
        flat_board = np.array(self._format_board()).flatten()
        return flat_board
        '''
        return encode_state(self.board,self.current_player)

    def classify_square(self,x,y):
        '''
        input: (x,y) -- coords
        output:
        0--normal
        1--edge
        2--corner
        '''

        xEdge = (x == self.side-1 or x == 0)
        yEdge = (y == self.side-1 or y == 0)

        return xEdge + yEdge
        

        
            

    def reset(self):
        """
        Resets the board. 
        Returns:
            state: The initialized board array from the perspective of the starting player.
            info: A dictionary containing extra metadata (optional).
        """
        self.board = [[0 for _ in range(self.side)] for _ in range(self.side)]
        self._set_middle()

        return self._flatten(),{}

    def step(self, action):
        global MUTE_PRINTS
        """
        Applies the chosen action to the board state.
        Returns:
            next_state: The new board array from the perspective of the NEXT active player.
            reward: 0.0 for mid-game turns. 
            done: True if a player wins, loses, draws, or the board is full.
            truncated: False (unless you use a hard step-limit turn counter).
            info: Extra metadata.
        """

        y, x = index_to_coord(action)

        legality = self.place_piece(self.current_player,x,y)
        if not MUTE_PRINTS:
            print(f"TEST: move: ({x},{y}) move_legal: {legality}")

        #do function

        gameOver = False

        self.current_player = (Game.WHITE if self.current_player == Game.BLACK else Game.BLACK)
        if len(self.get_all_legal_moves(self.current_player))==0:
            #no legal moves
            self.current_player = (Game.WHITE if self.current_player == Game.BLACK else Game.BLACK)

        if len(self.get_all_legal_moves(Game.WHITE if self.current_player == Game.BLACK else Game.BLACK)) == 0:
            gameOver = True

        if self.check_game_over():
            gameOver = True
        

        reward = self.classify_square(x,y)*0.02
        truncated = False
        
        return self._flatten(),reward,gameOver,truncated,{}

    def get_legal_moves(self):
        global MUTE_PRINTS
        """
        Returns:
            legal_moves: A binary mask (list or NumPy array of 0s and 1s) 
                         matching the length of action_dim. 
                         1 indicates a legal position, 0 indicates a blocked/invalid move.
        """
        legal = self.get_all_legal_moves(self.current_player)

        return legal_moves_to_np_arr(legal,self.action_dim)
        '''
        if not MUTE_PRINTS:
            print("TESTlegal:",legal)

        arr = np.zeros(self.action_dim)
        for mx,my in legal:
            arr[coord_to_index(my, mx)] = 1

        return arr
        '''
        

    def get_player_reward(self, player_id):
        """
        Evaluates the endgame state.
        Returns:
            reward (float): e.g., +1.0 if player_id won, -1.0 if they lost, 0.0 for a draw.
        """
        scores = self.get_score()
        vals = list(scores.values())
        if vals[0] == vals[1]:
            return 0.0 #draw
        elif scores[player_id] == max(vals):
            return 1 #win
        else:
            return -1 #lose

    def print_board(self): #for debugging
        string = ""
        empty = 0
        for y in range(self.side):
            for x in range(self.side):
                val = self.board[y][x]
                string += ("X" if self.board[y][x] == self.current_player else ("_" if self.board[y][x] == 0 else "O"))
                if self.board[y][x] == 0:
                    empty += 1
            string+="\n"

        print(string)
        print("NUM EMPTIES:",empty)
                
                
        

def load_agent(file):
    env = OthelloEnv()
    trained_agent = Agent(env.state_dim, env.action_dim)

    # 2. Load the file from disk and push the weights into the network
    weights = torch.load(file)
    trained_agent.policyNet.load_state_dict(weights)

    # 3. CRITICAL: Switch the network to evaluation mode 
    # This locks gradients and sets up layers properly for pure inference gameplay
    trained_agent.policyNet.eval()

    return trained_agent


if __name__ == "__main__":
    env = OthelloEnv()

    CHECKPOINT_FOLDER = os.getcwd()+"/models/checkpoints"
    MODEL_FOLDER = os.getcwd()+"/models"

    #memory
    memory = ReplayBuffer(capacity=20000)
    batch_size = 64
    
    #training loop
    pool = OpponentPool(greedy_bot=Computer(env,None))
    agent = Agent(env.state_dim,env.action_dim)
    historical_agent = Agent(env.state_dim,env.action_dim)
    
    num_episodes = EPOCHS

    epsilon = 1
    epsilon_decay = 0.9995
    min_epsilon = 0.01

    UPDATE = 500
    SAV_FREQ = max(min(int(EPOCHS * 0.05),20000),1000)

    start = datetime.now()

    print("Started training at "+(str(start).split(".")[0]))

    if not os.path.exists(CHECKPOINT_FOLDER):
        os.mkdir(CHECKPOINT_FOLDER)

    #models = os.listdir(os.getcwd()+"/models")

    for episode in range(num_episodes):
        try:
            epsilon = max(epsilon * epsilon_decay, min_epsilon)

            
            opponent_type = pool.select_opponent(episode, num_episodes)
            state, _ = env.reset()
            done = False
            
            while not done:
                current_player = env.current_player
                if not MUTE_PRINTS:
                    input("PRESS ENTER FOR NEXT TURN: ")
                
                if current_player == agent.id:
                    if not MUTE_PRINTS:
                        print("model to move")
                    
                    
                else:
                    # Opponent plays based on the selected pool strategy
                    if opponent_type == "LATEST_SELF":
                        if not MUTE_PRINTS:
                            print("opponent (self) to move")
                        action = agent.select_action(state, env.get_legal_moves(), epsilon) # Exploit self
                    elif opponent_type == pool.greedy_bot:
                        if not MUTE_PRINTS:
                            print("opponent (greedy) to move")
                        action = pool.greedy_bot.pick_greedy(color = current_player,place = False)
                        action = (action[1],action[0])#flip coords to match
                        #action = pool.greedy_bot.pick_greedy(state, env.get_legal_moves())
                    else:
                        if not MUTE_PRINTS:
                            print("opponent (historical) to move")
                        # Load historical model weights temporarily for the turn
                        historical_agent.policyNet.load_state_dict(opponent_type)
                        action = historical_agent.select_action(state, env.get_legal_moves(), epsilon=0.0)
                        
                    next_state, reward, done, _, _ = env.step(action)

                if done:
                    reward = env.get_player_reward(agent.id) #end of game stuff

                # Main Agent plays using epsilon-greedy & collects gradients
                try:
                    action = agent.select_action(state, env.get_legal_moves(), epsilon) #MAIN AGENT
                    next_state, reward, done, _, _ = env.step(action)

                    # (Store in Replay Buffer...)
                    if done:
                        reward = env.get_player_reward(agent.id)
                        
                    memory.push(state, action, reward, next_state, done)
                    state = next_state
                except IndexError as e:
                    #game over, but didn't break like was supposed to.
                    #print("TESTdone:",done) -- True -- means Done function works properly
                    pass
                    
                
                # 3. CRITICAL ADDITION: Run the optimizer optimization steps
                optimize(agent, memory, batch_size)
                    

            if episode % 10 == 0:
                agent.targetNet.load_state_dict(agent.policyNet.state_dict())
                    
            # Every 500 episodes, snapshot the agent and add it to the pool
            if episode>0:
                if episode % 500 == 0:
                    pool.add_checkpoint(agent.policyNet.state_dict())
                if episode % SAV_FREQ == 0:
                    path = f"{CHECKPOINT_FOLDER}/othello_{VERSION}_{round(episode/1000,1)}k-sav.pth"
                    torch.save(agent.policyNet.state_dict(), path)
                    print(f"Saved checkpoint at \"{path}\"; timestamp: {str(datetime.now()).split('.')[0]}")
            
                if (episode%UPDATE ==UPDATE-1):
                    perc = (episode+1)/num_episodes
                    print(f"FINISHED EPISODE {episode+1} OF {num_episodes} -- {round(perc * 100,2)}% -- ends at {predict_finish(start,perc)}")
        except KeyboardInterrupt as e:
            path = f"{MODEL_FOLDER}/othello_{VERSION}_{episode * 100//num_episodes}_ABORTED.pth"
            torch.save(agent.policyNet.state_dict(), path)
            print(f"Aborted; saved at \"{path}\"")
            raise e
            


    path = f"{MODEL_FOLDER}/othello_{VERSION}_{num_episodes//1000}k.pth"
    torch.save(agent.policyNet.state_dict(), path)
    print(f"Saved final version at \"{path}\"")

    #os.remove(CHECKPOINT_FOLDER)
    for file in os.listdir(CHECKPOINT_FOLDER):
        os.remove(os.path.join(CHECKPOINT_FOLDER,file))
        
        
