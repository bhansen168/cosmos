import warnings,random,torch,sys,os,re
warnings.filterwarnings("ignore")

from torch import nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
import numpy as np
from datetime import datetime,timedelta
import optuna

sys.path.append(os.getcwd())
from game import Game
from computer import Computer,create_genetic_computer

MUTE_PRINTS = True
EPOCHS = 1000
VERSION = "v02"

def predict_finish(start,amtCompleted):
    #start is datetime from start, amtCompleted is 0-1 decimal
    secsElapsed = (datetime.now()-start).total_seconds()
    totalSecs = int((1/amtCompleted)*secsElapsed)

    finish = datetime.now() + timedelta(seconds = totalSecs - secsElapsed)
    return str(finish).split(".")[0]
    


class QNet(nn.Module):
    def __init__(self,params=64,actions=64,hidden_size=128):
        super().__init__()
        
        self.layer1 = nn.Linear(in_features=params, out_features=hidden_size)
        self.layer2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.layer3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.layer4 = nn.Linear(in_features=hidden_size, out_features=actions)

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
    def __init__(self, stateDim, actionDim, lr=1e-3, gamma=0.95, hidden_size=128):
        self.actionDim = actionDim
        self.gamma = gamma
        
        self.policyNet = QNet(stateDim, actionDim, hidden_size)
        self.targetNet = QNet(stateDim, actionDim, hidden_size)
        self.targetNet.load_state_dict(self.policyNet.state_dict())

        self.id = 1 
        
        # FIX: Added () to .parameters() so PyTorch can register the weights
        self.optimizer = optim.Adam(self.policyNet.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

        
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

    def get_value_prediction(self, state, legal_moves):
        """Return the estimated value (max Q over legal moves) for the current position."""
        legal_moves = np.array(legal_moves)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policyNet(state_t).squeeze(0)
            
            # Mask illegal moves
            masked_q = q_values.clone()
            illegal_indices = np.where(legal_moves == 0)[0]
            masked_q[illegal_indices] = -float('inf')
            
            return masked_q.max().item()

def opponent_of(color):
    return Game.WHITE if color == Game.BLACK else Game.BLACK


# Terminal (end-of-game) score returned by the minimax instead of the DQN value.
# Added to the disc differential so a win/loss dominates any DQN leaf estimate
# (which lives in roughly [-1, 1]) and the search can always "see" the outcome.
DQN_MINIMAX_WIN_SCORE = 1.0


def dqn_minimax_value(agent, game, color, depth, alpha, beta, root_color):
    """Alpha-beta leaf evaluation using the DQN agent as the value function.

    Uses standard (max/min) minimax — identical in structure to GeneticPlayer,
    MinimaxPlayer, and DQNMinimaxPlayer (benchmark).  All leaves return a value
    from **root_color**'s perspective so callers never need to negate.

    Operates on a Game/OthelloEnv by applying/undoing moves in place, so the
    caller must ensure all play() calls are paired with undo().
    """
    legal_moves = game.legal_moves(color)
    other = opponent_of(color)

    if not legal_moves:
        other_moves = game.legal_moves(other)
        if not other_moves:
            scores = game.get_score()
            diff = scores[root_color] - scores[opponent_of(root_color)]
            if diff > 0:
                return DQN_MINIMAX_WIN_SCORE + diff
            if diff < 0:
                return -DQN_MINIMAX_WIN_SCORE + diff
            return 0.0
        if depth <= 0:
            # Pass at leaf: evaluate from the side that CAN move (other),
            # then adjust to root's perspective.
            legal_np = legal_moves_to_np_arr([(m.x, m.y) for m in other_moves], agent.actionDim)
            val = agent.get_value_prediction(encode_state(game.board, other), legal_np)
            return val if other == root_color else -val
        # Pass with depth remaining (no ply consumed, same alpha/beta).
        return dqn_minimax_value(agent, game, other, depth, alpha, beta, root_color)

    if depth <= 0:
        legal_np = legal_moves_to_np_arr([(m.x, m.y) for m in legal_moves], agent.actionDim)
        val = agent.get_value_prediction(encode_state(game.board, color), legal_np)
        return val if color == root_color else -val

    # ---- standard minimax: max for root_color, min for opponent ----
    if color == root_color:
        value = float("-inf")
        for move in legal_moves:
            game.play(color, move)
            child = dqn_minimax_value(agent, game, other, depth - 1, alpha, beta, root_color)
            game.undo(color, move)
            if child > value:
                value = child
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value

    value = float("inf")
    for move in legal_moves:
        game.play(color, move)
        child = dqn_minimax_value(agent, game, other, depth - 1, alpha, beta, root_color)
        game.undo(color, move)
        if child < value:
            value = child
        if value < beta:
            beta = value
        if alpha >= beta:
            break
    return value


def dqn_minimax_select_action(agent, game, color, depth=2, epsilon=0.0):
    """Select a move for `color` via alpha-beta minimax with the DQN as evaluator.

    Returns an action index (0-63) compatible with OthelloEnv.step, or None to
    pass. With probability `epsilon` a random legal move is chosen instead so
    the training loop retains exploration.
    """
    root_moves = game.legal_moves(color)
    if not root_moves:
        return None

    if random.random() < epsilon:
        mv = random.choice(root_moves)
        return coord_to_index(mv.y, mv.x)

    best_move = None
    best_value = -float("inf")
    alpha = -float("inf")
    for move in root_moves:
        game.play(color, move)
        value = dqn_minimax_value(agent, game, opponent_of(color), depth - 1, alpha, float("inf"), color)
        game.undo(color, move)
        if value > best_value:
            best_value = value
            best_move = move
        if value > alpha:
            alpha = value

    if best_move is None:
        return None
    return coord_to_index(best_move.y, best_move.x)

def optimize(agent,memory,batchSize):
    if len(memory) < batchSize:
        return

    states,actions,rewards,nextStates,dones = memory.sample(batchSize)

    currentQ = agent.policyNet(states).gather(1,actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN: use policy net to select action, target net to evaluate
        next_actions = agent.policyNet(nextStates).argmax(1)
        nextQ = agent.targetNet(nextStates).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targetQ = rewards + (agent.gamma * nextQ * (1-dones))

    loss = F.mse_loss(currentQ,targetQ)
    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policyNet.parameters(), max_norm=1.0)
    agent.optimizer.step()


class OpponentPool:
    def __init__(self):
        self.past_versions = []  # Stores saved state_dicts of your agent
        
    def add_checkpoint(self, agent_state_dict):
        # Save historical snapshots to prevent regression
        self.past_versions.append(agent_state_dict)
        
    def select_opponent(self, episode, total_episodes):
        # Fixed distribution: 50% genetic, 20% latest self, 20% historic self, 10% random
        roll = random.random()
        if roll < 0.50:
            return "GENETIC"
        elif roll < 0.70:
            return "LATEST_SELF"
        elif roll < 0.90:
            if self.past_versions:
                return random.choice(self.past_versions)
            return "LATEST_SELF"  # fallback if no history yet
        return "RANDOM"

def index_to_coord(action_idx):
    if isinstance(action_idx,tuple): #already formatted correctly
        return action_idx
    
    y,x = divmod(action_idx,8)
    
    #y = action_idx // 8
    #x = action_idx % 8
    return y, x

def coord_to_index(y, x):
    return y * 8 + x


def is_edge_move(action_idx):
    """True if action lands on an edge square (not a corner)."""
    y, x = index_to_coord(action_idx)
    on_border = (x == 0 or x == 7 or y == 0 or y == 7)
    on_corner = (x in (0, 7) and y in (0, 7))
    return on_border and not on_corner


def is_corner_move(action_idx):
    """True if action lands on a corner square."""
    y, x = index_to_coord(action_idx)
    return x in (0, 7) and y in (0, 7)


def move_reward(action_idx, edge_bonus, corner_bonus):
    """Intermediate reward for a single placement on edge/corner."""
    if action_idx is None:
        return 0.0
    r = 0.0
    if is_edge_move(action_idx):
        r += edge_bonus
    if is_corner_move(action_idx):
        r += corner_bonus
    return r



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

    def _count_opponent_square_access(self, color):
        """Count opponent's legal moves on edges/corners after current move."""
        opp = Game.WHITE if color == Game.BLACK else Game.BLACK
        legal = self.get_all_legal_moves(opp)
        edge_count = 0
        corner_count = 0
        for lx, ly in legal:
            if (lx == 0 or lx == 7) and (ly == 0 or ly == 7):
                corner_count += 1
            elif lx == 0 or lx == 7 or ly == 0 or ly == 7:
                edge_count += 1
        return edge_count, corner_count
        

        
            

    def reset(self):
        """
        Resets the board. 
        Returns:
            state: The initialized board array from the perspective of the starting player.
            info: A dictionary containing extra metadata (optional).
        """
        self.board = [[0 for _ in range(self.side)] for _ in range(self.side)]
        self.current_player = 1
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
            # Opponent can't move; switch back
            self.current_player = (Game.WHITE if self.current_player == Game.BLACK else Game.BLACK)
            # Game over only if NEITHER player can move
            if len(self.get_all_legal_moves(self.current_player)) == 0:
                gameOver = True

        if self.check_game_over():
            gameOver = True
        
        
        # No intermediate rewards — only terminal rewards for Othello.
        # Edge/corner shaping with γ=0.99 inflates Q-values across ~30 moves.
        reward = 0.0
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
        

    def get_player_reward(self, player_id, marginal_bonus=1.0):
        """
        Evaluates the endgame state.
        Base reward is always ±1 for win/loss.  `marginal_bonus` scales the
        extra margin component so that 64-0 earns more than 33-31.
        """
        scores = self.get_score()
        score = scores[player_id]
        if score > 32:
            return 1.0 + marginal_bonus * (score - 32) / 32
        if score < 32:
            return -(1.0 + marginal_bonus * (32 - score) / 32)
        return 0.0

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
    try:
        weights = torch.load(file,map_location="cpu",weights_only=True)
    except TypeError:
        weights = torch.load(file,map_location="cpu")

    try:
        hidden_size = int(weights["layer1.weight"].shape[0])
        input_size = int(weights["layer1.weight"].shape[1])
        output_size = int(weights["layer4.weight"].shape[0])
    except (KeyError,TypeError,IndexError,AttributeError) as exc:
        raise ValueError(f"Unrecognized DQN checkpoint: {file}") from exc
    if input_size != 64 or output_size != 64:
        raise ValueError(f"DQN checkpoint must use 64 inputs and outputs: {file}")

    env = OthelloEnv()
    trained_agent = Agent(env.state_dim,env.action_dim,hidden_size=hidden_size)

    trained_agent.policyNet.load_state_dict(weights)

    # 3. CRITICAL: Switch the network to evaluation mode 
    # This locks gradients and sets up layers properly for pure inference gameplay
    trained_agent.policyNet.eval()

    return trained_agent


def find_latest_checkpoint(checkpoint_folder, model_folder, version=VERSION):
    """Return (path, episode) for the most recently modified checkpoint across the
    regular *-sav.pth files (in `checkpoint_folder`) and the *-ABORTED.pth files
    (in `model_folder`), or (None, 0) if none exist. Used to resume training by
    default so progress is never lost, even after an interrupt."""
    candidates = []  # (path, episode)
    if os.path.isdir(checkpoint_folder):
        sav_pat = re.compile(rf"othello_{re.escape(version)}_(\d+(?:\.\d+)?)k-sav\.pth$")
        for fname in os.listdir(checkpoint_folder):
            match = sav_pat.match(fname)
            if match:
                candidates.append(
                    (os.path.join(checkpoint_folder, fname), int(round(float(match.group(1)) * 1000)))
                )
    if os.path.isdir(model_folder):
        ab_pat = re.compile(rf"othello_{re.escape(version)}_(\d+(?:\.\d+)?)k_ABORTED\.pth$")
        for fname in os.listdir(model_folder):
            match = ab_pat.match(fname)
            if match:
                candidates.append(
                    (os.path.join(model_folder, fname), int(round(float(match.group(1)) * 1000)))
                )
    if not candidates:
        return None, 0
    latest_path, latest_ep = max(candidates, key=lambda item: os.path.getmtime(item[0]))
    return latest_path, latest_ep


def train_with_params(params, n_episodes=500, verbose=False):
    """Train a DQN agent with the given hyperparameters.

    Returns the average terminal reward over the last 100 episodes.
    """
    env = OthelloEnv()
    agent = Agent(env.state_dim, env.action_dim, lr=params['lr'], gamma=params['gamma'])
    memory = ReplayBuffer(capacity=20000)
    pool = OpponentPool()

    genetic_path = os.getcwd() + "/models/genetic_gen_0049_v2.json"
    genetic_bot = None
    if os.path.isfile(genetic_path):
        genetic_bot = create_genetic_computer(env, Game.BLACK, genetic_path, search_depth=1)

    historical_agent = Agent(env.state_dim, env.action_dim)

    edge_bonus = params['edge_bonus']
    corner_bonus = params['corner_bonus']
    marginal_bonus = params['marginal_bonus']

    epsilon_decay = 0.9995
    min_epsilon = 0.01
    epsilon = 1.0

    recent_terminal_rewards = deque(maxlen=100)

    for episode in range(n_episodes):
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        opponent_type = pool.select_opponent(episode, n_episodes)
        state, _ = env.reset()
        done = False
        opp_penalty = 0.0

        while not done:
            current_player = env.current_player

            if current_player != agent.id:
                # Opponent plays
                if opponent_type == "LATEST_SELF":
                    opp_state = encode_state(env.board, env.current_player)
                    action = agent.select_action(opp_state, env.get_legal_moves(), epsilon)
                elif opponent_type == "GENETIC":
                    if genetic_bot is not None:
                        xy = genetic_bot.pick_minimax(color=current_player, place=False)
                        action = (xy[1], xy[0])
                    else:
                        opp_state = encode_state(env.board, env.current_player)
                        action = agent.select_action(opp_state, env.get_legal_moves(), epsilon=0.0)
                elif opponent_type == "RANDOM":
                    legal = env.get_legal_moves()
                    valid = np.where(legal == 1)[0]
                    action = random.choice(valid) if len(valid) > 0 else None
                else:
                    historical_agent.policyNet.load_state_dict(opponent_type)
                    opp_state = encode_state(env.board, env.current_player)
                    action = historical_agent.select_action(opp_state, env.get_legal_moves(), epsilon=0.0)

                next_state, _, done, _, _ = env.step(action)
                state = next_state

                if action is not None:
                    opp_penalty -= move_reward(action, edge_bonus, corner_bonus)

            if done:
                reward = env.get_player_reward(agent.id, marginal_bonus)

            # Main Agent
            try:
                if len(env.get_all_legal_moves(env.current_player)) == 0:
                    env.current_player = Game.WHITE if env.current_player == Game.BLACK else Game.BLACK
                    if len(env.get_all_legal_moves(env.current_player)) == 0:
                        done = True
                    state = env._flatten()
                    if done:
                        reward = env.get_player_reward(agent.id, marginal_bonus)
                    continue

                action = agent.select_action(state, env.get_legal_moves(), epsilon)
                next_state, _, done, _, _ = env.step(action)
                next_state = encode_state(env.board, agent.id)

                if done:
                    reward = env.get_player_reward(agent.id, marginal_bonus)
                else:
                    reward = opp_penalty + move_reward(action, edge_bonus, corner_bonus)
                    opp_penalty = 0.0

                memory.push(state, action, reward, next_state, done)
                state = next_state
            except IndexError:
                pass

        # Track terminal reward for evaluation
        terminal_reward = env.get_player_reward(agent.id, marginal_bonus)
        recent_terminal_rewards.append(terminal_reward)

        optimize(agent, memory, 64)
        agent.scheduler.step()

        # Polyak averaging
        tau = 0.005
        for target_param, policy_param in zip(agent.targetNet.parameters(), agent.policyNet.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

        if episode > 0 and episode % 500 == 0:
            pool.add_checkpoint(agent.policyNet.state_dict())

        if verbose and episode % 50 == 0:
            avg = np.mean(recent_terminal_rewards) if recent_terminal_rewards else 0
            print(f"  Ep {episode}: avg_reward={avg:.3f}  eps={epsilon:.3f}")

    return float(np.mean(recent_terminal_rewards)) if recent_terminal_rewards else 0.0


def evaluate(agent, env, n_games=100):
    """Play n_games vs random opponent with greedy policy. Returns (wins, losses, draws)."""
    wins = losses = draws = 0
    for _ in range(n_games):
        state, _ = env.reset()
        done = False
        while not done:
            if env.current_player == agent.id:
                legal = env.get_legal_moves()
                if len(np.where(legal == 1)[0]) == 0:
                    env.current_player = Game.WHITE if env.current_player == Game.BLACK else Game.BLACK
                    if len(env.get_all_legal_moves(env.current_player)) == 0:
                        done = True
                    continue
                action = agent.select_action(state, legal, epsilon=0.0)
                next_state, _, done, _, _ = env.step(action)
                state = encode_state(env.board, agent.id)
            else:
                legal = env.get_legal_moves()
                valid = np.where(legal == 1)[0]
                if len(valid) == 0:
                    env.current_player = Game.WHITE if env.current_player == Game.BLACK else Game.BLACK
                    if len(env.get_all_legal_moves(env.current_player)) == 0:
                        done = True
                    continue
                action = random.choice(valid)
                next_state, _, done, _, _ = env.step(action)
                state = encode_state(env.board, agent.id)
        scores = env.get_score()
        if scores[agent.id] > scores[1 if agent.id == 2 else 2]:
            wins += 1
        elif scores[agent.id] < scores[1 if agent.id == 2 else 2]:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if "--optimize" in sys.argv:
        # ── Hyperparameter optimization ──────────────────────────────────
        EPIS_PER_TRIAL = 300
        N_TRIALS = 30

        def objective(trial):
            params = {
                'gamma':           trial.suggest_float('gamma', 0.90, 0.99),
                'lr':              trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                'marginal_bonus':  trial.suggest_float('marginal_bonus', 0.0, 2.0),
                'edge_bonus':      trial.suggest_float('edge_bonus', 0.0, 0.5),
                'corner_bonus':    trial.suggest_float('corner_bonus', 0.0, 1.0),
            }
            try:
                score = train_with_params(params, n_episodes=EPIS_PER_TRIAL, verbose=True)
            except KeyboardInterrupt:
                raise
            return score

        def progress_callback(study, trial):
            best = study.best_trial.value if study.best_trial else None
            best_str = f"{best:.4f}" if best is not None else "n/a"
            print(f"  Trial {trial.number+1:>3}/{N_TRIALS}  "
                  f"score={trial.value:.4f}  "
                  f"best={best_str}  "
                  f"params={trial.params}")

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[progress_callback])

        print("\n═══════════════════════════════════════════")
        print("  BEST TRIAL")
        print("═══════════════════════════════════════════")
        print(f"  Value (avg terminal reward): {study.best_trial.value:.4f}")
        for k, v in study.best_trial.params.items():
            print(f"  {k:20s}: {v}")
        print("═══════════════════════════════════════════")
    else:
        # ── Normal training (with edge/corner & marginal bonuses) ────────
        params = {
            'gamma':          0.9,
            'lr':             0.0005,
            'marginal_bonus': 0.2936,
            'edge_bonus':     0.03,
            'corner_bonus':   0.08,
        }

        env = OthelloEnv()
        CHECKPOINT_FOLDER = os.getcwd() + "/models/checkpoints"
        MODEL_FOLDER = os.getcwd() + "/models"

        memory = ReplayBuffer(capacity=20000)
        batch_size = 64

        pool = OpponentPool()
        genetic_path = os.getcwd() + "/models/genetic_gen_0049_v2.json"
        if os.path.isfile(genetic_path):
            genetic_bot = create_genetic_computer(env, Game.BLACK, genetic_path, search_depth=1)
        else:
            print(f"Genetic checkpoint not found at {genetic_path}; disabling GENETIC opponent")
            genetic_bot = None
        agent = Agent(env.state_dim, env.action_dim, lr=params['lr'], gamma=params['gamma'])
        historical_agent = Agent(env.state_dim, env.action_dim)

        edge_bonus = params['edge_bonus']
        corner_bonus = params['corner_bonus']
        marginal_bonus = params['marginal_bonus']

        num_episodes = EPOCHS

        RESUME_FROM_CHECKPOINT = True
        resume_path, start_episode = (None, 0)
        if RESUME_FROM_CHECKPOINT:
            resume_path, start_episode = find_latest_checkpoint(CHECKPOINT_FOLDER, MODEL_FOLDER)
        if resume_path is not None:
            print(f"Resuming training from {resume_path} (episode {start_episode})")
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
            agent.policyNet.load_state_dict(ckpt)
            agent.targetNet.load_state_dict(ckpt)
            agent.scheduler.step(start_episode)
        else:
            print("No checkpoint found; starting training from scratch")

        total_episodes = start_episode + num_episodes

        epsilon_decay = 0.9995
        min_epsilon = 0.01
        epsilon = max(1.0 * (epsilon_decay ** start_episode), min_epsilon)

        UPDATE = 100
        SAV_FREQ = max(min(int(EPOCHS * 0.05), 20000), 1000)

        start = datetime.now()
        print("Started training at " + str(start).split(".")[0])

        if not os.path.exists(CHECKPOINT_FOLDER):
            os.mkdir(CHECKPOINT_FOLDER)

        for episode in range(start_episode, total_episodes):
            try:
                epsilon = max(epsilon * epsilon_decay, min_epsilon)
                opponent_type = pool.select_opponent(episode, total_episodes)
                state, _ = env.reset()
                done = False
                opp_penalty = 0.0

                while not done:
                    current_player = env.current_player
                    if not MUTE_PRINTS:
                        input("PRESS ENTER FOR NEXT TURN: ")

                    if current_player != agent.id:
                        # Opponent plays
                        if opponent_type == "LATEST_SELF":
                            if not MUTE_PRINTS:
                                print("opponent (self) to move")
                            opp_state = encode_state(env.board, env.current_player)
                            action = agent.select_action(opp_state, env.get_legal_moves(), epsilon)
                        elif opponent_type == "GENETIC":
                            if genetic_bot is not None:
                                if not MUTE_PRINTS:
                                    print("opponent (genetic) to move")
                                xy = genetic_bot.pick_minimax(color=current_player, place=False)
                                action = (xy[1], xy[0])
                            else:
                                if not MUTE_PRINTS:
                                    print("opponent (self, greedy) to move")
                                opp_state = encode_state(env.board, env.current_player)
                                action = agent.select_action(opp_state, env.get_legal_moves(), epsilon=0.0)
                        elif opponent_type == "RANDOM":
                            if not MUTE_PRINTS:
                                print("opponent (random) to move")
                            legal = env.get_legal_moves()
                            valid = np.where(legal == 1)[0]
                            action = random.choice(valid) if len(valid) > 0 else None
                        else:
                            if not MUTE_PRINTS:
                                print("opponent (historical) to move")
                            historical_agent.policyNet.load_state_dict(opponent_type)
                            opp_state = encode_state(env.board, env.current_player)
                            action = historical_agent.select_action(opp_state, env.get_legal_moves(), epsilon=0.0)

                        next_state, _, done, _, _ = env.step(action)
                        state = next_state

                        if action is not None:
                            opp_penalty -= move_reward(action, edge_bonus, corner_bonus)

                    if done:
                        reward = env.get_player_reward(agent.id, marginal_bonus)

                    # Main Agent
                    try:
                        if len(env.get_all_legal_moves(env.current_player)) == 0:
                            env.current_player = Game.WHITE if env.current_player == Game.BLACK else Game.BLACK
                            if len(env.get_all_legal_moves(env.current_player)) == 0:
                                done = True
                            state = env._flatten()
                            if done:
                                reward = env.get_player_reward(agent.id, marginal_bonus)
                            continue

                        action = agent.select_action(state, env.get_legal_moves(), epsilon)
                        next_state, _, done, _, _ = env.step(action)
                        next_state = encode_state(env.board, agent.id)

                        if done:
                            reward = env.get_player_reward(agent.id, marginal_bonus)
                        else:
                            reward = opp_penalty + move_reward(action, edge_bonus, corner_bonus)
                            opp_penalty = 0.0

                        memory.push(state, action, reward, next_state, done)
                        state = next_state
                    except IndexError:
                        pass

                optimize(agent, memory, batch_size)
                agent.scheduler.step()

                tau = 0.005
                for target_param, policy_param in zip(agent.targetNet.parameters(), agent.policyNet.parameters()):
                    target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

                if episode > 0:
                    if episode % 500 == 0:
                        pool.add_checkpoint(agent.policyNet.state_dict())
                    if episode % SAV_FREQ == 0:
                        path = f"{CHECKPOINT_FOLDER}/othello_{VERSION}_{round(episode/1000,1)}k-sav.pth"
                        torch.save(agent.policyNet.state_dict(), path)
                        print(f"Saved checkpoint at \"{path}\"; timestamp: {str(datetime.now()).split('.')[0]}")
                    if (episode % UPDATE == UPDATE - 1):
                        perc = (episode + 1) / total_episodes
                        print(f"FINISHED EPISODE {episode+1} OF {total_episodes} -- {round(perc * 100,2)}% -- ends at {predict_finish(start, perc)}")
                    if episode % 1000 == 0:
                        w, l, d = evaluate(agent, env, n_games=100)
                        print(f"EVAL vs random: {w}/{l}/{d}  win_rate={100*w/(w+l or 1):.1f}%")
            except KeyboardInterrupt as e:
                path = f"{MODEL_FOLDER}/othello_{VERSION}_{episode//1000}k_ABORTED.pth"
                torch.save(agent.policyNet.state_dict(), path)
                print(f"Aborted; saved at \"{path}\"")
                raise e

        path = f"{MODEL_FOLDER}/othello_{VERSION}_{total_episodes//1000}k.pth"
        torch.save(agent.policyNet.state_dict(), path)
        print(f"Saved final version at \"{path}\"")

        for file in os.listdir(CHECKPOINT_FOLDER):
            os.remove(os.path.join(CHECKPOINT_FOLDER, file))
        
        
