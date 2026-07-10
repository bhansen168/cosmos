import warnings,random,torch,sys,os
warnings.filterwarnings("ignore")

from torch import nn
from collections import deque
import torch.nn.functional as F
import numpy as np

sys.path.append(os.getcwd())
from game import Game



"""
from torch import multiprocessing


from collections import defaultdict

import matplotlib.pyplot as plt
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
"""
"""
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
"""


class QNet(nn.Module):
    def __init__(self,params=64,actions=64):
        self.layer1 = nn.Linear(in_features=params, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=10)
        self.layer4 = nn.Linear(in_features=10, out_features=actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
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
    def __init__(self,stateDim,actionDim,lr = 1e-3,gamma = 0.99):
        self.actionDim = actionDim
        self.gamma = gamma


        self.policyNet = QNet(stateDim,actionDim)
        self.targetNet = QNet(stateDim,actionDim)
        self.targetNet.load_state_dict(self.policyNet.state_dict())


        self.optimizer = torch.optim.Adam(self.policyNet.parameters,lr=lr)

    def select_action(self,state,epsilon):
        if random.random() < epsilon:
            return random.randint(0,self.actionDim-1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                return self.policyNet(state_t).argmax().item() #not sure how you can tack params onto a class
        

def optimize(agent,memory,batchSize):
    if len(memory) < batchSize:
        return

    states,actions,rewards,nextStates,dones = memory.sample(batchSize)

    currentQ = agent.policyNet(states).gather(1,actions.unsqeeze(1)).squeeze(1)

    with torch.no_grad():
        nextQ = agent.targetNet(nextStates).max(1)[0]
        targetQ = rewards + (agent.gamma * nextQ * (1-dones))

    loss = F.mse_loss(currentQ,targetQ)
    agent.optimizer.zero_grad()
    loss.backwards()#backpropagate?
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
        if roll < 0.10:
            return self.greedy_bot  # 10% chance to test against baseline
        elif roll < 0.40:
            return random.choice(self.past_versions)  # 30% chance to play older selves
        else:
            return "LATEST_SELF"  # 60% chance to play against its most recent self

def select_board_action(agent, state, legal_moves, epsilon):
    """
    legal_moves: binary mask tensor or list of indices, e.g., [0, 1, 0, 0, 1] 
                 where 1 means the move is legal.
    """
    if random.random() < epsilon:
        # Explore ONLY within legal spaces
        return random.choice([i for i, allowed in enumerate(legal_moves) if allowed == 1])
    else:
        # Exploit using masked action selection
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent.policy_net(state_t).squeeze(0)
            
            # Set illegal moves to a highly negative value (-infinity)
            masked_q = q_values.clone()
            illegal_indices = [i for i, allowed in enumerate(legal_moves) if allowed == 0]
            masked_q[illegal_indices] = -float('inf')
            
            return masked_q.argmax().item()


def index_to_coord(action_idx):
    y,x = divmod(action_idx,8)
    
    #y = action_idx // 8
    #x = action_idx % 8
    return y, x

def coord_to_index(y, x):
    return y * 8 + x

class OthelloEnv(Game):
    def __init__(self,side=8):
        super().__init__(side = side)
        self.current_player = 1 #player; 2 is player 2

        self.state_dim = 64
        self.action_dim = 64 #not really, but it makes it easier

    def _format_board(self):
        #formats board in favor of current player
        boardCopy = [[0 for _ in range(self.side)] for _ in range(self.side)]
        mapDict = {self.current_player:1,(1 if self.current_player==2 else 2):-1}
        for y in range(self.side):
            for x in range(self.side):
                boardCopy[y][x] = mapDict[self.board[y][x]]

        return boardCopy

        
            

    def reset(self):
        """
        Resets the board. 
        Returns:
            state: The initialized board array from the perspective of the starting player.
            info: A dictionary containing extra metadata (optional).
        """
        self.board = [[0 for _ in range(self.side)] for _ in range(self.side)]
        self._set_middle()

        return self._format_board(),{}

    def step(self, action):
        """
        Applies the chosen action to the board state.
        Returns:
            next_state: The new board array from the perspective of the NEXT active player.
            reward: 0.0 for mid-game turns. 
            done: True if a player wins, loses, draws, or the board is full.
            truncated: False (unless you use a hard step-limit turn counter).
            info: Extra metadata.
        """
        return self._format_board(),

    def get_legal_moves(self):
        """
        Returns:
            legal_moves: A binary mask (list or NumPy array of 0s and 1s) 
                         matching the length of action_dim. 
                         1 indicates a legal position, 0 indicates a blocked/invalid move.
        """
        pass

    def get_player_reward(self, player_id):
        """
        Evaluates the endgame state.
        Returns:
            reward (float): e.g., +1.0 if player_id won, -1.0 if they lost, 0.0 for a draw.
        """
        pass
        


if __name__ == "__main__":
    env = OthelloEnv()
    
    #training loop
    pool = OpponentPool(greedy_bot=MyGreedyRulesBot())
    num_episodes = 10000

    for episode in range(num_episodes):
        opponent_type = pool.select_opponent(episode, num_episodes)
        state, _ = env.reset()
        done = False
        
        while not done:
            current_player = env.current_player
            
            if current_player == agent.id:
                # Main Agent plays using epsilon-greedy & collects gradients
                action = agent.select_action(state, epsilon)
                next_state, reward, done, _, _ = env.step(action)
                # (Store in Replay Buffer...)
            else:
                # Opponent plays based on the selected pool strategy
                if opponent_type == "LATEST_SELF":
                    action = agent.select_action(state, epsilon=0.0) # Exploit self
                elif opponent_type == self.greedy_bot:
                    action = pool.greedy_bot.get_move(state, env.get_legal_moves())
                else:
                    # Load historical model weights temporarily for the turn
                    historical_agent.load_state_dict(opponent_type)
                    action = historical_agent.select_action(state, epsilon=0.0)
                    
                next_state, reward, done, _, _ = env.step(action)
                
        # Every 500 episodes, snapshot the agent and add it to the pool
        if episode % 500 == 0 and episode > 0:
            pool.add_checkpoint(agent.policy_net.state_dict())
        
