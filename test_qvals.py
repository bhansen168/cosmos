import torch
import sys
sys.path.append('.')
from computerRL import encode_state, legal_moves_to_np_arr, coord_to_index
import numpy as np

class QNet(torch.nn.Module):
    def __init__(self, params=64, actions=64, hidden=10):
        super().__init__()
        self.layer1 = torch.nn.Linear(params, hidden)
        self.layer2 = torch.nn.Linear(hidden, hidden)
        self.layer3 = torch.nn.Linear(hidden, hidden)
        self.layer4 = torch.nn.Linear(hidden, actions)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return self.layer4(x)

for model_name in ['othello_v02_1k.pth', 'othello_v02_2k.pth', 'othello_v02_5k.pth', 'othello_v02_10k.pth', 'othello_v02_70_ABORTED.pth']:
    try:
        model = QNet(64, 64, hidden=10)
        weights = torch.load(f'models/{model_name}', map_location='cpu')
        model.load_state_dict(weights)
        model.eval()

        board = [[0]*8 for _ in range(8)]
        board[3][3] = 2
        board[4][4] = 2
        board[3][4] = 1
        board[4][3] = 1

        state = encode_state(board, 1)
        legal = [(3,2), (2,3), (4,5), (5,4)]
        legal_arr = legal_moves_to_np_arr(legal, 64)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_t).squeeze(0)
            
            print(f'{model_name}:')
            print(f'  All Q range: {q_values.min().item():.2f} to {q_values.max().item():.2f}')
            legal_qs = [q_values[coord_to_index(my, mx)].item() for mx, my in legal]
            print(f'  Legal Qs: {[f"{q:.2f}" for q in legal_qs]}')
    except Exception as e:
        print(f'{model_name}: ERROR - {e}')