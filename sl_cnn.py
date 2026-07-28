import os
import sys
import pickle
import numpy as np
from datetime import datetime,timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())

def predict_finish(start,amtCompleted):
    #start is datetime from start, amtCompleted is 0-1 decimal
    secsElapsed = (datetime.now()-start).total_seconds()
    totalSecs = int((1/amtCompleted)*secsElapsed)

    finish = datetime.now() + timedelta(seconds = totalSecs - secsElapsed)
    return str(finish).split(".")[0]


# -------------------------------------------------------------
# 1. SPATIAL STATE ENCODING & MAPPING
# -------------------------------------------------------------
def encode_state_spatial(board, active_player):
    """
    Encodes the board into a 2-channel 8x8 spatial tensor image:
    Channel 0: 1s where the active player has pieces, 0 elsewhere.
    Channel 1: 1s where the opponent has pieces, 0 elsewhere.
    """
    board_np = np.asarray(board, dtype=np.int8).reshape(8, 8)
    #board_np = np.array(board, dtype=np.int8)
    opponent = 1 if active_player == 2 else 2
    
    player_channel = (board_np == active_player).astype(np.float32)
    opponent_channel = (board_np == opponent).astype(np.float32)
    
    # Shape: (2, 8, 8) - Perfect input matrix for a 2D CNN
    out =  np.stack([player_channel, opponent_channel], axis=0)

    #print("TESTshp",out.shape)

    return out

def index_to_xgb(idx):
    """Maps a 0-63 board index to a dense 0-59 index by stripping center gaps."""
    ct = 0
    missing = [27, 28, 35, 36]
    for val in missing:
        if idx > val:
            ct += 1
    return idx - ct

# Reverse dictionary map for inference decoding
XGB_DICT = {index_to_xgb(i): i for i in (list(range(27)) + list(range(29, 35)) + list(range(37, 65)))}

# -------------------------------------------------------------
# 2. PYTORCH DATASET PIPELINE
# -------------------------------------------------------------
class OthelloDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------------------------------------------
# 3. SPATIAL GEOMETRY CONVOLUTIONAL NETWORK
# -------------------------------------------------------------
class OthelloCNN(nn.Module):
    def __init__(self):
        super(OthelloCNN, self).__init__()
        # Conv block 1: Looks at immediate neighbors (3x3 kernel with padding)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Conv block 2: Chains neighbor dependencies to see patterns further out
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Conv block 3: Deep spatial feature consolidation
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.relu = nn.ReLU()
        
        # Fully Connected policy head outputs exactly 60 logits
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 60)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten spatial feature grid to vector
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)            # Outputs raw unnormalized scores (logits)
        return x

# -------------------------------------------------------------
# 4. DATA LOADER & TRAINING MANAGER
# -------------------------------------------------------------
class CompSupervisedCNN:
    FORMATTED = os.path.join(os.getcwd(), "data-formatted-sup")

    def __init__(self):
        self.games = []
        # Automatically unpack preexisting formatted games from your pipeline path
        for file in os.listdir(self.FORMATTED):
            if file.endswith(".fmtd"):
                path = os.path.join(self.FORMATTED, file)
                with open(path, "rb") as fRef:
                    self.games.extend(pickle.load(fRef))
        print(f"Loaded {len(self.games):,} formatted games ready for training.")

    def train(self, savePath="models/cnn_model.pth", epochs=20, batch_size=256):
        X_list = []
        y_list = []
        
        print("Extracting spatial structures and flipping active player perspectives...")
        for game in self.games:
            for board_state, action, active_player in game:
                if action == 64:  # Drop passes to keep classes clean
                    continue
                X_list.append(encode_state_spatial(board_state, active_player))
                y_list.append(index_to_xgb(action))
                
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int16)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        
        train_loader = DataLoader(OthelloDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(OthelloDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        
        # Hardware acceleration check (CUDA/MPS support)
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"Running execution pipeline natively on hardware target: {device}")
        
        model = OthelloCNN().to(device)
        criterion = nn.CrossEntropyLoss()  # Automatically computes stable multi-class log loss
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        startDT = datetime.now()
        print(f"Commencing Deep CNN Optimization Run at {str(startDT).split('.')[0]}...")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
            # Validation Step
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
            epoch_train_loss = train_loss / len(X_train)
            epoch_val_loss = val_loss / len(X_val)
            val_acc = (correct / total) * 100


            finish = predict_finish(startDT,(epoch+1)/epochs)
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train loss (mlogloss): {epoch_train_loss:.4f} | Val loss: {epoch_val_loss:.4f} | Val Accuracy: {val_acc:.2f}% | Elapsed: {str(datetime.now() - startDT).split('.')[0]} | Finish at: {str(finish).split('.')[0]}")
            
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        torch.save(model.state_dict(), savePath)
        print(f"Deep network weights successfully saved to: {savePath}")

# -------------------------------------------------------------
# 5. HIGH-SPEED AGENT INFERENCE INTERFACE
# -------------------------------------------------------------
class CNNAgent:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = OthelloCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def pick_xgb(self, legal_moves, board_state, active_player):
        if not legal_moves:
            return 64
            
        spatial_board = encode_state_spatial(board_state, active_player)
        # Convert to a single-batch tensor: Shape (1, 2, 8, 8)
        input_tensor = torch.tensor(spatial_board, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            # Apply Softmax to match multi:softprob behavior
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
        best_move = None
        best_prob = -1
        
        for move in legal_moves:
            if move == 64:
                continue
            xgb_slot = index_to_xgb(move)
            prob = probabilities[xgb_slot]
            
            if prob > best_prob:
                best_prob = prob
                best_move = move
                
        return best_move if best_move is not None else 64

    def pick(self, legal, board, active_player, asCoord=True):
        sel = self.pick_xgb(legal, board, active_player)
        if sel == 64:
            return None
        if asCoord:
            y, x = divmod(sel, 8)
            return (x, y)
        return sel

if __name__ == "__main__":
    trainer = CompSupervisedCNN()
    trainer.train(epochs=25, batch_size=64)
    #trainer.train(epochs=15, batch_size=256)
