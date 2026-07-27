import torch
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from computerRL import QNet

CHECKPOINT_DIR = "models/checkpoints"

def load_checkpoint(path):
    """Load a checkpoint and return the state dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Handle both raw state_dict and wrapped checkpoints
    if "policy_state_dict" in ckpt:
        return ckpt["policy_state_dict"]
    elif "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt

def analyze_weights():
    """Compare weight statistics across checkpoints."""
    checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')])
    print(f"{'='*80}")
    print(f"WEIGHT ANALYSIS ACROSS {len(checkpoints)} CHECKPOINTS")
    print(f"{'='*80}")
    
    prev_sd = None
    prev_name = None
    
    for fname in checkpoints:
        path = os.path.join(CHECKPOINT_DIR, fname)
        sd = load_checkpoint(path)
        
        # Extract episode count from filename
        import re
        m = re.search(r'v02_(\d+(?:\.\d+)?)k', fname)
        ep_label = m.group(1) if m else fname
        
        print(f"\n--- {fname} (ep ~{ep_label}k) ---")
        
        # Analyze each layer
        layer_names = [k for k in sd.keys() if 'weight' in k or 'bias' in k]
        for name in layer_names:
            w = sd[name].numpy()
            print(f"  {name:40s}  shape={str(list(w.shape)):20s}  "
                  f"mean={w.mean():+.6f}  std={w.std():.6f}  "
                  f"min={w.min():+.6f}  max={w.max():+.6f}  "
                  f"norm={np.linalg.norm(w):.4f}")
        
        # Compute weight change from previous checkpoint
        if prev_sd is not None:
            total_delta = 0.0
            total_norm = 0.0
            for key in sd:
                if key in prev_sd:
                    delta = (sd[key] - prev_sd[key]).numpy()
                    total_delta += np.abs(delta).mean()
                    total_norm += np.linalg.norm(delta)
            print(f"  >> Mean absolute change from {prev_name}: {total_delta:.6f}")
            print(f"  >> Total L2 change: {total_norm:.6f}")
        
        prev_sd = sd
        prev_name = fname

def analyze_q_values():
    """Analyze Q-value distributions for different board states."""
    from othello_engine import HeadlessOthello, BLACK, WHITE, EMPTY
    
    checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')])
    
    print(f"\n{'='*80}")
    print(f"Q-VALUE ANALYSIS")
    print(f"{'='*80}")
    
    for fname in checkpoints:
        path = os.path.join(CHECKPOINT_DIR, fname)
        sd = load_checkpoint(path)
        
        # Reconstruct the network
        net = QNet(64, 64, 128)
        net.load_state_dict(sd)
        net.eval()
        
        import re
        m = re.search(r'v02_(\d+(?:\.\d+)?)k', fname)
        ep_label = m.group(1) if m else fname
        
        print(f"\n--- {fname} (ep ~{ep_label}k) ---")
        
        # Test on empty board (opening)
        env = HeadlessOthello()
        state = encode_state(env.board, BLACK)
        with torch.no_grad():
            q = net(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
        legal = moves_to_mask(env.legal_moves(BLACK))
        legal_q = q[legal == 1]
        print(f"  Opening (BLACK to move):  "
              f"Q-range=[{legal_q.min():.4f}, {legal_q.max():.4f}]  "
              f"mean={legal_q.mean():.4f}  std={legal_q.std():.4f}")
        
        # Best opening move
        best_idx = np.argmax(legal_q)
        best_y, best_x = divmod(best_idx, 8)
        print(f"  Best opening move: ({best_x},{best_y}) Q={legal_q.max():.4f}")
        
        # Play a few random moves and analyze Q-values at different stages
        env2 = HeadlessOthello()
        for turn in range(4):  # 4 half-moves
            color = BLACK if turn % 2 == 0 else WHITE
            moves = env2.legal_moves(color)
            if not moves:
                break
            # Pick the first legal move
            env2.play(color, moves[0])
        
        # Now analyze from this position
        color = BLACK
        state2 = encode_state(env2.board, color)
        with torch.no_grad():
            q2 = net(torch.FloatTensor(state2).unsqueeze(0)).squeeze(0).numpy()
        legal2 = moves_to_mask(env2.legal_moves(color))
        legal_q2 = q2[legal2 == 1]
        print(f"  Mid-game (turn ~8):      "
              f"Q-range=[{legal_q2.min():.4f}, {legal_q2.max():.4f}]  "
              f"mean={legal_q2.mean():.4f}  std={legal_q2.std():.4f}")
        
        # Test on a corner-focused position
        env3 = HeadlessOthello()
        # Simulate a corner capture
        moves3 = env3.legal_moves(BLACK)
        # Find a move that captures toward corner
        for m in moves3:
            if m.x == 2 and m.y == 3:  # d5
                env3.play(BLACK, m)
                break
        moves_w = env3.legal_moves(WHITE)
        if moves_w:
            env3.play(WHITE, moves_w[0])
        moves_b = env3.legal_moves(BLACK)
        if moves_b:
            env3.play(BLACK, moves_b[0])
        color = BLACK
        state3 = encode_state(env3.board, color)
        with torch.no_grad():
            q3 = net(torch.FloatTensor(state3).unsqueeze(0)).squeeze(0).numpy()
        legal3 = moves_to_mask(env3.legal_moves(color))
        if legal3.sum() > 0:
            legal_q3 = q3[legal3 == 1]
            print(f"  Corner pos:              "
                  f"Q-range=[{legal_q3.min():.4f}, {legal_q3.max():.4f}]  "
                  f"mean={legal_q3.mean():.4f}  std={legal_q3.std():.4f}")

def analyze_positional_bias():
    """See which board positions the network values most."""
    from othello_engine import HeadlessOthello, BLACK, WHITE
    
    checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')])
    
    print(f"\n{'='*80}")
    print(f"POSITIONAL BIAS (average Q-value per position across opening moves)")
    print(f"{'='*80}")
    
    for fname in checkpoints:
        path = os.path.join(CHECKPOINT_DIR, fname)
        sd = load_checkpoint(path)
        net = QNet(64, 64, 128)
        net.load_state_dict(sd)
        net.eval()
        
        import re
        m = re.search(r'v02_(\d+(?:\.\d+)?)k', fname)
        ep_label = m.group(1) if m else fname
        
        print(f"\n--- {fname} (ep ~{ep_label}k) ---")
        
        # Average Q-values for each position from opening state
        env = HeadlessOthello()
        state = encode_state(env.board, BLACK)
        with torch.no_grad():
            q = net(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
        
        # Print as 8x8 grid
        q_grid = q.reshape(8, 8)
        for row in range(8):
            vals = [f"{q_grid[row, col]:+7.3f}" for col in range(8)]
            print(f"  {' '.join(vals)}")
        
        # Highlight top 5 positions
        legal = moves_to_mask(env.legal_moves(BLACK))
        legal_indices = np.where(legal == 1)[0]
        legal_q = q[legal_indices]
        top5 = legal_indices[np.argsort(legal_q)[-5:]][::-1]
        print(f"  Top 5 positions: ", end="")
        for idx in top5:
            y, x = divmod(idx, 8)
            print(f"({x},{y})={q[idx]:+.4f} ", end="")
        print()

def analyze_weight_evolution():
    """Track how specific weights evolve across training."""
    from othello_engine import HeadlessOthello, BLACK
    
    checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')])
    
    print(f"\n{'='*80}")
    print(f"WEIGHT EVOLUTION (input layer weight norms per position)")
    print(f"{'='*80}")
    
    all_input_weights = []
    labels = []
    
    for fname in checkpoints:
        path = os.path.join(CHECKPOINT_DIR, fname)
        sd = load_checkpoint(path)
        
        import re
        m = re.search(r'v02_(\d+(?:\.\d+)?)k', fname)
        ep_label = m.group(1) if m else fname
        labels.append(ep_label)
        
        # Get first layer weights (input to hidden)
        w1 = sd['layer1.weight'].numpy()  # shape: [128, 64]
        all_input_weights.append(w1)
    
    if len(all_input_weights) < 2:
        return
    
    # Compute per-position weight norms across checkpoints
    print(f"\nPer-position L2 norm across training:")
    for pos in range(64):
        y, x = divmod(pos, 8)
        norms = [np.linalg.norm(w[:, pos]) for w in all_input_weights]
        delta = norms[-1] - norms[0]
        print(f"  ({x},{y}): {' -> '.join(f'{n:.3f}' for n in norms)}  (delta={delta:+.3f})")
    
    # Weight similarity between consecutive checkpoints
    print(f"\nCosine similarity between consecutive checkpoints:")
    for i in range(1, len(all_input_weights)):
        w_prev = all_input_weights[i-1].flatten()
        w_curr = all_input_weights[i].flatten()
        cos_sim = np.dot(w_prev, w_curr) / (np.linalg.norm(w_prev) * np.linalg.norm(w_curr))
        print(f"  {labels[i-1]}k -> {labels[i]}k: {cos_sim:.6f}")

def encode_state(board, current_player):
    """Encode board state for the network."""
    boardCopy = [[0 for _ in range(8)] for _ in range(8)]
    mapDict = {current_player: 1, (1 if current_player == 2 else 2): -1}
    for y in range(8):
        for x in range(8):
            if board[y][x] != 0:
                boardCopy[y][x] = mapDict[board[y][x]]
    return np.array(boardCopy).flatten().astype(np.float32)

def moves_to_mask(moves, dim=64):
    """Convert list of LegalMove objects to binary mask."""
    mask = np.zeros(dim)
    for move in moves:
        mask[move.y * 8 + move.x] = 1
    return mask

if __name__ == "__main__":
    analyze_weights()
    analyze_q_values()
    analyze_positional_bias()
    analyze_weight_evolution()
