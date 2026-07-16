import sys
sys.path.append('.')
from computerRL import OthelloEnv
from game import Game

# Test reward calculation
env = OthelloEnv()

print("=== Testing reward calculation ===\n")

# Test: Play a few moves and check rewards
env.reset()
state = env._flatten()

for i in range(8):
    legal = env.get_all_legal_moves(env.current_player)
    print(f"\nMove {i+1}: Player {env.current_player}, legal: {legal}")
    
    for mx, my in legal:
        edge_r = (1 if (mx == 0 or mx == 7 or my == 0 or my == 7) else 0) * 0.02
        corner_r = (1 if ((mx == 0 or mx == 7) and (my == 0 or my == 7)) else 0) * 0.04
        
        # Check opponent penalty after this move
        env_test = OthelloEnv()
        env_test.board = [row[:] for row in env.board]
        env_test.current_player = env.current_player
        env_test.place_piece(env.current_player, mx, my)
        opp_edge, opp_corner = env_test._count_opponent_square_access(env.current_player)
        opp_pen = opp_edge * 0.02 + opp_corner * 0.04
        
        total = edge_r + corner_r - opp_pen
        print(f"  ({mx},{my}): edge={edge_r:.3f}, corner={corner_r:.3f}, opp_pen={opp_pen:.3f} -> total={total:.3f}")
    
    # Make first legal move
    if legal:
        mx, my = legal[0]
        action = my * 8 + mx  # coord_to_index(y, x)
        state, reward, done, trunc, _ = env.step(action)
        print(f"  Actual reward: {reward:.3f}")
        if done:
            break

print("\n=== Testing specific scenarios ===")

# Scenario: Force a position where taking corner gives opponent access
env = OthelloEnv()
# Set up a position: black has piece at (2,2), white at (1,2) and (2,1) - black can take corner (1,1)
env.board = [[0]*8 for _ in range(8)]
env.board[3][3] = Game.WHITE
env.board[4][4] = Game.WHITE
env.board[3][4] = Game.BLACK
env.board[4][3] = Game.BLACK
env.current_player = Game.BLACK

# Add pieces to allow corner capture
env.board[1][2] = Game.WHITE  # white at (2,1) in 0-indexed = (1,2)
env.board[2][1] = Game.WHITE  # white at (1,2)
env.board[2][2] = Game.BLACK  # black at (2,2)

print(f"\nCustom board - black legal: {env.get_all_legal_moves(Game.BLACK)}")

for mx, my in env.get_all_legal_moves(Game.BLACK):
    edge_r = (1 if (mx == 0 or mx == 7 or my == 0 or my == 7) else 0) * 0.02
    corner_r = (1 if ((mx == 0 or mx == 7) and (my == 0 or my == 7)) else 0) * 0.04
    
    env_test = OthelloEnv()
    env_test.board = [row[:] for row in env.board]
    env_test.current_player = Game.BLACK
    env_test.place_piece(Game.BLACK, mx, my)
    opp_edge, opp_corner = env_test._count_opponent_square_access(Game.BLACK)
    opp_pen = opp_edge * 0.02 + opp_corner * 0.04
    
    total = edge_r + corner_r - opp_pen
    print(f"  Black ({mx},{my}): edge={edge_r:.3f}, corner={corner_r:.3f}, opp_pen={opp_pen:.3f} -> total={total:.3f}")