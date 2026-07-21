# cosmos

COSMOS project -- Reversi / Othello

## Comparing models

`benchmark_models.py` accepts stable model names and resolves learned models to
their latest checkpoint automatically:

```powershell
python benchmark_models.py --player-1 genetic --player-2 ppo --games 100
python benchmark_models.py --player-1 dqn --player-2 minimax:3 --games 100
```

The available names are `random`, `greedy`, `minimax`, `dqn`, `bard`,
`genetic`, `ppo`, and `ppo-raw`. `ppo` combines the learned policy/value network
with depth-2 policy-guided search and an exact eight-empty endgame search;
`ppo-raw` measures the network without search. The interactive benchmark picker
and the `watch_models.py` dropdown include the path currently selected as
latest. Explicit paths such as `genetic:models/genetic/genetic_gen_0024.json`
remain available for reproducible comparisons with older checkpoints.

## Genetic training

The genetic trainer uses paired randomized openings, population coevolution,
historical genetic opponents, a fixed seed evaluator, and a range of lightweight
search anchors. Bard and DQN models are not used as training or validation
opponents.

New runs write versioned filenames such as
`models/genetic/genetic_gen_0049_v2.json` and
`models/genetic/latest_v2.json`; the original unsuffixed checkpoints are left
untouched. To continue the existing population under the new training process:

```powershell
python genetic_model.py --resume models/genetic/latest.json --generations 100
```

`--generations` is the total generation target. The checkpoint suffix can be
changed with `--checkpoint-suffix`, but defaults to `v2`.

## PPO training

The PPO trainer uses a residual convolutional actor-critic, legal-action
masking, board symmetries, batched parallel games, and separate Black and White
trajectories. Its critic learns a Monte Carlo target combining the final result
with a small disc-margin tie-break, while GAE is retained for policy advantages.
Entropy is normalized by the number of legal moves and the learning rate follows
a cosine schedule with a nonzero floor.

Rollouts adaptively mix current-policy self-play, a rolling PPO league, promoted
champions, and a curriculum ranging from random/greedy through minimax depth 4
and genetic v2. Opponents near the learner's current strength receive more
sampling weight. Historical and scripted opponent moves are never included in
the on-policy likelihood-ratio update.

This training path does **not** load WTHOR or any other human-game corpus. By
default, 2% of current rollout positions receive an auxiliary depth-3 minimax
action target generated online. This can be disabled with
`--teacher-fraction 0`.

Start a training run with:

```powershell
python train_ppo.py --iterations 800
```

Training writes resumable `.ppo` checkpoints under `models/ppo`. The iteration
target is cumulative when resuming:

```powershell
python train_ppo.py --resume models/latest.ppo --iterations 800
```

Existing checkpoints from the earlier trainer are architecture-compatible and
can be resumed directly; new checkpoints default to `models/ppo`. The defaults
collect 8,192 decisions across up to 32 simultaneous games, use four PPO epochs,
and checkpoint every ten iterations. Fast validation runs every ten iterations;
every 50 iterations a larger searched evaluation against minimax depths 2-4 and
genetic v2 controls promotion to `best.ppo` and the persistent champion pool.

For a quick CPU smoke run, reduce the network and rollout size:

```powershell
python train_ppo.py --iterations 1 --rollout-steps 128 --channels 16 `
    --blocks 1 --ppo-epochs 1 --validation-every 0 --champion-every 0
```

The latest PPO checkpoint is automatically represented by the searched `ppo`
alias in `benchmark_models.py` and `watch_models.py`. Use `ppo-raw` to isolate
network quality. Older checkpoints can still be selected explicitly:

```powershell
python benchmark_models.py --player-1 ppo:models/ppo/best.ppo `
    --player-2 minimax:4 --games 100
python benchmark_models.py --player-1 ppo-raw:models/ppo/best.ppo `
    --player-2 minimax:4 --games 100
```

The interactive game in `main.py` also exposes `ppo` and `ppo-raw` while
cycling opponents with the left/right arrow keys. The bound-computer interface
is available directly for other game modes:

```python
from computer import ComputerPPO

computer = ComputerPPO(game, color)  # latest checkpoint, search depth 2
raw_computer = ComputerPPO(
    game,
    color,
    search_depth=0,
    endgame_exact_empties=0,
)
```

`computer.py` imports PyTorch only when a PPO computer is constructed.

