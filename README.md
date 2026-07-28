# cosmos

COSMOS project -- Reversi / Othello

## Comparing models

`benchmark_models.py` accepts stable model names and resolves learned models to
their latest checkpoint automatically:

```powershell
python benchmark_models.py --player-1 genetic --player-2 ppo --games 100
python benchmark_models.py --player-1 dqn --player-2 minimax:3 --games 100
python benchmark_models.py --player-1 alphazero --player-2 minimax:4 --games 20
```

Benchmarks use four randomized opening plies by default and reuse each opening
with the players' colors swapped. This prevents deterministic players from
repeating the same two games throughout a large match. Use
`--opening-plies 0` to benchmark only the standard starting position.

The available names are `random`, `greedy`, `minimax`, `dqn`, `bard`,
`genetic`, `ppo`, and `alphazero` (also `az`). `ppo` combines its learned
policy/value network with depth-2 search, while `alphazero` uses its native
neural MCTS with 256 simulations and exact eight-empty endgames. The interactive
benchmark picker and the `watch_models.py` dropdown include the path currently
selected as latest. Explicit paths such as
`genetic:models/genetic_gen_0024_v2.json` and
`alphazero:models/alphazero/best.az` remain available for reproducible
comparisons with older checkpoints.

The bare `minimax` name means depth 4. When comparing results, keep the minimax
depth, opening plies, game count, and seed identical; a result against
`minimax:2` is not directly comparable to one against bare `minimax`.

## Genetic training

The version-3 genetic trainer keeps the 30-gene, three-phase evaluator introduced
in v2. Controlled like-for-like tests found v2 generation 24 stronger than the
original version-1 player, but also found that v2 generation 49 regressed against
minimax depths 3 and 4. Version 3 therefore protects the strong v2
generation-24 genome, admits only promoted champions to the hall of fame, gives
the deepest affordable minimax target most of the minimax weight, and rejects
promotions that regress against protected opponents.

New runs write versioned filenames such as
`models/genetic/genetic_gen_0049_v3.json` and
`models/genetic/latest_v3.json`; v1 and v2 checkpoints are left untouched. To
continue the latest v2 population under the new training process:

```powershell
python genetic_model.py --resume models/latest_v2.json --generations 100
```

`--generations` is the total generation target. The checkpoint suffix can be
changed with `--checkpoint-suffix`, but defaults to `v3`. Use
`--reference-weight 0` for an ablation run without protected references.

Search now uses transposition caches in both the genetic exact endgame and the
shared minimax player. This preserves move choices while avoiding repeated
evaluation of the same board. Depth 4 is now practical for benchmarking, but it
is still substantially more expensive than depth 3 and is not a default
full-population training opponent.

The exact original version-1 generation-24 best-ever player was recovered from
Git history as
`models/history/genetic/original_gen24_v1_reference.json`. It is reference-only,
not a resumable population checkpoint.

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
alias in `benchmark_models.py` and `watch_models.py`. Older checkpoints can
still be selected explicitly:

```powershell
python benchmark_models.py --player-1 ppo:models/ppo/best.ppo `
    --player-2 minimax:4 --games 100
```

The interactive game in `main.py` also exposes `ppo` while cycling opponents
with the left/right arrow keys. The bound-computer interface is available
directly for other game modes:

```python
from computer import ComputerPPO

computer = ComputerPPO(game, color)  # latest checkpoint, search depth 2
```

`computer.py` imports PyTorch only when a PPO or AlphaZero computer is
constructed.

## AlphaZero training

The AlphaZero implementation is independent of PPO. It trains a residual
policy/WDL network from neural-guided MCTS visit distributions, exact endgame
results, and completed self-play games. It does not use PPO advantages,
likelihood ratios, fixed-opponent actions, WTHOR, or any other human-game
corpus.

All optimized AlphaZero code is isolated in the `alphazero/` package:
`board.py` contains its private rules engine, `mcts.py` its search,
`model.py` its network/inference code, `replay.py` its replay store, and
`training.py` its training pipeline. The top-level `train_alphazero.py` is only
a compatibility launcher. The shared `game.py` and the PPO/genetic model files
are not modified or imported by the training hot path.

Start a standard run with:

```powershell
python train_alphazero.py --generations 500
```

Checkpoints, replay data, snapshots, and metrics are isolated under
`models/alphazero`. Resume targets are cumulative:

```powershell
python train_alphazero.py --resume models/alphazero/latest.az `
    --generations 800
```

Self-play keeps several games active and now selects up to eight collision-aware
leaves per game in each search wave. Temporary virtual loss diversifies those
paths, then one cached neural batch evaluates them and the temporary statistics
are removed before normal backup. This also makes interactive single-game
search genuinely batched instead of issuing one model call per simulation.
Set `--leaf-batch-size 1` for strictly sequential MCTS.

On CUDA, one process owns the model. On CPU, independent game groups run in
persistent spawned processes. The default on a 12-logical-CPU machine is 10
single-threaded workers, matching the Core 5 120U's core topology while
avoiding OpenMP oversubscription. It can still be overridden:

```powershell
python train_alphazero.py --device cpu --self-play-workers 10 `
    --worker-torch-threads 1
```

The private search engine uses immutable two-`uint64` bitboards, cached legal
moves and flip masks, precomputed move rays and D4 symmetry lookup tables,
vectorized NumPy encoders, subtree reuse, prior-mass first-play urgency, and an
exact-endgame transposition table. Endgame search orders TT hints, corners, odd
empty regions, and low opponent mobility; it evicts a bounded cache tranche
instead of clearing the full table. This makes exact solving at the new
ten-empty default practical. Network inference uses channels-last convolution
storage, reuses the already-transferred legal plane as its mask, and evaluates
only the policy and WDL heads; margin and ownership remain training-only.

Replay augmentation is vectorized, cached legal masks avoid regenerating moves,
and prioritized sampling uses a Fenwick sum tree rather than rebuilding a
probability vector over the whole replay buffer. Priorities use policy KL
error, not raw cross-entropy, so a correctly predicted high-entropy target is
not oversampled forever. Duplicate samples are aggregated before updates.
Replay checkpoints are uncompressed by default because that is much faster to
save. Use `--compress-replay` if disk space is more important than save speed.
CPU workers reuse network objects and receive weights through shared memory.

Each generation also refreshes a prioritized sample of old replay positions
with current-network MCTS. A `0.995` exponential moving average of the learned
weights drives self-play, arena evaluation, and deployment, reducing update
noise while raw weights and optimizer state remain resumable. Older version-1
checkpoints remain loadable; their EMA starts from the existing model. Arena
gating now uses 32 color-paired games, longer randomized openings, and a 55%
promotion threshold. Equivalence tests continuously compare the private rules
engine with the unchanged reference `Game`.

For a quick CPU smoke run:

```powershell
python train_alphazero.py --generations 1 --games-per-generation 2 `
    --simulations 2 --channels 8 --blocks 1 --value-hidden 16 `
    --training-steps 1 --batch-size 32 --evaluation-every 0
```

An AlphaZero checkpoint can be used programmatically without changing the main
game engine:

```python
from alphazero.model import AlphaZeroPlayer

player = AlphaZeroPlayer(
    "models/alphazero/best.az",
    simulations=512,
    leaf_batch_size=8,
)
```

The benchmark, spectator, and original game interfaces prefer the promoted
`best.az`; `latest.az` remains the checkpoint to resume training:

```powershell
python benchmark_models.py --player-1 alphazero --player-2 ppo --games 20
python watch_models.py --black alphazero --white minimax:4
```

`main.py` includes an `alphazero` opponent in its left/right selection cycle.
For code using the original bound-computer API:

```python
from computer import ComputerAlphaZero

computer = ComputerAlphaZero(game, color)  # best.az, 512 batched simulations
```

