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
`genetic`, and `ppo`. The interactive benchmark picker and the `watch_models.py`
dropdown show only one entry per family, including the path currently selected
as latest. Explicit paths such as `genetic:models/genetic/genetic_gen_0024.json`
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
masking, terminal win/draw/loss rewards, board symmetries, and separate Black
and White trajectories. Rollouts mix current-policy self-play, historical PPO
snapshots, and lightweight scripted opponents. Historical or scripted moves
are never included in the on-policy PPO update.

Start a training run with:

```powershell
python train_ppo.py --iterations 500
```

Training writes resumable `.ppo` checkpoints under `models/ppo`. The iteration
target is cumulative when resuming:

```powershell
python train_ppo.py --resume models/ppo/latest.ppo --iterations 1000
```

For a quick CPU smoke run, reduce the network and rollout size:

```powershell
python train_ppo.py --iterations 1 --rollout-steps 128 --channels 16 `
    --blocks 1 --ppo-epochs 1 --validation-every 0
```

The latest PPO checkpoint is automatically represented by the `ppo` alias in
`benchmark_models.py` and `watch_models.py`. Older checkpoints can still be
selected explicitly:

```powershell
python benchmark_models.py --player-1 ppo:models/ppo/best.ppo `
    --player-2 minimax:2 --games 100
```

