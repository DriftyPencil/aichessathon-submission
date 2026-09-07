# AlphaZero-lite training

The submitted agent is pure neural PUCT. It contains no material values, piece-square tables,
handwritten position scores, or classical search. The training pipeline has two optional stages:
development-only teacher distillation, followed by self-play reinforcement learning. Only the
network weights are shipped.

## Teacher warm start

Stockfish is used only on the development machine to produce policy targets from MultiPV lines and
value targets from its WDL output. The engine, its executable, and this script are never packaged.

```bash
make distill
```

The default run mixes teacher and student moves, uses MultiPV targets, and resumes the existing
network. Validation holds out entire games. Only epochs improving the held-out policy and value
loss are saved. Use a separate candidate checkpoint when running experiments.
Pass `--engine PATH` when the executable is not on `PATH`.
It also writes a compressed development dataset to `/tmp/az_lite_teacher.npz` for mixed replay.

`--policy-hard-mix` blends the top teacher move with its soft MultiPV distribution. Larger
networks can start from the current learned function instead of random initialization:

```bash
.venv/bin/python -m training.widen --output training/runs/wide/initial.pt --channels 64 --blocks 4
```

Pass that checkpoint to distillation with `--initial-weights`. Width and residual depth are
recovered from the saved tensors at loading time. No externally trained weights are used.

## Self-play reinforcement learning

Run from the repository root (MPS or CUDA when available, otherwise CPU):

```bash
make train TEACHER=/tmp/az_lite_teacher.npz
```

The trainer runs both colors from the same network. Each position stores the PUCT root visit
distribution as its policy target. The value target is the final game result from that position's
side-to-move perspective. A game stopped early by `--max-plies` contributes policy targets only:
its unknown result does not train the value head. The actual competition ply limit remains a draw.
Search preserves repetition history and uses exact terminal proofs. File-reflection augmentation
is applied only when neither side retains castling rights.

The trainer initializes from `weights/az_lite.pt` and writes a separate candidate to
`training/runs/selfplay/candidate.pt`, preserving the submitted checkpoint. Each iteration retains
its PGNs alongside the candidate. `--resume` continues a candidate that already exists.

`make train TEACHER=...` mixes a bounded teacher sample into each update
while still generating the positions and final-result targets through self-play. Omit `TEACHER`
for a strictly self-play-only run.

For a quick end-to-end check without replacing the main checkpoint:

```bash
.venv/bin/python -m training.train --smoke --out /tmp/az-lite-smoke.pt
```

Longer runs can increase self-play volume:

```bash
.venv/bin/python -m training.train --resume --games 32 --iterations 12 --simulations 96
```

## Baseline strength gate

```bash
.venv/bin/python -m training.benchmark --output training/runs/baseline-gate
```

This runs twelve games from six openings with swapped colors and seeded baseline randomness,
using the unmodified harness and its full ply limit. It saves every PGN, checkpoint hashes, the
time control, results, and terminations in `report.json`. It fails on any loss, any runtime failure
(including an opponent flag), or more than one draw. Seeds make the opponent reproducible; wall
clock search can still vary. Passing a finite test is not proof of never losing.

The default uses the competition clock from `harness.rules` (currently 120s + 0.5s), so a full
run takes tens of minutes. A fast stress test uses `--base-ms 5000 --increment-ms 100`; report it
separately. Use a new `--output` directory for each run. Stop training during timed matches.
Use one match worker by default; two independent single-threaded workers are reasonable on a
machine with ample idle performance cores and RAM. Record any concurrency with the results.

For the complete five-starter acceptance test, use an immutable copy of the agent and weights:

```bash
.venv/bin/python -m training.suite --agent training/runs/frozen-agent --output training/runs/full-suite --workers 2
```

Each opponent gets twelve games and its own strict gate. The aggregate report requires every
opponent to pass and the source/checkpoint hashes to remain unchanged.

## Search resources

The runtime reuses the surviving subtree between moves and maintains a bounded LRU cache of
neural outputs. Cache keys include every encoded input; terminal outcomes are never cached across
histories. Batched inference clones each stored policy row to avoid retaining whole batch tensors.
The cache has a 512 MiB allocation budget; the tree has a separate 500,000-node bound (with at
most one batch of expansion overshoot). These are not a claim about total process memory.

Measure the actual peak, including PyTorch, cache storage, and search nodes:

```bash
.venv/bin/python -m training.profile --stress-cache --simulations 16384
```

Keep substantial headroom under the container limit. CPU timing and allocator behavior can differ
between this Mac and the Linux competition machine.
