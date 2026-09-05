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

The default run samples diverse positions, uses MultiPV targets, and resumes the existing network.
Pass `--engine PATH` when the executable is not on `PATH`.
It also writes a compressed development dataset to `/tmp/az_lite_teacher.npz` for mixed replay.

## Self-play reinforcement learning

Run the same-day MPS training preset from the repository root:

```bash
make train TEACHER=/tmp/az_lite_teacher.npz
```

The trainer runs both colors from the same network. Each position stores the PUCT root visit
distribution as its policy target. The value target is the final game result from that position's
side-to-move perspective; a game capped at `--max-plies` is a draw. It writes
`weights/az_lite.pt`, which is loaded on import by `agent.py` and included by `make zip`.

When `TEACHER` exists, `make train TEACHER=...` mixes a bounded teacher sample into each update
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
