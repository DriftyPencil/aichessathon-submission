# AlphaZero-lite training

Run the same-day MPS training preset from the repository root:

```bash
make train
```

The trainer first labels positions with the team's own depth-2 search, then runs PUCT self-play
and learns from root visit counts and final game results. It writes `weights/az_lite.pt`, which is
loaded on import by `agent.py` and included by `make zip`.

For a quick end-to-end check without replacing the main checkpoint:

```bash
uv run python -m training.train --smoke --out /tmp/az-lite-smoke.pt
```

Longer runs can resume the current network and increase self-play volume:

```bash
uv run python -m training.train --resume --games 24 --iterations 4 --simulations 48
```
