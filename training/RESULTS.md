# Reliability experiments, 2026-09-06

All matches below use the unmodified local harness, six paired openings, baseline seeds 601-606,
5s + 0.1s clocks, and the full 600-ply limit. These are local measurements, not Elo estimates.
Every match has a PGN and a JSON report containing the source and checkpoint hashes.

| Build | Wins | Draws | Losses | Score | Report directory |
| --- | ---: | ---: | ---: | ---: | --- |
| Previous submitted checkpoint and search | 3 | 5 | 4 | 45.8% | `runs/before-reliability` |
| Search fixes, previous weights | 3 | 5 | 4 | 45.8% | `runs/search-fixes` |
| Student-position distillation | 6 | 4 | 2 | 66.7% | `runs/distilled-gate` |
| Distillation + self-play + learned exploration | 8 | 3 | 1 | 79.2% | `runs/rl-gate` |

The previous build's wins include one opponent flag. The other runs had no failed
terminations. All four failed the zero-loss strength gate. The earlier 12-game result in the
conversation used a short 160-ply cap and did not establish reliability across openings.

## Training

Distillation generated 16,000 positions with a 60% student move mixture, 16,000 teacher search
nodes, MultiPV 4, and policy temperature 50. The learner started from the previous team-trained
checkpoint. Whole games were split into 14,512 training and 1,488 validation positions. The
selected epoch reduced validation value MSE from 0.2366 to 0.1928; policy agreement remained
approximately 30%. Match strength, not this loss, determines promotion.

Self-play then completed two batches of 24 games at 96 simulations per move, producing 7,107
positions. All games reached a rules-based outcome: 23 checkmates, 12 repetition draws, 4
insufficient-material draws, and 9 fifty-move draws. There were no truncated games. Two training
epochs followed each batch, with teacher replay and a learning rate of 0.00005. The PGNs are in
`runs/selfplay-candidate/selfplay-001` and `selfplay-002`.

Search subsequently gained learned first-play urgency and a correction that keeps proven draws
selectable. The final row measures those changes together with the self-play candidate.

## Competition clock

The same self-play checkpoint and search scored **12 wins, 0 draws, 0 losses** against
`baselines/basic` at 120s + 0.5s, with the full 600-ply limit and no failed terminations. This
run used the same six openings, starting with the Caro-Kann pair, and fresh baseline seeds
1901-1906. Evidence is retained in `runs/rl-competition/report.json` and its twelve PGNs.

The expanded acceptance target is at least eleven wins, no losses, and at most one draw in twelve
games against each of `basic`, `random`, `greedy`, `minimax`, and `numba`. The external Stockfish
development yardstick is not a starter agent. The other starter matches are still pending.

The same frozen build subsequently scored **12 wins, 0 draws, 0 losses** against `minimax`,
and **10 wins, 1 draw, 1 loss** against `numba`. All wins/losses were checkmates and there were
no failed terminations. These two runs used independent workers concurrently, with no training
running, on a Mac with eight performance cores. Reports are in `runs/rl-minimax` and
`runs/rl-numba`. The numba loss means this build does **not** satisfy the expanded target.

## Wider candidate

The 48-channel, three-block network was expanded to 64 channels and four blocks by channel
replication, outgoing-weight splitting, and an initially identity residual block. Outputs were
verified against the source on 49 positions before training. The wider checkpoint is 1.40 MB.

A fresh 32,000-position teacher dataset used 32,000 search nodes, MultiPV 4, 65% student moves,
temperature 50, and a 60% hard-best-move policy mixture. Labelling completed in 577 seconds.
The initial CPU learning stage was interrupted after saving the dataset, to move learning onto
MPS. The restricted runner did not expose the GPU; an approved host-side run did.

GPU learning used 28,520 training and 3,480 whole-game held-out positions, batch size 256,
learning rate 0.00015, and twelve epochs. Epoch ten had the best combined validation loss:
policy 2.3478 -> 2.3023, value MSE 0.2237 -> 0.2012, top-move agreement 31.0% -> 32.8%.
The held-out positions were excluded from subsequent teacher replay. The candidate is in
`runs/wide-gpu-distill/candidate.pt`; these losses alone do not establish playing strength.

The working runtime now has batched leaf inference, subtree-size accounting, a 500,000-node tree
cap, and a 512 MiB LRU neural-output cache. A stress run on the smaller checkpoint filled all
27,235 cache slots, then searched three positions at 8,192 simulations each; peak process memory
was 1,030.3 MiB. This is a local measurement, not the Linux container's peak or a larger-model
measurement. A cached serial 4,096-simulation test earlier improved from 2.460 to 2.311 seconds.
Do not attribute older match results to these newer runtime changes.

## Final starter gate

The frozen snapshot `runs/rl-batched-agent` was tested with the current runtime, full 600-ply
limit, and the competition clock of 120 seconds plus 0.5 seconds per move. Two independent match
workers ran concurrently; no training was active. The full suite used fresh seeded games and
verified the source and weight hashes before accepting each result.

| Starter | Wins | Draws | Losses | Gate |
| --- | ---: | ---: | ---: | --- |
| basic | 12 | 0 | 0 | PASS |
| random | 12 | 0 | 0 | PASS |
| greedy | 12 | 0 | 0 | PASS |
| minimax | 12 | 0 | 0 | PASS |
| numba | 12 | 0 | 0 | PASS |

Aggregate: **60 wins, 0 draws, 0 losses**, all by checkmate, with zero failed terminations.
The retained report is `runs/rl-batched-full-suite/report.json`; the corresponding PGNs are in
each starter subdirectory. This finite local suite is evidence against these seeds and openings,
not a guarantee against every hidden rated position.
