# Hard Evaluation Agent: Improvement Backlog

Research captured on 2026-09-08 from the Chess Programming Wiki Basics section and a local
`cProfile` run of `hard_evaluation_agent.py`. This is a development backlog, not evidence that
an item improves Elo until it passes paired-game testing.

## Current profile

A full-clock starting-position search under `cProfile` took 4.17 seconds. Cumulative timings
overlap, but identify the main costs:

- quiescence search: 2.09 seconds
- move generation and ordering: 1.69 seconds
- evaluation: 1.56 seconds
- legal move generation: 1.11 seconds
- recomputing Polyglot hashes: 0.36 seconds
- about 54,000 main-search and quiescence nodes

## Correctness audit

Fixed on 2026-09-08 in `hard_evaluation_agent.py`:

- Root twofold repetitions no longer return before iterative deepening selects a move.
- Quiescence and null-window pruning distinguish stalemate from a quiet position.
- Internal insufficient-material positions score as draws, and checkmate takes precedence over
  the fifty-move threshold.
- Root TT cutoffs restore the stored legal move instead of returning only a score.
- Mate scores are normalized when stored in and loaded from the TT.
- Time checks run every 128 nodes, and forced moves bypass search.

These fixes remove reproduced search errors; they are not, by themselves, an Elo claim. At the
3,000 ms + 100 ms local test control, the corrected agent scored 4-0 against `baselines/basic`,
2-0 against `baselines/minimax`, and 0-1-1 against `baselines/experiment`.

## Recommended order

1. Replace full move-list sorting with staged ordering: TT move, good captures/promotions,
   killers, history-ranked quiets, then bad captures.
2. Add static exchange evaluation (SEE), then use it for capture ordering and pruning.
3. Add SEE and delta pruning to quiescence, which currently consumes roughly half the search.
4. Keep the TT direct-mapped but use depth-preferred replacement, generation aging, and cached
   static evaluation/phase.
5. Incrementally maintain PST/material/phase and the Zobrist key, or first introduce a small
   evaluation cache and pawn hash as a lower-risk step.
6. Make LMR contextual: reduce less for checks, killers, passed-pawn pushes, PV nodes, improving
   positions, and moves with strong history; tune a depth/move-count reduction table.
7. Refine null-move pruning with evaluation-dependent reductions and verification in susceptible
   piece endgames.
8. Make time use respond to best-move stability, score oscillation, aspiration failures, legal
   move count, and the known increment.
9. Expand evaluation with tuned knight mobility, bishop pair, pawn structure, rook files, king
    shelter/storm, and nonlinear king-attack scaling.
10. Consider a position-keyed opening book and compact endgame bitbases after the search is sound.
11. Treat a custom Numba bitboard/make-unmake implementation as a high-ceiling, high-risk project;
    require perft and state-restoration tests before strength testing.

## Testing discipline

- Apply one behavioral change at a time.
- Keep deterministic fixed-depth/node-count regression positions for search correctness.
- Test tactical, mate-distance, stalemate, repetition, fifty-move, castling, en-passant, and
  promotion cases.
- Measure nodes, completed depth, fail-high-first rate, TT hit/cutoff rate, qsearch share, and
  timeout overshoot.
- Use paired openings with both colors and enough games for SPRT or equivalent confidence.
- Do not infer Elo from one game per opponent.

## Sources

- https://chessprogramming.org/Getting_Started
- https://chessprogramming.org/Board_Representation
- https://chessprogramming.org/Search
- https://chessprogramming.org/Move_Ordering
- https://chessprogramming.org/Static_Exchange_Evaluation
- https://chessprogramming.org/Quiescence_Search
- https://chessprogramming.org/Transposition_Table
- https://chessprogramming.org/Incremental_Updates
- https://chessprogramming.org/Null_Move_Pruning
- https://chessprogramming.org/Late_Move_Reductions
- https://chessprogramming.org/Evaluation
- https://chessprogramming.org/Pawn_Structure
- https://chessprogramming.org/King_Safety
- https://chessprogramming.org/Automated_Tuning
- https://chessprogramming.org/Time_Management
- https://chessprogramming.org/Opening_Book
- https://chessprogramming.org/Endgame_Tablebases
