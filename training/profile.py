"""Print local neural-search latency for representative positions."""

import time

import chess

import agent

FENS = (
    chess.STARTING_FEN,
    "r1bq1rk1/ppp2ppp/2n1pn2/3p4/3P4/2N1PN2/PPPN1PPP/R1BQKB1R w KQ - 2 6",
    "r3k2r/ppp2ppp/2n1bn2/3p4/3P4/2N1PN2/PPPN1PPP/R2QKB1R w KQ - 2 10",
)

for fen in FENS:
    for clock in (10_000, 120_000):
        agent._GAME_BOARD = None
        started = time.perf_counter()
        move = agent.get_move(fen, clock)
        elapsed = (time.perf_counter() - started) * 1000
        print(f"clock={clock:>6} move={move} time={elapsed:>7.0f}ms")
