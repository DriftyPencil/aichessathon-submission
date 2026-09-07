"""Measure local neural-search latency, fixed-node throughput, and peak process memory."""

import argparse
import random
import resource
import sys
import time

import chess

import agent

FENS = (
    chess.STARTING_FEN,
    "r1bq1rk1/ppp2ppp/2n1pn2/3p4/3P4/2N1PN2/PPPN1PPP/R1BQKB1R w KQ - 2 6",
    "r3k2r/ppp2ppp/2n1bn2/3p4/3P4/2N1PN2/PPPN1PPP/R2QKB1R w KQ - 2 10",
)

def peak_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def fill_cache() -> None:
    rng = random.Random(20260906)
    board = chess.Board()
    target = agent._CACHE_MISSES + agent.MAX_CACHED_EVALUATIONS + 1024
    while target > agent._CACHE_MISSES:
        boards = []
        for _ in range(64):
            if board.is_game_over(claim_draw=True) or board.ply() >= 120:
                board.reset()
            board.push(rng.choice(list(board.legal_moves)))
            boards.append(board.copy(stack=False))
        agent._predict_many(boards)
    assert len(agent._EVALUATIONS) == agent.MAX_CACHED_EVALUATIONS
    print(f"full LRU: {len(agent._EVALUATIONS)} entries, peak={peak_mib():.1f} MiB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-cache", action="store_true")
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--batch-size", type=int, default=agent.SEARCH_BATCH_SIZE)
    parser.add_argument("--position-count", type=int, default=len(FENS))
    args = parser.parse_args()
    agent.SEARCH_BATCH_SIZE = args.batch_size
    if args.stress_cache:
        fill_cache()
    for fen in FENS[: args.position_count]:
        for clock in ((120_000,) if args.simulations else (100, 1_000, 10_000, 120_000)):
            agent._GAME_BOARD = None
            agent._GAME_ROOT = None
            if not args.stress_cache:
                agent._EVALUATIONS.clear()
            started = time.perf_counter()
            if args.simulations:
                agent.MAX_SIMULATIONS = args.simulations
                move = agent._mcts_move(chess.Board(fen), started + 600).uci()
            else:
                move = agent.get_move(fen, clock)
            elapsed = time.perf_counter() - started
            assert chess.Move.from_uci(move) in chess.Board(fen).legal_moves
            root = agent._GAME_ROOT
            print(
                f"clock={clock:>6} move={move} time={elapsed * 1000:>7.0f}ms "
                f"retained_tree={root.size if root else 0} peak={peak_mib():.1f} MiB",
                flush=True,
            )
    print(f"cache hits={agent._CACHE_HITS}, misses={agent._CACHE_MISSES}", flush=True)


if __name__ == "__main__":
    main()
