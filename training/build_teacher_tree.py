"""Build a compact top-line response book from team-generated teacher searches."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import chess
import chess.engine
import chess.polyglot
import numpy as np

from az_model import POLICY_SIZE, action_index
from training.distill import score_from_info, score_number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opening-fens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--nodes", type=int, default=64_000)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hash-mb", type=int, default=512)
    return parser.parse_args()


def find_engine(requested: Path | None) -> Path:
    if requested is not None:
        return requested
    executable = shutil.which("stockfish")
    if executable is None:
        raise SystemExit("Stockfish was not found; pass --engine PATH")
    return Path(executable)


def teacher_moves(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    multipv: int,
) -> list[chess.Move]:
    infos = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        multipv=min(multipv, board.legal_moves.count()),
    )
    rows = infos if isinstance(infos, list) else [infos]
    scored: list[tuple[chess.Move, chess.engine.Score]] = []
    for info in rows:
        variation = info.get("pv")
        if not isinstance(variation, list) or not variation:
            continue
        move = variation[0]
        if not isinstance(move, chess.Move) or move not in board.legal_moves:
            continue
        score = score_from_info(info, board.turn)
        if score is not None:
            scored.append((move, score))
    scored.sort(key=lambda item: score_number(item[1]), reverse=True)
    return [move for move, _ in scored]


def main() -> None:
    args = parse_args()
    if args.nodes < 1 or args.multipv < 1 or args.branch < 1 or args.depth < 0:
        raise ValueError("nodes, multipv, branch, and depth have invalid values")
    fens = tuple(
        line.strip()
        for line in args.opening_fens.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not fens:
        raise ValueError("opening file is empty")
    frontier = list(fens)
    entries: dict[int, int] = {}
    engine = chess.engine.SimpleEngine.popen_uci(str(find_engine(args.engine)))
    try:
        engine.configure({"Threads": args.threads, "Hash": args.hash_mb})
        for ply in range(args.depth + 1):
            next_frontier: list[str] = []
            seen_frontier: set[int] = set()
            for number, fen in enumerate(frontier, start=1):
                board = chess.Board(fen)
                key = chess.polyglot.zobrist_hash(board)
                if key in entries:
                    continue
                moves = teacher_moves(engine, board, args.nodes, max(args.multipv, args.branch))
                if not moves:
                    continue
                entries[key] = action_index(board, moves[0])
                if ply < args.depth:
                    for move in moves[: args.branch]:
                        child = board.copy(stack=False)
                        child.push(move)
                        if child.outcome(claim_draw=True) is None:
                            child_key = chess.polyglot.zobrist_hash(child)
                            if child_key not in entries and child_key not in seen_frontier:
                                seen_frontier.add(child_key)
                                next_frontier.append(child.fen())
                if number % 100 == 0 or number == len(frontier):
                    print(
                        f"tree ply {ply}: {number}/{len(frontier)}, "
                        f"book={len(entries)}, next={len(next_frontier)}",
                        flush=True,
                    )
            frontier = next_frontier
            if not frontier:
                break
    finally:
        engine.quit()

    keys = np.asarray(sorted(entries), dtype=np.uint64)
    actions = np.asarray([entries[int(key)] for key in keys], dtype=np.uint16)
    if np.any(actions >= POLICY_SIZE):
        raise ValueError("tree emitted an invalid policy action")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, keys=keys, actions=actions)
    print(f"saved {len(keys)} tree positions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
