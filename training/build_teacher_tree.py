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
    parser.add_argument("--response-book", action="store_true")
    parser.add_argument("--opponent-branch", type=int, default=8)
    parser.add_argument("--guide-book", type=Path)
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


def guided_move(
    board: chess.Board,
    keys: np.ndarray,
    actions: np.ndarray,
) -> chess.Move | None:
    if not len(keys):
        return None
    key = np.uint64(chess.polyglot.zobrist_hash(board))
    row = int(np.searchsorted(keys, key))
    if row >= len(keys) or keys[row] != key:
        return None
    target = int(actions[row])
    return next(
        (move for move in board.legal_moves if action_index(board, move) == target),
        None,
    )


def main() -> None:
    args = parse_args()
    if (
        args.nodes < 1
        or args.multipv < 1
        or args.branch < 1
        or args.depth < 0
        or args.opponent_branch < 1
    ):
        raise ValueError("nodes, multipv, branch, and depth have invalid values")
    if args.guide_book is not None and not args.response_book:
        raise ValueError("--guide-book requires --response-book")
    fens = tuple(
        line.strip()
        for line in args.opening_fens.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not fens:
        raise ValueError("opening file is empty")
    guide_keys = np.empty(0, dtype=np.uint64)
    guide_actions = np.empty(0, dtype=np.uint16)
    if args.guide_book is not None:
        with np.load(args.guide_book) as archive:
            guide_keys = archive["keys"].astype(np.uint64)
            guide_actions = archive["actions"].astype(np.uint16)
        if len(guide_keys) != len(guide_actions) or (
            len(guide_keys) > 1 and np.any(guide_keys[1:] <= guide_keys[:-1])
        ):
            raise ValueError("guide book keys/actions are malformed")
    frontier: list[tuple[str, chess.Color | None]] = (
        [(fen, color) for fen in fens for color in (chess.WHITE, chess.BLACK)]
        if args.response_book
        else [(fen, None) for fen in fens]
    )
    entries: dict[int, int] = {}
    visited: set[tuple[int, chess.Color | None]] = set()
    engine = chess.engine.SimpleEngine.popen_uci(str(find_engine(args.engine)))
    try:
        engine.configure({"Threads": args.threads, "Hash": args.hash_mb})
        for ply in range(args.depth + 1):
            next_frontier: list[tuple[str, chess.Color | None]] = []
            seen_frontier: set[tuple[int, chess.Color | None]] = set()
            for number, (fen, agent_color) in enumerate(frontier, start=1):
                board = chess.Board(fen)
                key = chess.polyglot.zobrist_hash(board)
                state = key, agent_color
                if state in visited:
                    continue
                visited.add(state)
                branch = (
                    1
                    if agent_color is not None and board.turn == agent_color
                    else args.opponent_branch
                    if agent_color is not None
                    else args.branch
                )
                guide = (
                    guided_move(board, guide_keys, guide_actions)
                    if agent_color is not None and board.turn == agent_color
                    else None
                )
                moves = (
                    [guide]
                    if guide is not None
                    else teacher_moves(engine, board, args.nodes, max(args.multipv, branch))
                )
                if not moves:
                    continue
                entries.setdefault(key, action_index(board, moves[0]))
                if ply < args.depth:
                    for move in moves[:branch]:
                        child = board.copy(stack=False)
                        child.push(move)
                        if child.outcome(claim_draw=True) is None:
                            child_key = chess.polyglot.zobrist_hash(child)
                            child_state = child_key, agent_color
                            if child_state not in visited and child_state not in seen_frontier:
                                seen_frontier.add(child_state)
                                next_frontier.append((child.fen(), agent_color))
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
