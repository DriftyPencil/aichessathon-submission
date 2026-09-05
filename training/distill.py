"""Development-only Stockfish teacher for policy/value distillation.

This module is never imported by the submitted agent. It contains no chess evaluator: every
training target comes from the external UCI teacher's principal variations and WDL score.
"""

from __future__ import annotations

import argparse
import random
import shutil
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch

from az_model import POLICY_SIZE, AlphaZeroLite, action_index, encode_board
from training.train import (
    Experience,
    choose_device,
    legal_action_indices,
    load_dataset,
    save_weights,
    train_epochs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("weights/az_lite.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("/tmp/az_lite_teacher.npz"))
    parser.add_argument("--reuse-dataset", action="store_true")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--positions", type=int, default=12_000)
    parser.add_argument("--nodes", type=int, default=5_000)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--parallel-games", type=int, default=16)
    parser.add_argument("--opening-plies", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--policy-temperature", type=float, default=95.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hash-mb", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def find_engine(requested: Path | None) -> Path:
    if requested is not None:
        return requested
    executable = shutil.which("stockfish")
    if executable is None:
        raise SystemExit("Stockfish was not found; install it or pass --engine PATH")
    return Path(executable)


def score_from_info(info: dict[str, object], turn: chess.Color) -> chess.engine.Score | None:
    raw_score = info.get("score")
    if not isinstance(raw_score, chess.engine.PovScore):
        return None
    return raw_score.pov(turn)


def score_value(score: chess.engine.Score) -> float:
    return float(2.0 * score.wdl(model="sf", ply=30).expectation() - 1.0)


def score_number(score: chess.engine.Score) -> int:
    value = score.score(mate_score=100_000)
    return 0 if value is None else value


def random_opening(rng: np.random.Generator, plies: int) -> chess.Board:
    for _ in range(8):
        board = chess.Board()
        count = int(rng.integers(plies + 1))
        for _ in range(count):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[int(rng.integers(len(moves)))])
        if board.outcome(claim_draw=True) is None and board.legal_moves.count():
            return board
    return chess.Board()


def analyse_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    multipv: int,
    policy_temperature: float,
) -> tuple[Experience, list[chess.Move], np.ndarray]:
    legal_count = board.legal_moves.count()
    result = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        multipv=min(multipv, legal_count),
    )
    infos = result if isinstance(result, list) else [result]
    candidates: list[tuple[chess.Move, chess.engine.Score]] = []
    for info in infos:
        variation = info.get("pv")
        if not isinstance(variation, list) or not variation:
            continue
        move = variation[0]
        if not isinstance(move, chess.Move) or move not in board.legal_moves:
            continue
        score = score_from_info(info, board.turn)
        if score is not None:
            candidates.append((move, score))
    if not candidates:
        raise RuntimeError(f"teacher returned no legal principal variation for {board.fen()}")

    candidates.sort(key=lambda item: score_number(item[1]), reverse=True)
    moves = [item[0] for item in candidates]
    scores = np.asarray([score_number(item[1]) for item in candidates], dtype=np.float32)
    logits = (scores - scores.max()) / max(1.0, policy_temperature)
    probabilities = np.exp(np.clip(logits, -30.0, 0.0))
    probabilities /= probabilities.sum()

    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    for move, probability in zip(moves, probabilities, strict=True):
        policy[action_index(board, move)] = probability
    value = score_value(candidates[0][1])
    return (
        Experience(encode_board(board), policy, value, legal_action_indices(board)),
        moves,
        probabilities,
    )


def teacher_examples(
    engine: chess.engine.SimpleEngine,
    count: int,
    nodes: int,
    multipv: int,
    parallel_games: int,
    opening_plies: int,
    max_plies: int,
    policy_temperature: float,
    rng: np.random.Generator,
) -> list[Experience]:
    boards = [random_opening(rng, opening_plies) for _ in range(parallel_games)]
    plies = [board.ply() for board in boards]
    examples: list[Experience] = []
    started = time.monotonic()
    cursor = 0

    while len(examples) < count:
        game_index = cursor % parallel_games
        cursor += 1
        board = boards[game_index]
        if (
            board.outcome(claim_draw=True) is not None
            or not board.legal_moves.count()
            or plies[game_index] >= max_plies
        ):
            board = random_opening(rng, opening_plies)
            boards[game_index] = board
            plies[game_index] = board.ply()

        example, moves, probabilities = analyse_position(
            engine, board, nodes, multipv, policy_temperature
        )
        examples.append(example)
        temperature = 1.0 if plies[game_index] < 24 else 0.45
        sampling = np.power(np.maximum(probabilities, 1e-12), 1.0 / temperature)
        sampling /= sampling.sum()
        board.push(moves[int(rng.choice(len(moves), p=sampling))])
        plies[game_index] += 1

        if len(examples) % 100 == 0 or len(examples) == count:
            elapsed = time.monotonic() - started
            print(
                f"teacher labels {len(examples):>5}/{count} "
                f"({elapsed:.1f}s, {len(examples) / elapsed:.1f} pos/s)",
                flush=True,
            )
    return examples


def save_dataset(examples: list[Experience], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    states = np.stack([example.state for example in examples]).astype(np.float16)
    policies = np.stack([example.policy for example in examples]).astype(np.float16)
    values = np.asarray([example.value for example in examples], dtype=np.float32)
    action_sets = [example.legal_actions for example in examples]
    if any(actions is None for actions in action_sets):
        raise ValueError("teacher examples must include legal action indices")
    max_actions = max(len(actions) for actions in action_sets if actions is not None)
    legal_actions = np.full((len(examples), max_actions), -1, dtype=np.int16)
    legal_counts = np.zeros(len(examples), dtype=np.int16)
    for row, actions in enumerate(action_sets):
        if actions is None:
            continue
        legal_actions[row, : len(actions)] = actions
        legal_counts[row] = len(actions)
    np.savez_compressed(
        destination,
        states=states,
        policies=policies,
        values=values,
        legal_actions=legal_actions,
        legal_counts=legal_counts,
    )
    print(f"saved teacher dataset {destination} ({destination.stat().st_size / 1_000_000:.1f} MB)")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.positions = 12
        args.nodes = 500
        args.multipv = 3
        args.parallel_games = 2
        args.opening_plies = 4
        args.epochs = 1
        args.batch_size = 6
        args.threads = 2
        args.hash_mb = 64

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    print(f"learner={device}, torch={torch.__version__}", flush=True)
    if args.reuse_dataset and args.dataset.exists():
        examples = load_dataset(args.dataset)
        print(f"reusing {len(examples)} teacher positions from {args.dataset}", flush=True)
    else:
        engine_path = find_engine(args.engine)
        print(f"teacher={engine_path}", flush=True)
        engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
        try:
            engine.configure({"Threads": args.threads, "Hash": args.hash_mb})
            examples = teacher_examples(
                engine,
                args.positions,
                args.nodes,
                args.multipv,
                args.parallel_games,
                args.opening_plies,
                args.max_plies,
                args.policy_temperature,
                rng,
            )
        finally:
            engine.quit()
        save_dataset(examples, args.dataset)

    model = AlphaZeroLite().to(device)
    if args.weights.exists():
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
        print(f"resumed {args.weights}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    train_epochs(model, examples, optimizer, device, args.epochs, args.batch_size, rng)
    save_weights(model, args.weights)


if __name__ == "__main__":
    main()
