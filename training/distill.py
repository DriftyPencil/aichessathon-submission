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
import chess.polyglot
import numpy as np
import torch

from az_model import POLICY_SIZE, AlphaZeroLite, action_index, encode_board, model_from_state
from training.train import (
    Experience,
    choose_device,
    legal_action_indices,
    load_dataset,
    mask_policy_logits,
    save_weights,
    train_epochs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("training/runs/distill/candidate.pt"))
    parser.add_argument("--initial-weights", type=Path, default=Path("weights/az_lite.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("/tmp/az_lite_teacher.npz"))
    parser.add_argument("--reuse-dataset", action="store_true")
    parser.add_argument("--mix-dataset", type=Path, action="append", default=[])
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--positions", type=int, default=12_000)
    parser.add_argument("--nodes", type=int, default=5_000)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--parallel-games", type=int, default=16)
    parser.add_argument("--opening-plies", type=int, default=10)
    parser.add_argument("--opening-fens", type=Path)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--policy-temperature", type=float, default=95.0)
    parser.add_argument("--policy-hard-mix", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hash-mb", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--student-fraction", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--epoch-directory", type=Path)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--generate-only", action="store_true")
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


def random_opening(
    rng: np.random.Generator,
    plies: int,
    student: AlphaZeroLite | None = None,
    starting_fens: tuple[str, ...] = (),
) -> chess.Board:
    for _ in range(8):
        if starting_fens:
            board = chess.Board(starting_fens[int(rng.integers(len(starting_fens)))])
        else:
            board = chess.Board()
        count = int(rng.integers(plies + 1))
        for _ in range(count):
            legal = list(board.legal_moves)
            if not legal:
                break
            if student is None:
                move = legal[int(rng.integers(len(legal)))]
            else:
                with torch.inference_mode():
                    logits, _ = student(torch.from_numpy(encode_board(board)).unsqueeze(0))
                scores = logits[0, [action_index(board, move) for move in legal]].numpy()
                probabilities = np.exp((scores - scores.max()) / 1.2)
                probabilities /= probabilities.sum()
                move = legal[int(rng.choice(len(legal), p=probabilities))]
            board.push(move)
        if board.outcome(claim_draw=True) is None and board.legal_moves.count():
            return board
    return chess.Board()


def analyse_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    multipv: int,
    policy_temperature: float,
    policy_hard_mix: float = 0.0,
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
    probabilities *= 1.0 - policy_hard_mix
    probabilities[0] += policy_hard_mix

    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    for move, probability in zip(moves, probabilities, strict=True):
        policy[action_index(board, move)] = probability
    value = score_value(candidates[0][1])
    return (
        Experience(
            encode_board(board),
            policy,
            value,
            legal_action_indices(board),
            position_key=chess.polyglot.zobrist_hash(board),
            best_action=action_index(board, moves[0]),
        ),
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
    student: AlphaZeroLite | None = None,
    student_fraction: float = 0.5,
    policy_hard_mix: float = 0.0,
    starting_fens: tuple[str, ...] = (),
) -> list[Experience]:
    if starting_fens:
        shuffled_fens = [
            starting_fens[int(index)] for index in rng.permutation(len(starting_fens))
        ]
        initial_fens = [
            (shuffled_fens[index % len(shuffled_fens)],)
            for index in range(parallel_games)
        ]
    else:
        initial_fens = [()] * parallel_games
    boards = [
        random_opening(rng, opening_plies, student, initial_fens[index])
        for index in range(parallel_games)
    ]
    plies = [board.ply() for board in boards]
    examples: list[Experience] = []
    started = time.monotonic()
    cursor = 0
    game_ids = list(range(parallel_games))
    next_game_id = parallel_games

    while len(examples) < count:
        game_index = cursor % parallel_games
        cursor += 1
        board = boards[game_index]
        if (
            board.outcome(claim_draw=True) is not None
            or not board.legal_moves.count()
            or plies[game_index] >= max_plies
        ):
            board = random_opening(rng, opening_plies, student, starting_fens)
            boards[game_index] = board
            plies[game_index] = board.ply()
            game_ids[game_index] = next_game_id
            next_game_id += 1

        example, moves, probabilities = analyse_position(
            engine, board, nodes, multipv, policy_temperature, policy_hard_mix
        )
        example.game_id = game_ids[game_index]
        examples.append(example)
        temperature = 1.0 if plies[game_index] < 24 else 0.45
        sampling = np.power(np.maximum(probabilities, 1e-12), 1.0 / temperature)
        sampling /= sampling.sum()
        if student is not None and rng.random() < student_fraction:
            legal = list(board.legal_moves)
            with torch.inference_mode():
                logits, _ = student(torch.from_numpy(example.state).unsqueeze(0))
            legal_logits = logits[0, [action_index(board, move) for move in legal]].numpy()
            legal_probabilities = np.exp((legal_logits - legal_logits.max()) / 0.8)
            legal_probabilities /= legal_probabilities.sum()
            move = legal[int(rng.choice(len(legal), p=legal_probabilities))]
        else:
            move = moves[int(rng.choice(len(moves), p=sampling))]
        board.push(move)
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
    arrays: dict[str, np.ndarray] = {
        "states": states,
        "policies": policies,
        "values": values,
        "legal_actions": legal_actions,
        "legal_counts": legal_counts,
        "game_ids": np.asarray([example.game_id for example in examples], dtype=np.int64),
    }
    if all(example.position_key is not None for example in examples) and all(
        example.best_action is not None for example in examples
    ):
        arrays["position_keys"] = np.asarray(
            [example.position_key for example in examples], dtype=np.uint64
        )
        arrays["best_actions"] = np.asarray(
            [example.best_action for example in examples], dtype=np.uint16
        )
    np.savez_compressed(destination, **arrays)
    print(f"saved teacher dataset {destination} ({destination.stat().st_size / 1_000_000:.1f} MB)")


def split_games(
    examples: list[Experience],
    fraction: float,
    rng: np.random.Generator,
) -> tuple[list[Experience], list[Experience]]:
    groups = sorted({example.game_id for example in examples})
    if -1 in groups or len(groups) < 2:
        raise ValueError("validation requires a newly generated dataset with game IDs")
    count = max(1, min(len(groups) - 1, round(len(groups) * fraction)))
    held_out = set(rng.choice(groups, count, replace=False).tolist())
    return (
        [e for e in examples if e.game_id not in held_out],
        [e for e in examples if e.game_id in held_out],
    )


def validation_metrics(
    model: AlphaZeroLite,
    examples: list[Experience],
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    policy_losses: list[float] = []
    predictions: list[float] = []
    targets: list[float] = []
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(examples), 256):
            batch = examples[start : start + 256]
            states = torch.from_numpy(np.stack([e.state for e in batch])).to(device)
            logits, values = model(states)
            predictions.extend(values.cpu().tolist())
            targets.extend(e.value for e in batch)
            if any(example.legal_actions is None for example in batch):
                raise ValueError("validation requires legal action indices")
            legal = mask_policy_logits(logits, [example.legal_actions for example in batch])
            target = torch.from_numpy(np.stack([example.policy for example in batch])).to(device)
            policy_losses.extend((-(target * legal.log_softmax(1)).sum(1)).cpu().tolist())
            correct += int((legal.argmax(1) == target.argmax(1)).sum().cpu())
    mse = float(np.mean((np.asarray(predictions) - targets) ** 2))
    return float(np.mean(policy_losses)), mse, correct / len(examples)


def main() -> None:
    args = parse_args()
    if not 0 <= args.policy_hard_mix <= 1:
        raise ValueError("--policy-hard-mix must be between 0 and 1")
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
    torch.set_num_threads(1)
    device = choose_device(args.device)
    initial_weights = args.weights if args.weights.exists() else args.initial_weights
    starting_fens: tuple[str, ...] = ()
    if args.opening_fens is not None:
        starting_fens = tuple(
            line.strip()
            for line in args.opening_fens.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        for fen in starting_fens:
            if chess.Board(fen).outcome(claim_draw=True) is not None:
                raise ValueError(f"opening is already terminal: {fen}")
        print(f"loaded {len(starting_fens)} rated opening positions", flush=True)
    print(f"learner={device}, torch={torch.__version__}", flush=True)
    if args.reuse_dataset and args.dataset.exists():
        examples = load_dataset(args.dataset)
        print(f"reusing {len(examples)} teacher positions from {args.dataset}", flush=True)
    else:
        engine_path = find_engine(args.engine)
        print(f"teacher={engine_path}", flush=True)
        engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
        student = AlphaZeroLite().eval()
        if initial_weights.exists():
            student = model_from_state(
                torch.load(initial_weights, map_location="cpu", weights_only=True)
            ).eval()
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
                student=student if initial_weights.exists() else None,
                student_fraction=args.student_fraction,
                policy_hard_mix=args.policy_hard_mix,
                starting_fens=starting_fens,
            )
        finally:
            engine.quit()
        save_dataset(examples, args.dataset)
    if args.generate_only:
        return
    next_game_id = max((example.game_id for example in examples), default=-1) + 1
    for source in args.mix_dataset:
        added = load_dataset(source)
        for example in added:
            if example.game_id >= 0:
                example.game_id += next_game_id
        examples.extend(added)
        next_game_id = max((example.game_id for example in examples), default=-1) + 1
        print(f"mixed in {len(added)} teacher positions from {source}", flush=True)

    model = AlphaZeroLite()
    if initial_weights.exists():
        model = model_from_state(torch.load(initial_weights, map_location="cpu", weights_only=True))
        print(f"initialized from {initial_weights}", flush=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    if args.smoke:
        train_epochs(
            model,
            examples,
            optimizer,
            device,
            args.epochs,
            args.batch_size,
            rng,
            value_loss_weight=args.value_loss_weight,
        )
        save_weights(model, args.weights)
        return
    training, validation = split_games(examples, args.validation_fraction, rng)
    print(f"game-disjoint split: {len(training)} train, {len(validation)} validation", flush=True)
    before = validation_metrics(model, validation, device)
    best = before[0] + args.value_loss_weight * before[1]
    print(
        f"initial validation policy={before[0]:.4f} mse={before[1]:.4f} top1={before[2]:.1%}",
        flush=True,
    )
    for epoch in range(args.epochs):
        train_epochs(
            model,
            training,
            optimizer,
            device,
            1,
            args.batch_size,
            rng,
            value_loss_weight=args.value_loss_weight,
        )
        policy_loss, mse, top1 = validation_metrics(model, validation, device)
        print(
            f"validation {epoch + 1}: policy={policy_loss:.4f} mse={mse:.4f} top1={top1:.1%}",
            flush=True,
        )
        if args.epoch_directory is not None:
            save_weights(model, args.epoch_directory / f"epoch-{epoch + 1:03d}.pt")
        if policy_loss + args.value_loss_weight * mse < best:
            best = policy_loss + args.value_loss_weight * mse
            save_weights(model, args.weights)


if __name__ == "__main__":
    main()
