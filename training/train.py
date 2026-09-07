"""Train AlphaZero-lite with neural PUCT self-play and game outcomes."""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
from torch import Tensor, nn

from az_model import (
    MIRROR_ACTION_INDICES,
    POLICY_SIZE,
    AlphaZeroLite,
    action_index,
    encode_board,
    mirror_policy,
    mirror_state,
    model_from_state,
)
from harness.rules import PLY_CAP

PUCT = 1.55
FPU_REDUCTION = 0.2


@dataclass(slots=True)
class Experience:
    state: np.ndarray
    policy: np.ndarray
    value: float
    legal_actions: np.ndarray | None = None
    value_weight: float = 1.0
    game_id: int = -1
    position_key: int | None = None
    best_action: int | None = None


@dataclass(slots=True)
class TrainingNode:
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    proven: float | None = None
    children: dict[chess.Move, TrainingNode] = field(default_factory=dict)

    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class SelfPlayGame:
    board: chess.Board = field(default_factory=chess.Board)
    trajectory: list[tuple[np.ndarray, np.ndarray, np.ndarray, chess.Color]] = field(
        default_factory=list
    )
    plies: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("training/runs/selfplay/candidate.pt"))
    parser.add_argument("--initial-weights", type=Path, default=Path("weights/az_lite.pt"))
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--opening-fens", type=Path)
    parser.add_argument("--simulations", type=int, default=96)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=40_000)
    parser.add_argument("--max-plies", type=int, default=PLY_CAP)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--teacher-dataset", type=Path)
    parser.add_argument("--teacher-mix", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "mps" or (name == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def predict_batch(
    model: AlphaZeroLite, boards: list[chess.Board], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    if not boards:
        return np.empty((0, POLICY_SIZE), dtype=np.float32), np.empty(0, dtype=np.float32)
    states = np.stack([encode_board(board) for board in boards])
    inputs = torch.from_numpy(states).to(device)
    with torch.inference_mode():
        policy, value = model(inputs)
    return policy.float().cpu().numpy(), value.float().cpu().numpy()


def expand_node(
    node: TrainingNode,
    board: chess.Board,
    logits: np.ndarray,
    add_noise: bool,
    rng: np.random.Generator,
) -> None:
    moves = list(board.legal_moves)
    if not moves:
        node.expanded = True
        return

    indices = np.asarray([action_index(board, move) for move in moves], dtype=np.int64)
    legal_logits = logits[indices]
    legal_logits = legal_logits - legal_logits.max()
    priors = np.exp(np.clip(legal_logits, -30.0, 0.0))
    priors /= priors.sum()
    if add_noise:
        noise = rng.dirichlet(np.full(len(moves), 0.3))
        priors = 0.75 * priors + 0.25 * noise

    node.children = {
        move: TrainingNode(prior=float(prior)) for move, prior in zip(moves, priors, strict=True)
    }
    if add_noise:
        for move, child in node.children.items():
            board.push(move)
            try:
                if board.is_checkmate():
                    child.proven = -1.0
                    node.proven = 1.0
            finally:
                board.pop()
    node.expanded = True


def select_child(node: TrainingNode) -> tuple[chess.Move, TrainingNode]:
    parent_scale = math.sqrt(max(1, node.visits))

    def score(item: tuple[chess.Move, TrainingNode]) -> float:
        child = item[1]
        if child.proven is not None:
            return 0.0 if child.proven == 0.0 else -math.inf
        value = -child.value() if child.visits else node.value() - FPU_REDUCTION
        return value + PUCT * child.prior * parent_scale / (1 + child.visits)

    return max(node.children.items(), key=score)


def backpropagate(path: list[TrainingNode], leaf_value: float) -> None:
    value = leaf_value
    for node in reversed(path):
        if node.proven is None and node.children:
            proofs = [child.proven for child in node.children.values()]
            if -1.0 in proofs:
                node.proven = 1.0
            elif all(proof is not None for proof in proofs):
                node.proven = max(-proof for proof in proofs if proof is not None)
        if node.proven is not None:
            value = node.proven
        node.visits += 1
        node.value_sum += value
        value = -value


def terminal_value(board: chess.Board, outcome: chess.Outcome) -> float:
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def search_terminal_value(board: chess.Board) -> float | None:
    """Return an exact result while avoiding speculative repetition scans on every leaf."""
    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        return terminal_value(board, outcome)
    if board.halfmove_clock >= 99 and board.can_claim_fifty_moves():
        return 0.0
    if board.is_repetition(2) and board.can_claim_threefold_repetition():
        return 0.0
    return None


def batched_search(
    model: AlphaZeroLite,
    boards: list[chess.Board],
    simulations: int,
    device: torch.device,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if not boards:
        return []

    roots = [TrainingNode() for _ in boards]
    root_logits, _ = predict_batch(model, boards, device)
    for root, board, logits in zip(roots, boards, root_logits, strict=True):
        expand_node(root, board, logits, True, rng)

    for _ in range(max(1, simulations)):
        pending: list[tuple[TrainingNode, chess.Board, list[TrainingNode]]] = []
        for root, source in zip(roots, boards, strict=True):
            if root.proven is not None:
                continue
            board = source.copy(stack=True)
            node = root
            path = [root]
            while node.expanded and node.children and node.proven is None:
                move, node = select_child(node)
                board.push(move)
                path.append(node)

            if node.proven is not None:
                backpropagate(path, node.proven)
                continue
            result = search_terminal_value(board)
            if result is not None:
                node.proven = result
                backpropagate(path, node.proven)
            else:
                pending.append((node, board, path))

        if pending:
            leaf_boards = [item[1] for item in pending]
            leaf_logits, leaf_values = predict_batch(model, leaf_boards, device)
            for item, logits, value in zip(pending, leaf_logits, leaf_values, strict=True):
                node, board, path = item
                expand_node(node, board, logits, False, rng)
                backpropagate(path, float(value))

    policies: list[np.ndarray] = []
    for root, board in zip(roots, boards, strict=True):
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        if root.proven is not None:
            solved = [
                move
                for move, child in root.children.items()
                if child.proven is not None and -child.proven == root.proven
            ]
            for move in solved:
                policy[action_index(board, move)] = 1.0 / len(solved)
            policies.append(policy)
            continue
        total = sum(child.visits for child in root.children.values())
        if total:
            for move, child in root.children.items():
                policy[action_index(board, move)] = child.visits / total
        else:
            moves = list(board.legal_moves)
            for move in moves:
                policy[action_index(board, move)] = 1.0 / len(moves)
        policies.append(policy)
    return policies


def sample_move(
    board: chess.Board, policy: np.ndarray, temperature: float, rng: np.random.Generator
) -> chess.Move:
    moves = list(board.legal_moves)
    visits = np.asarray([policy[action_index(board, move)] for move in moves], dtype=np.float64)
    if not np.isfinite(visits).all() or visits.sum() <= 0.0:
        return moves[int(rng.integers(len(moves)))]
    if temperature <= 0.05:
        return moves[int(visits.argmax())]
    probabilities = np.power(np.maximum(visits, 1e-12), 1.0 / temperature)
    probabilities /= probabilities.sum()
    return moves[int(rng.choice(len(moves), p=probabilities))]


def legal_action_indices(board: chess.Board) -> np.ndarray:
    return np.asarray([action_index(board, move) for move in board.legal_moves], dtype=np.int64)


def finish_game(
    game: SelfPlayGame,
    winner: chess.Color | None,
    *,
    truncated: bool = False,
) -> list[Experience]:
    return [
        Experience(
            state,
            policy,
            0.0 if winner is None else (1.0 if winner == turn else -1.0),
            legal_actions,
            0.0 if truncated else 1.0,
        )
        for state, policy, legal_actions, turn in game.trajectory
    ]


def self_play(
    model: AlphaZeroLite,
    game_count: int,
    simulations: int,
    max_plies: int,
    device: torch.device,
    rng: np.random.Generator,
    log_directory: Path | None = None,
    starting_fens: tuple[str, ...] = (),
) -> list[Experience]:
    if starting_fens:
        active = [
            SelfPlayGame(board=chess.Board(starting_fens[index % len(starting_fens)]))
            for index in rng.permutation(game_count)
        ]
    else:
        active = [SelfPlayGame() for _ in range(game_count)]
    examples: list[Experience] = []
    started_at = time.monotonic()
    completed = 0
    reported = 0
    terminations: dict[str, int] = {}
    if log_directory is not None:
        log_directory.mkdir(parents=True, exist_ok=True)

    model.eval()
    while active:
        policies = batched_search(model, [game.board for game in active], simulations, device, rng)
        continuing: list[SelfPlayGame] = []
        for game, policy in zip(active, policies, strict=True):
            game.trajectory.append(
                (
                    encode_board(game.board),
                    policy,
                    legal_action_indices(game.board),
                    game.board.turn,
                )
            )
            temperature = 1.0 if game.plies < 20 else 0.05
            game.board.push(sample_move(game.board, policy, temperature, rng))
            game.plies += 1

            outcome = game.board.outcome(claim_draw=True)
            if outcome is not None:
                examples.extend(finish_game(game, outcome.winner))
                completed += 1
                termination = outcome.termination.name.lower()
            elif game.plies >= max_plies or game.board.ply() >= PLY_CAP:
                truncated = game.board.ply() < PLY_CAP
                examples.extend(finish_game(game, None, truncated=truncated))
                completed += 1
                termination = "truncated" if truncated else "ply_cap"
            else:
                continuing.append(game)
                continue
            terminations[termination] = terminations.get(termination, 0) + 1
            if log_directory is not None:
                pgn = chess.pgn.Game.from_board(game.board)
                pgn.headers["Result"] = outcome.result() if outcome is not None else "*"
                if termination == "ply_cap":
                    pgn.headers["Result"] = "1/2-1/2"
                pgn.headers["Termination"] = termination
                (log_directory / f"{completed:03d}.pgn").write_text(str(pgn) + "\n")
        active = continuing
        if completed > reported and (completed % 2 == 0 or not active):
            elapsed = time.monotonic() - started_at
            print(
                f"self-play {completed:>3}/{game_count} games, "
                f"{len(examples):>5} positions ({elapsed:.1f}s)",
                flush=True,
            )
            reported = completed
    print(f"self-play terminations: {terminations}", flush=True)
    return examples


def train_epochs(
    model: AlphaZeroLite,
    examples: list[Experience],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
    freeze_batchnorm: bool = False,
    value_loss_weight: float = 1.0,
) -> None:
    if not examples:
        return
    model.train()
    if freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    indices = np.arange(len(examples))
    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        total_policy = 0.0
        total_value = 0.0
        batches = 0
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            states: list[np.ndarray] = []
            policies: list[np.ndarray] = []
            legal_actions: list[np.ndarray | None] = []
            for index in selected:
                example = examples[index]
                # File reflection is a chess symmetry only after all castling rights are gone.
                if not example.state[12:16].any() and rng.random() < 0.5:
                    states.append(mirror_state(example.state))
                    policies.append(mirror_policy(example.policy))
                    legal_actions.append(
                        None
                        if example.legal_actions is None
                        else MIRROR_ACTION_INDICES[example.legal_actions]
                    )
                else:
                    states.append(example.state)
                    policies.append(example.policy)
                    legal_actions.append(example.legal_actions)
            state_batch = torch.from_numpy(np.stack(states)).to(device)
            target_policy = torch.from_numpy(np.stack(policies)).to(device)
            target_value = torch.tensor(
                [examples[index].value for index in selected], dtype=torch.float32, device=device
            )

            policy, value = model(state_batch)
            legal_logits = mask_policy_logits(policy, legal_actions)
            policy_loss = -(target_policy * legal_logits.log_softmax(1)).sum(1).mean()
            value_weights = torch.tensor(
                [examples[index].value_weight for index in selected],
                dtype=torch.float32,
                device=device,
            )
            value_errors = nn.functional.mse_loss(value, target_value, reduction="none")
            value_loss = (value_errors * value_weights).sum() / value_weights.sum().clamp_min(1)
            loss = policy_loss + value_loss_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_policy += float(policy_loss.detach().cpu())
            total_value += float(value_loss.detach().cpu())
            batches += 1
        print(
            f"epoch {epoch:>2}/{epochs}: policy={total_policy / batches:.4f} "
            f"value={total_value / batches:.4f}",
            flush=True,
        )


def mask_policy_logits(policy: Tensor, action_sets: list[np.ndarray | None]) -> Tensor:
    """Mask whole batches at once to avoid thousands of tiny accelerator operations."""
    mask = np.ones(tuple(policy.shape), dtype=np.bool_)
    for row, actions in enumerate(action_sets):
        if actions is not None:
            mask[row] = False
            mask[row, actions] = True
    return policy.masked_fill(~torch.from_numpy(mask).to(policy.device), -1e9)


def load_dataset(source: Path) -> list[Experience]:
    with np.load(source) as archive:
        states = archive["states"].astype(np.float32)
        policies = archive["policies"].astype(np.float32)
        values = archive["values"].astype(np.float32)
        game_ids = archive["game_ids"] if "game_ids" in archive else np.full(len(states), -1)
        if "legal_actions" in archive and "legal_counts" in archive:
            padded_actions = archive["legal_actions"].astype(np.int64)
            legal_counts = archive["legal_counts"].astype(np.int64)
            action_sets: list[np.ndarray | None] = [
                row[: int(count)] for row, count in zip(padded_actions, legal_counts, strict=True)
            ]
        else:
            action_sets = [None] * len(states)
    return [
        Experience(state, policy, float(value), legal_actions, game_id=int(game_id))
        for state, policy, value, legal_actions, game_id in zip(
            states, policies, values, action_sets, game_ids, strict=True
        )
    ]


def mixed_examples(
    self_play_examples: list[Experience],
    teacher_examples: list[Experience],
    teacher_mix: float,
    rng: np.random.Generator,
) -> list[Experience]:
    if not self_play_examples or not teacher_examples or teacher_mix <= 0.0:
        return self_play_examples
    fraction = min(0.9, teacher_mix)
    teacher_count = max(1, round(len(self_play_examples) * fraction / (1.0 - fraction)))
    if teacher_count >= len(teacher_examples):
        selected = teacher_examples
    else:
        selected = [
            teacher_examples[index]
            for index in rng.choice(len(teacher_examples), teacher_count, replace=False)
        ]
    return [*self_play_examples, *selected]


def save_weights(model: AlphaZeroLite, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(state, destination)
    print(f"saved {destination} ({destination.stat().st_size / 1_000_000:.2f} MB)", flush=True)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.iterations = 1
        args.games = 2
        args.simulations = 2
        args.train_epochs = 1
        args.batch_size = 16
        args.max_plies = 8

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    print(f"training on {device} with torch {torch.__version__}", flush=True)

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

    model = AlphaZeroLite()
    if args.resume and args.out.exists():
        model = model_from_state(torch.load(args.out, map_location="cpu", weights_only=True))
        print(f"resumed {args.out}", flush=True)
    elif not args.smoke and args.initial_weights.exists():
        model = model_from_state(
            torch.load(args.initial_weights, map_location="cpu", weights_only=True)
        )
        print(f"initialized from {args.initial_weights}", flush=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    replay: list[Experience] = []
    teacher_examples: list[Experience] = []
    if args.teacher_dataset is not None:
        teacher_examples = load_dataset(args.teacher_dataset)
        print(f"loaded {len(teacher_examples)} teacher positions", flush=True)
    if teacher_examples and not 0.0 <= args.teacher_mix <= 0.9:
        raise ValueError("--teacher-mix must be between 0 and 0.9")

    for iteration in range(1, args.iterations + 1):
        print(f"self-play iteration {iteration}/{args.iterations}", flush=True)
        generated = self_play(
            model,
            args.games,
            args.simulations,
            args.max_plies,
            device,
            rng,
            log_directory=args.out.parent / f"selfplay-{iteration:03d}",
            starting_fens=starting_fens,
        )
        replay.extend(generated)
        if len(replay) > args.replay_size:
            replay = replay[-args.replay_size :]
        training_examples = mixed_examples(replay, teacher_examples, args.teacher_mix, rng)
        print(
            f"training pool: {len(replay)} self-play + "
            f"{len(training_examples) - len(replay)} teacher",
            flush=True,
        )
        train_epochs(
            model,
            training_examples,
            optimizer,
            device,
            args.train_epochs,
            args.batch_size,
            rng,
            freeze_batchnorm=True,
            value_loss_weight=args.value_loss_weight,
        )
        save_weights(model, args.out)


if __name__ == "__main__":
    main()
