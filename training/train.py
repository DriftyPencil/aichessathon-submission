"""Train the compact policy/value network with a search warm-start and self-play."""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn

from az_model import POLICY_SIZE, AlphaZeroLite, action_index, encode_board

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
PUCT = 1.55


@dataclass(slots=True)
class Experience:
    state: np.ndarray
    policy: np.ndarray
    value: float


@dataclass(slots=True)
class TrainingNode:
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: dict[chess.Move, TrainingNode] = field(default_factory=dict)

    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class SelfPlayGame:
    board: chess.Board = field(default_factory=chess.Board)
    trajectory: list[tuple[np.ndarray, np.ndarray, chess.Color]] = field(default_factory=list)
    plies: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("weights/az_lite.pt"))
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--bootstrap-positions", type=int, default=5_000)
    parser.add_argument("--bootstrap-epochs", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "mps" or (name == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def static_evaluate(board: chess.Board) -> int:
    side = board.turn
    score = 0
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        own_rank = rank if piece.color == chess.WHITE else 7 - rank
        file_index = chess.square_file(square)
        center = (
            6 - min(abs(file_index - 3), abs(file_index - 4)) - min(abs(rank - 3), abs(rank - 4))
        )
        positional = 0
        if piece.piece_type == chess.PAWN:
            positional = own_rank * 8 + center * 2
        elif piece.piece_type == chess.KNIGHT:
            positional = center * 12
        elif piece.piece_type == chess.BISHOP:
            positional = center * 8
        elif piece.piece_type == chess.ROOK:
            positional = own_rank * 3
        value = PIECE_VALUE[piece.piece_type] + positional
        score += value if piece.color == side else -value
    score += 2 * board.legal_moves.count()
    if board.is_check():
        score -= 45
    return score


def teacher_negamax(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is None:
            return 0.0
        return -10_000.0 - depth
    if depth == 0:
        return float(static_evaluate(board))

    best = -math.inf
    moves = sorted(board.legal_moves, key=lambda move: move_priority(board, move), reverse=True)
    for move in moves:
        board.push(move)
        score = -teacher_negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


def teacher_target(board: chess.Board) -> tuple[np.ndarray, float, list[chess.Move], np.ndarray]:
    moves = list(board.legal_moves)
    scores = []
    for move in moves:
        board.push(move)
        scores.append(-teacher_negamax(board, 1, -math.inf, math.inf))
        board.pop()

    score_array = np.asarray(scores, dtype=np.float32)
    scaled = np.clip((score_array - score_array.max()) / 110.0, -20.0, 0.0)
    move_policy = np.exp(scaled)
    move_policy /= move_policy.sum()
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    for move, probability in zip(moves, move_policy, strict=True):
        policy[action_index(board, move)] = probability
    value = math.tanh(float(score_array.max()) / 700.0)
    return policy, value, moves, move_policy


def bootstrap_examples(count: int, rng: np.random.Generator) -> list[Experience]:
    examples: list[Experience] = []
    board = chess.Board()
    episode_plies = 0
    started_at = time.monotonic()

    while len(examples) < count:
        if board.outcome(claim_draw=True) is not None or episode_plies >= 100:
            board = chess.Board()
            episode_plies = 0
            for _ in range(int(rng.integers(0, 7))):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(moves[int(rng.integers(len(moves)))])

        policy, value, moves, move_policy = teacher_target(board)
        examples.append(Experience(encode_board(board), policy, value))
        exploration = 0.18
        sampling = (1.0 - exploration) * move_policy + exploration / len(moves)
        board.push(moves[int(rng.choice(len(moves), p=sampling))])
        episode_plies += 1

        if len(examples) % 250 == 0 or len(examples) == count:
            elapsed = time.monotonic() - started_at
            print(f"bootstrap {len(examples):>5}/{count} positions ({elapsed:.1f}s)")
    return examples


def predict_batch(
    model: AlphaZeroLite, boards: list[chess.Board], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
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
    indices = np.asarray([action_index(board, move) for move in moves])
    legal_logits = logits[indices]
    legal_logits -= legal_logits.max()
    priors = np.exp(np.clip(legal_logits, -20.0, 0.0))
    priors /= priors.sum()

    tactical = np.asarray([move_priority(board, move) / 180.0 for move in moves])
    tactical -= tactical.max()
    tactical_priors = np.exp(np.clip(tactical, -20.0, 0.0))
    tactical_priors /= tactical_priors.sum()
    priors = 0.88 * priors + 0.12 * tactical_priors
    if add_noise:
        noise = rng.dirichlet(np.full(len(moves), 0.3))
        priors = 0.75 * priors + 0.25 * noise

    node.children = {
        move: TrainingNode(prior=float(prior)) for move, prior in zip(moves, priors, strict=True)
    }
    node.expanded = True


def select_child(node: TrainingNode) -> tuple[chess.Move, TrainingNode]:
    parent_scale = math.sqrt(max(1, node.visits))

    def score(item: tuple[chess.Move, TrainingNode]) -> float:
        child = item[1]
        return -child.value() + PUCT * child.prior * parent_scale / (1 + child.visits)

    return max(node.children.items(), key=score)


def backpropagate(path: list[TrainingNode], leaf_value: float) -> None:
    value = leaf_value
    for node in reversed(path):
        node.visits += 1
        node.value_sum += value
        value = -value


def terminal_value(board: chess.Board, outcome: chess.Outcome) -> float:
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def batched_search(
    model: AlphaZeroLite,
    boards: list[chess.Board],
    simulations: int,
    device: torch.device,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    roots = [TrainingNode() for _ in boards]
    root_logits, _ = predict_batch(model, boards, device)
    for root, board, logits in zip(roots, boards, root_logits, strict=True):
        expand_node(root, board, logits, True, rng)

    for _ in range(simulations):
        pending: list[tuple[TrainingNode, chess.Board, list[TrainingNode]]] = []
        for root, source in zip(roots, boards, strict=True):
            board = source.copy(stack=False)
            node = root
            path = [root]
            while node.expanded and node.children:
                move, node = select_child(node)
                board.push(move)
                path.append(node)

            outcome = board.outcome(claim_draw=False)
            if outcome is not None:
                backpropagate(path, terminal_value(board, outcome))
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
        total = sum(child.visits for child in root.children.values())
        if total == 0:
            total = 1
        for move, child in root.children.items():
            policy[action_index(board, move)] = child.visits / total
        policies.append(policy)
    return policies


def sample_move(
    board: chess.Board, policy: np.ndarray, temperature: float, rng: np.random.Generator
) -> chess.Move:
    moves = list(board.legal_moves)
    visits = np.asarray([policy[action_index(board, move)] for move in moves])
    if temperature <= 0.05:
        return moves[int(visits.argmax())]
    probabilities = np.power(np.maximum(visits, 1e-8), 1.0 / temperature)
    probabilities /= probabilities.sum()
    return moves[int(rng.choice(len(moves), p=probabilities))]


def adjudicated_winner(board: chess.Board) -> chess.Color | None:
    balance = sum(
        value
        * (len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK)))
        for piece_type, value in PIECE_VALUE.items()
    )
    if balance > 0:
        return chess.WHITE
    if balance < 0:
        return chess.BLACK
    return None


def finish_game(game: SelfPlayGame, winner: chess.Color | None) -> list[Experience]:
    return [
        Experience(state, policy, 0.0 if winner is None else (1.0 if winner == turn else -1.0))
        for state, policy, turn in game.trajectory
    ]


def self_play(
    model: AlphaZeroLite,
    game_count: int,
    simulations: int,
    max_plies: int,
    device: torch.device,
    rng: np.random.Generator,
) -> list[Experience]:
    active = [SelfPlayGame() for _ in range(game_count)]
    examples: list[Experience] = []
    started_at = time.monotonic()
    completed = 0

    model.eval()
    while active:
        policies = batched_search(model, [game.board for game in active], simulations, device, rng)
        continuing: list[SelfPlayGame] = []
        for game, policy in zip(active, policies, strict=True):
            game.trajectory.append((encode_board(game.board), policy, game.board.turn))
            temperature = 1.0 if game.plies < 18 else 0.05
            game.board.push(sample_move(game.board, policy, temperature, rng))
            game.plies += 1

            outcome = game.board.outcome(claim_draw=True)
            if outcome is not None:
                examples.extend(finish_game(game, outcome.winner))
                completed += 1
            elif game.plies >= max_plies:
                examples.extend(finish_game(game, adjudicated_winner(game.board)))
                completed += 1
            else:
                continuing.append(game)
        active = continuing
        if completed and (completed % 2 == 0 or not active):
            elapsed = time.monotonic() - started_at
            print(
                f"self-play {completed:>3}/{game_count} games, "
                f"{len(examples):>5} positions ({elapsed:.1f}s)"
            )
    return examples


def train_epochs(
    model: AlphaZeroLite,
    examples: list[Experience],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
) -> None:
    model.train()
    indices = np.arange(len(examples))
    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        total_policy = 0.0
        total_value = 0.0
        batches = 0
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            states = torch.from_numpy(np.stack([examples[index].state for index in selected])).to(
                device
            )
            target_policy = torch.from_numpy(
                np.stack([examples[index].policy for index in selected])
            ).to(device)
            target_value = torch.tensor(
                [examples[index].value for index in selected], dtype=torch.float32, device=device
            )

            policy, value = model(states)
            policy_loss = -(target_policy * torch.log_softmax(policy, dim=1)).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(value, target_value)
            loss = policy_loss + value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_policy += float(policy_loss.detach().cpu())
            total_value += float(value_loss.detach().cpu())
            batches += 1
        print(
            f"epoch {epoch:>2}/{epochs}: policy={total_policy / batches:.4f} "
            f"value={total_value / batches:.4f}"
        )


def move_priority(board: chess.Board, move: chess.Move) -> int:
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        victim_value = PIECE_VALUE[chess.PAWN] if victim is None else PIECE_VALUE[victim.piece_type]
        attacker = board.piece_at(move.from_square)
        attacker_value = PIECE_VALUE[attacker.piece_type] if attacker is not None else 0
        score += victim_value - attacker_value // 12
    if move.promotion is not None:
        score += PIECE_VALUE[move.promotion]
    if board.gives_check(move):
        score += 55
    return score


def save_weights(model: AlphaZeroLite, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(state, destination)
    print(f"saved {destination} ({destination.stat().st_size / 1_000_000:.2f} MB)")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.bootstrap_positions = 24
        args.bootstrap_epochs = 1
        args.iterations = 1
        args.games = 2
        args.simulations = 2
        args.train_epochs = 1
        args.batch_size = 16
        args.max_plies = 6

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    print(f"training on {device} with torch {torch.__version__}")

    model = AlphaZeroLite().to(device)
    if args.resume and args.out.exists():
        model.load_state_dict(torch.load(args.out, map_location=device, weights_only=True))
        print(f"resumed {args.out}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    replay = bootstrap_examples(args.bootstrap_positions, rng)
    train_epochs(model, replay, optimizer, device, args.bootstrap_epochs, args.batch_size, rng)
    save_weights(model, args.out)

    for iteration in range(1, args.iterations + 1):
        print(f"self-play iteration {iteration}/{args.iterations}")
        generated = self_play(model, args.games, args.simulations, args.max_plies, device, rng)
        replay.extend(generated)
        replay = replay[-20_000:]
        train_epochs(model, replay, optimizer, device, args.train_epochs, args.batch_size, rng)
        save_weights(model, args.out)


if __name__ == "__main__":
    main()
