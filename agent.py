"""AlphaZero-lite submission agent: neural policy/value guidance with PUCT search."""

from __future__ import annotations

import math
import time
from contextlib import suppress
from pathlib import Path
from typing import cast

import chess
import numpy as np
import torch
from torch import Tensor

from az_model import AlphaZeroLite, action_index, encode_board

torch.set_num_threads(1)
with suppress(RuntimeError):
    torch.set_num_interop_threads(1)

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
MATE_SCORE = 1_000_000
MAX_SIMULATIONS = 256
PUCT = 1.55


class SearchNode:
    __slots__ = ("children", "expanded", "prior", "value_sum", "visits")

    def __init__(self, prior: float = 0.0) -> None:
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[chess.Move, SearchNode] = {}
        self.expanded = False

    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


_MODEL = AlphaZeroLite()
_MODEL_READY = False
_WEIGHT_PATH = Path(__file__).with_name("weights") / "az_lite.pt"
try:
    loaded: object = torch.load(_WEIGHT_PATH, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError("checkpoint is not a state dictionary")
    _MODEL.load_state_dict(cast(dict[str, Tensor], loaded))
    _MODEL.eval()
    _MODEL_READY = True
except (OSError, RuntimeError, TypeError, ValueError) as error:
    print(f"AlphaZero-lite weights unavailable, using classical fallback: {error}")

_GAME_BOARD: chess.Board | None = None


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move within a conservative share of the game clock."""
    global _GAME_BOARD

    board = _restore_history(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"
    if len(moves) == 1:
        chosen = moves[0]
    else:
        mate = _mate_in_one(board, moves)
        if mate is not None:
            chosen = mate
        else:
            budget_ms = _move_budget_ms(time_left_ms)
            deadline = time.perf_counter() + budget_ms / 1000.0
            chosen = (
                _mcts_move(board, deadline) if _MODEL_READY else _classical_move(board, deadline)
            )

    board.push(chosen)
    _GAME_BOARD = board
    return chosen.uci()


def _restore_history(fen: str) -> chess.Board:
    """Keep repetition history when the new FEN follows the previous position."""
    target = chess.Board(fen)
    if _GAME_BOARD is None:
        return target
    for reply in _GAME_BOARD.legal_moves:
        candidate = _GAME_BOARD.copy(stack=True)
        candidate.push(reply)
        if candidate.fen() == target.fen():
            return candidate
    return target


def _move_budget_ms(time_left_ms: int) -> float:
    if time_left_ms < 750:
        return max(12.0, time_left_ms * 0.04)
    if time_left_ms < 3_000:
        return min(80.0, time_left_ms * 0.035)
    return min(1_500.0, max(80.0, time_left_ms / 60.0))


def _mcts_move(board: chess.Board, deadline: float) -> chess.Move:
    root = SearchNode()
    _expand(root, board)
    simulations = 0

    while simulations < MAX_SIMULATIONS and time.perf_counter() < deadline:
        position = board.copy(stack=True)
        node = root
        path = [node]

        while node.expanded and node.children:
            move, node = _select_child(node)
            position.push(move)
            path.append(node)

        outcome = position.outcome(claim_draw=True)
        if outcome is None:
            leaf_value = _expand(node, position)
        elif outcome.winner is None:
            leaf_value = 0.0
        else:
            leaf_value = 1.0 if outcome.winner == position.turn else -1.0

        _backpropagate(path, leaf_value)
        simulations += 1

    ranked = sorted(
        root.children.items(),
        key=lambda item: (item[1].visits, item[1].prior),
        reverse=True,
    )
    safe = [item for item in ranked if not _allows_mate_in_one(board, item[0])]
    return (safe or ranked)[0][0]


def _select_child(node: SearchNode) -> tuple[chess.Move, SearchNode]:
    parent_scale = math.sqrt(max(1, node.visits))

    def score(item: tuple[chess.Move, SearchNode]) -> float:
        child = item[1]
        exploitation = -child.value()
        exploration = PUCT * child.prior * parent_scale / (1 + child.visits)
        return exploitation + exploration

    return max(node.children.items(), key=score)


def _expand(node: SearchNode, board: chess.Board) -> float:
    logits, value = _predict(board)
    moves = list(board.legal_moves)
    if not moves:
        node.expanded = True
        return value

    indices = torch.tensor([action_index(board, move) for move in moves], dtype=torch.long)
    network_priors = torch.softmax(logits.index_select(0, indices), dim=0).numpy()
    tactical_logits = np.asarray([_move_priority(board, move) / 180.0 for move in moves])
    tactical_logits -= tactical_logits.max()
    tactical_priors = np.exp(np.clip(tactical_logits, -20.0, 0.0))
    tactical_priors /= tactical_priors.sum()
    priors = 0.88 * network_priors + 0.12 * tactical_priors

    node.children = {
        move: SearchNode(float(prior)) for move, prior in zip(moves, priors, strict=True)
    }
    node.expanded = True
    return value


def _predict(board: chess.Board) -> tuple[Tensor, float]:
    state = torch.from_numpy(encode_board(board)).unsqueeze(0)
    with torch.inference_mode():
        policy, value = _MODEL(state)
    return policy[0], float(value[0])


def _backpropagate(path: list[SearchNode], leaf_value: float) -> None:
    value = leaf_value
    for node in reversed(path):
        node.visits += 1
        node.value_sum += value
        value = -value


def _mate_in_one(board: chess.Board, moves: list[chess.Move]) -> chess.Move | None:
    for move in moves:
        if not board.gives_check(move):
            continue
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return move
    return None


def _allows_mate_in_one(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    replies = list(board.legal_moves)
    allows_mate = _mate_in_one(board, replies) is not None
    board.pop()
    return allows_mate


def _classical_move(board: chess.Board, deadline: float) -> chess.Move:
    best_move = next(iter(board.legal_moves))
    best_score = -math.inf
    ordered = sorted(board.legal_moves, key=lambda item: _move_priority(board, item), reverse=True)
    for move in ordered:
        if time.perf_counter() >= deadline:
            break
        board.push(move)
        score = -_negamax(board, 1, -math.inf, math.inf, deadline)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def _negamax(board: chess.Board, depth: int, alpha: float, beta: float, deadline: float) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is None:
            return 0.0
        return -MATE_SCORE - depth
    if depth == 0 or time.perf_counter() >= deadline:
        return float(_evaluate(board))

    best = -math.inf
    ordered = sorted(board.legal_moves, key=lambda item: _move_priority(board, item), reverse=True)
    for move in ordered:
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha, deadline)
        board.pop()
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta or time.perf_counter() >= deadline:
            break
    return best


def _evaluate(board: chess.Board) -> int:
    side = board.turn
    material = sum(
        value * (len(board.pieces(piece_type, side)) - len(board.pieces(piece_type, not side)))
        for piece_type, value in PIECE_VALUE.items()
    )
    return material + 3 * board.legal_moves.count() - (45 if board.is_check() else 0)


def _move_priority(board: chess.Board, move: chess.Move) -> int:
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
