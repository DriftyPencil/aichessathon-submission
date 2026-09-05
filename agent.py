"""Pure AlphaZero-lite agent: neural policy/value inference with PUCT search."""

from __future__ import annotations

import math
import time
from contextlib import suppress
from pathlib import Path
from typing import cast

import chess
import torch
from torch import Tensor

from az_model import AlphaZeroLite, action_index, encode_board

torch.set_num_threads(1)
with suppress(RuntimeError):
    torch.set_num_interop_threads(1)

MAX_SIMULATIONS = 16_384
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
_WEIGHT_PATH = Path(__file__).with_name("weights") / "az_lite.pt"
try:
    checkpoint: object = torch.load(_WEIGHT_PATH, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint is not a state dictionary")
    _MODEL.load_state_dict(cast(dict[str, Tensor], checkpoint))
    _MODEL.eval()
    with torch.inference_mode():
        _MODEL(torch.from_numpy(encode_board(chess.Board())).unsqueeze(0))
except (OSError, RuntimeError, TypeError, ValueError) as error:
    raise RuntimeError(f"could not load trained AlphaZero-lite weights: {error}") from error

_GAME_BOARD: chess.Board | None = None


def get_move(fen: str, time_left_ms: int) -> str:
    """Return the highest-visit legal move from neural PUCT search."""
    global _GAME_BOARD

    board = _restore_history(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"
    if len(moves) == 1:
        chosen = moves[0]
    else:
        deadline = time.perf_counter() + _move_budget_ms(time_left_ms) / 1000.0
        chosen = _mcts_move(board, deadline)

    board.push(chosen)
    _GAME_BOARD = board
    return chosen.uci()


def _restore_history(fen: str) -> chess.Board:
    target = chess.Board(fen)
    if _GAME_BOARD is None:
        return target
    target_fen = target.fen()
    for reply in _GAME_BOARD.legal_moves:
        candidate = _GAME_BOARD.copy(stack=True)
        candidate.push(reply)
        if candidate.fen() == target_fen:
            return candidate
    return target


def _move_budget_ms(time_left_ms: int) -> float:
    if time_left_ms <= 500:
        return max(8.0, time_left_ms * 0.03)
    if time_left_ms <= 3_000:
        return min(200.0, max(35.0, time_left_ms * 0.08))
    return min(4_000.0, max(80.0, time_left_ms / 30.0))


def _mcts_move(board: chess.Board, deadline: float) -> chess.Move:
    root = SearchNode()
    _expand_node(root, board)
    simulations = 0

    while simulations < MAX_SIMULATIONS and time.perf_counter() < deadline:
        position = board.copy(stack=True)
        node = root
        path = [root]

        while node.expanded and node.children:
            move, node = _select_child(node)
            position.push(move)
            path.append(node)

        outcome = position.outcome(claim_draw=True)
        if outcome is None:
            leaf_value = _expand_node(node, position)
        elif outcome.winner is None:
            leaf_value = 0.0
        else:
            leaf_value = 1.0 if outcome.winner == position.turn else -1.0

        _backpropagate(path, leaf_value)
        simulations += 1

    if not root.children:
        return next(iter(board.legal_moves))
    return max(root.children.items(), key=lambda item: (item[1].visits, item[1].prior))[0]


def _select_child(node: SearchNode) -> tuple[chess.Move, SearchNode]:
    parent_scale = math.sqrt(max(1, node.visits))

    def score(item: tuple[chess.Move, SearchNode]) -> float:
        child = item[1]
        exploitation = -child.value()
        exploration = PUCT * child.prior * parent_scale / (1 + child.visits)
        return exploitation + exploration

    return max(node.children.items(), key=score)


def _expand_node(node: SearchNode, board: chess.Board) -> float:
    logits, value = _predict(board)
    moves = list(board.legal_moves)
    if not moves:
        node.expanded = True
        return value

    indices = torch.tensor([action_index(board, move) for move in moves], dtype=torch.long)
    priors = torch.softmax(logits.index_select(0, indices), dim=0).tolist()
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
