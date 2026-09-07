"""Neural policy/value agent with iterative alpha-beta and tactical quiescence."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort

from chess_encoding import POLICY_SIZE, action_index, encode_board

MAX_DEPTH = 16
QUIESCENCE_DEPTH = 6
MATE_VALUE = 2.0
EVALUATION_CACHE_BYTES = 896 * 1024 * 1024
MAX_CACHED_EVALUATIONS = EVALUATION_CACHE_BYTES // (POLICY_SIZE * 4 + 1152)
MAX_MOVE_HINTS = 500_000


class SearchTimeout(Exception):
    pass


_OPTIONS = ort.SessionOptions()
_OPTIONS.intra_op_num_threads = 1
_OPTIONS.inter_op_num_threads = 1
_OPTIONS.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
_OPTIONS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_MODEL_PATH = Path(__file__).with_name("weights") / "az_lite-int8.onnx"
try:
    _INFERENCE = ort.InferenceSession(
        _MODEL_PATH,
        sess_options=_OPTIONS,
        providers=["CPUExecutionProvider"],
    )
    _INFERENCE.run(None, {"state": np.zeros((1, 18, 8, 8), dtype=np.float32)})
except (OSError, RuntimeError, ValueError) as error:
    raise RuntimeError(f"could not load trained AlphaZero-lite ONNX model: {error}") from error

_GAME_BOARD: chess.Board | None = None
_EVALUATIONS: OrderedDict[tuple[int, ...], tuple[np.ndarray, float]] = OrderedDict()
_MOVE_HINTS: OrderedDict[tuple[int, ...], chess.Move] = OrderedDict()
_DEADLINE = 0.0
_NODES = 0


def get_move(fen: str, time_left_ms: int) -> str:
    """Return the best move from the last fully completed neural alpha-beta iteration."""
    global _DEADLINE, _GAME_BOARD, _NODES
    started = time.perf_counter()
    board = _restore_history(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"
    if len(moves) == 1 or time_left_ms < 20:
        chosen = moves[0]
    else:
        _DEADLINE = started + _move_budget_ms(time_left_ms) / 1000.0
        _NODES = 0
        chosen = _iterative_move(board, moves)
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
        return max(0.0, min(time_left_ms * 0.03, time_left_ms - 10.0))
    if time_left_ms <= 3_000:
        return min(200.0, max(35.0, time_left_ms * 0.08))
    return min(4_000.0, max(80.0, time_left_ms / 30.0))


def _iterative_move(board: chess.Board, legal_moves: list[chess.Move]) -> chess.Move:
    best = legal_moves[0]
    for depth in range(1, MAX_DEPTH + 1):
        try:
            _, candidate = _search_root(board, depth, best)
        except SearchTimeout:
            break
        best = candidate
    return best


def _search_root(
    board: chess.Board, depth: int, preferred: chess.Move
) -> tuple[float, chess.Move]:
    alpha = -math.inf
    beta = math.inf
    best_move = preferred
    logits, _ = _predict(board)
    moves = _ordered_moves(board, logits, preferred=preferred)
    for move in moves:
        _check_time()
        board.push(move)
        try:
            score = -_negamax(board, depth - 1, -beta, -alpha, 1)
        finally:
            board.pop()
        if score > alpha:
            alpha = score
            best_move = move
    _remember_move(_input_key(board), best_move)
    return alpha, best_move


def _negamax(board: chess.Board, depth: int, alpha: float, beta: float, ply: int) -> float:
    global _NODES
    _NODES += 1
    _check_time()
    terminal = _terminal_value(board, ply)
    if terminal is not None:
        return terminal
    if depth <= 0:
        return _quiescence(board, alpha, beta, QUIESCENCE_DEPTH, ply)
    key = _input_key(board)
    logits, _ = _predict(board)
    best = -math.inf
    best_move: chess.Move | None = None
    for move in _ordered_moves(board, logits, preferred=_MOVE_HINTS.get(key)):
        board.push(move)
        try:
            score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1)
        finally:
            board.pop()
        if score > best:
            best = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    if best_move is not None:
        _remember_move(key, best_move)
    return best


def _quiescence(
    board: chess.Board, alpha: float, beta: float, depth: int, ply: int
) -> float:
    global _NODES
    _NODES += 1
    _check_time()
    terminal = _terminal_value(board, ply)
    if terminal is not None:
        return terminal
    logits, stand_pat = _predict(board)
    if stand_pat >= beta:
        return stand_pat
    best = stand_pat
    if stand_pat > alpha:
        alpha = stand_pat
    if depth <= 0:
        return best
    moves = [move for move in board.legal_moves if board.is_capture(move) or move.promotion]
    for move in _ordered_moves(board, logits, moves=moves):
        board.push(move)
        try:
            score = -_quiescence(board, -beta, -alpha, depth - 1, ply + 1)
        finally:
            board.pop()
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best


def _terminal_value(board: chess.Board, ply: int) -> float | None:
    if board.ply() >= 600:
        return 0.0
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    magnitude = MATE_VALUE - min(ply, 1000) * 1e-5
    return magnitude if outcome.winner == board.turn else -magnitude


def _ordered_moves(
    board: chess.Board,
    logits: np.ndarray,
    *,
    preferred: chess.Move | None = None,
    moves: list[chess.Move] | None = None,
) -> list[chess.Move]:
    ordered = list(board.legal_moves) if moves is None else moves
    ordered.sort(
        key=lambda move: (move == preferred, float(logits[action_index(board, move)])),
        reverse=True,
    )
    return ordered


def _check_time() -> None:
    if time.perf_counter() >= _DEADLINE:
        raise SearchTimeout


def _input_key(board: chess.Board) -> tuple[int, ...]:
    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        int(board.turn),
        board.castling_rights,
        -1 if board.ep_square is None else board.ep_square,
        min(board.halfmove_clock, 100),
    )


def _predict(board: chess.Board) -> tuple[np.ndarray, float]:
    key = _input_key(board)
    cached = _EVALUATIONS.get(key)
    if cached is not None:
        _EVALUATIONS.move_to_end(key)
        return cached
    state = encode_board(board)[None, ...]
    policies, values = _INFERENCE.run(None, {"state": state})
    prediction = policies[0].copy(), float(values[0])
    if len(_EVALUATIONS) >= MAX_CACHED_EVALUATIONS:
        _EVALUATIONS.popitem(last=False)
    _EVALUATIONS[key] = prediction
    return prediction


def _remember_move(key: tuple[int, ...], move: chess.Move) -> None:
    if key in _MOVE_HINTS:
        _MOVE_HINTS.move_to_end(key)
    elif len(_MOVE_HINTS) >= MAX_MOVE_HINTS:
        _MOVE_HINTS.popitem(last=False)
    _MOVE_HINTS[key] = move
