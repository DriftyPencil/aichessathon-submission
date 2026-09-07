"""Torch-free board and move encodings shared with the ONNX competition runtime."""

from __future__ import annotations

import chess
import numpy as np

INPUT_PLANES = 18
POLICY_PLANES = 73
POLICY_SIZE = 64 * POLICY_PLANES

_QUEEN_DIRECTIONS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
_KNIGHT_DIRECTIONS = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
_UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
_PROMOTION_SLOTS = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}


def _oriented_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    return square if turn == chess.WHITE else chess.square_mirror(square)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a position from the current player's point of view."""
    encoded = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    turn = board.turn
    for square, piece in board.piece_map().items():
        oriented = _oriented_square(square, turn)
        offset = 0 if piece.color == turn else 6
        channel = offset + piece.piece_type - 1
        encoded[channel, chess.square_rank(oriented), chess.square_file(oriented)] = 1.0
    castling = (
        board.has_kingside_castling_rights(turn),
        board.has_queenside_castling_rights(turn),
        board.has_kingside_castling_rights(not turn),
        board.has_queenside_castling_rights(not turn),
    )
    for channel, available in enumerate(castling, start=12):
        if available:
            encoded[channel].fill(1.0)
    if board.ep_square is not None:
        square = _oriented_square(board.ep_square, turn)
        encoded[16, chess.square_rank(square), chess.square_file(square)] = 1.0
    encoded[17].fill(min(board.halfmove_clock, 100) / 100.0)
    return encoded


def _calculate_action_index(turn: chess.Color, move: chess.Move) -> int:
    from_square = _oriented_square(move.from_square, turn)
    to_square = _oriented_square(move.to_square, turn)
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    file_delta = chess.square_file(to_square) - from_file
    rank_delta = chess.square_rank(to_square) - from_rank
    if move.promotion in _UNDERPROMOTIONS:
        promotion_direction = file_delta + 1
        promotion = _UNDERPROMOTIONS.index(move.promotion)
        plane = 64 + promotion_direction * 3 + promotion
        return from_square * POLICY_PLANES + plane
    knight_delta = (file_delta, rank_delta)
    if knight_delta in _KNIGHT_DIRECTIONS:
        plane = 56 + _KNIGHT_DIRECTIONS.index(knight_delta)
        return from_square * POLICY_PLANES + plane
    distance = max(abs(file_delta), abs(rank_delta))
    if distance == 0:
        return -1
    queen_direction = (file_delta // distance, rank_delta // distance)
    if queen_direction not in _QUEEN_DIRECTIONS:
        return -1
    plane = _QUEEN_DIRECTIONS.index(queen_direction) * 7 + distance - 1
    return from_square * POLICY_PLANES + plane


_ACTION_INDICES = np.full((2, 64, 64, 5), -1, dtype=np.int16)
for _turn in (chess.BLACK, chess.WHITE):
    for _source in chess.SQUARES:
        for _target in chess.SQUARES:
            for _promotion, _slot in _PROMOTION_SLOTS.items():
                _ACTION_INDICES[int(_turn), _source, _target, _slot] = _calculate_action_index(
                    _turn, chess.Move(_source, _target, promotion=_promotion)
                )


def action_index(board: chess.Board, move: chess.Move) -> int:
    """Map a move to its policy action using a precomputed orientation table."""
    slot = _PROMOTION_SLOTS.get(move.promotion)
    if slot is None:
        raise ValueError(f"cannot encode promotion: {move.uci()}")
    index = int(_ACTION_INDICES[int(board.turn), move.from_square, move.to_square, slot])
    if index < 0:
        raise ValueError(f"cannot encode move: {move.uci()}")
    return index
