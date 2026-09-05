"""The pre-AlphaZero basic agent, retained as a regression opponent."""

import math
import random

import chess

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
MATE = 1_000_000


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"
    if len(moves) == 1:
        return moves[0].uci()

    depth = 3 if time_left_ms > 5_000 and len(moves) <= 20 else 2
    best_score = -math.inf
    best_moves: list[chess.Move] = []
    for move in ordered_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, -math.inf, math.inf)
        board.pop()
        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)
    return random.choice(best_moves).uci()


def negamax(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is None:
            return 0.0
        return -MATE - depth
    if depth == 0:
        return float(evaluate(board))

    best = -math.inf
    for move in ordered_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


def evaluate(board: chess.Board) -> int:
    side = board.turn
    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type] + positional_bonus(piece, square)
        score += value if piece.color == side else -value
    if len(board.pieces(chess.BISHOP, side)) >= 2:
        score += 30
    if len(board.pieces(chess.BISHOP, not side)) >= 2:
        score -= 30
    score += 2 * board.legal_moves.count()
    if board.is_check():
        score -= 40
    return score


def positional_bonus(piece: chess.Piece, square: chess.Square) -> int:
    rank = chess.square_rank(square)
    own_rank = rank if piece.color == chess.WHITE else 7 - rank
    center = center_bonus(square)
    if piece.piece_type == chess.PAWN:
        return own_rank * 8 + center * 2
    if piece.piece_type == chess.KNIGHT:
        return center * 12
    if piece.piece_type == chess.BISHOP:
        return center * 8
    if piece.piece_type == chess.ROOK:
        return own_rank * 3
    if piece.piece_type == chess.QUEEN:
        return center * 4
    return 0


def center_bonus(square: chess.Square) -> int:
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    file_distance = min(abs(file_index - 3), abs(file_index - 4))
    rank_distance = min(abs(rank_index - 3), abs(rank_index - 4))
    return 6 - file_distance - rank_distance


def ordered_moves(board: chess.Board) -> list[chess.Move]:
    moves = list(board.legal_moves)
    random.shuffle(moves)
    moves.sort(key=lambda move: move_priority(board, move), reverse=True)
    return moves


def move_priority(board: chess.Board, move: chess.Move) -> int:
    score = 0
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        victim = PIECE_VALUE[chess.PAWN] if captured is None else PIECE_VALUE[captured.piece_type]
        attacker = board.piece_at(move.from_square)
        attacker_value = PIECE_VALUE[attacker.piece_type] if attacker is not None else 0
        score += 10 * victim - attacker_value
    if move.promotion is not None:
        score += PIECE_VALUE[move.promotion]
    board.push(move)
    if board.is_checkmate():
        score += MATE
    elif board.is_check():
        score += 50
    board.pop()
    return score
