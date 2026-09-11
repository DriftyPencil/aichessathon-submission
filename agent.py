"""Ground-up hard-evaluation chess agent for local development.

Search includes iterative deepening, aspiration
windows, fail-soft PVS/negamax, quiescence, a direct-mapped transposition table,
null-move pruning, late-move reductions, and history/killer move ordering.

``_Searcher(selective=False, use_tt=False)`` disables every speculative pruning
rule and the TT while retaining the same PVS and quiescence core. That exact mode
exists as a correctness oracle for shallow differential tests.
"""

from __future__ import annotations

import time

import chess
import chess.polyglot

_INFINITY = 2_000_000
_MATE_SCORE = 1_000_000
_MAX_PLY = 192
_MATE_TT_THRESHOLD = _MATE_SCORE - _MAX_PLY
_TT_SIZE = 1 << 21
_TT_MASK = _TT_SIZE - 1
_U64_MASK = (1 << 64) - 1
_RULE_50_KEY = 0x9E3779B97F4A7C15
_SELECTIVE_TT_KEY = 0xD1B54A32D192ED03

_EXACT = 0
_LOWER = 1
_UPPER = 2

_PHASE_WEIGHT = (0, 0, 1, 1, 2, 4, 0)

# Tables are indexed by piece type, then by a square from White's perspective.
# Values already include the old rank/file sum and scale factor.
_MG_PIECE_SQUARE: tuple[tuple[int, ...], ...] = (
    (),
    (
        32, 48, 56, 72, 80, 104, 96, 40, 64, 80, 88, 104, 112, 136, 128, 72,
        72, 88, 96, 112, 120, 144, 136, 80, 72, 88, 96, 112, 120, 144, 136, 80,
        88, 104, 112, 128, 136, 160, 152, 96, 112, 128, 136, 152, 160, 184, 176, 120,
        240, 256, 264, 280, 288, 312, 304, 248, 32, 48, 56, 72, 80, 104, 96, 40,
    ),
    (
        392, 424, 440, 448, 448, 448, 440, 416, 416, 448, 464, 472, 472, 472, 464, 440,
        432, 464, 480, 488, 488, 488, 480, 456, 448, 480, 496, 504, 504, 504, 496, 472,
        456, 488, 504, 512, 512, 512, 504, 480, 496, 528, 544, 552, 552, 552, 544, 520,
        464, 496, 512, 520, 520, 520, 512, 488, 280, 312, 328, 336, 336, 336, 328, 304,
    ),
    (
        464, 480, 480, 480, 480, 472, 488, 472, 480, 496, 496, 496, 496, 488, 504, 488,
        480, 496, 496, 496, 496, 488, 504, 488, 480, 496, 496, 496, 496, 488, 504, 488,
        480, 496, 496, 496, 496, 488, 504, 488, 496, 512, 512, 512, 512, 504, 520, 504,
        464, 480, 480, 480, 480, 472, 488, 472, 408, 424, 424, 424, 424, 416, 432, 416,
    ),
    (
        592, 592, 608, 616, 608, 608, 592, 592, 568, 568, 584, 592, 584, 584, 568, 568,
        576, 576, 592, 600, 592, 592, 576, 576, 576, 576, 592, 600, 592, 592, 576, 576,
        600, 600, 616, 624, 616, 616, 600, 600, 632, 632, 648, 656, 648, 648, 632, 632,
        632, 632, 648, 656, 648, 648, 632, 632, 640, 640, 656, 664, 656, 656, 640, 640,
    ),
    (
        1392, 1400, 1400, 1400, 1392, 1400, 1416, 1408,
        1384, 1392, 1392, 1392, 1384, 1392, 1408, 1400,
        1368, 1376, 1376, 1376, 1368, 1376, 1392, 1384,
        1352, 1360, 1360, 1360, 1352, 1360, 1376, 1368,
        1344, 1352, 1352, 1352, 1344, 1352, 1368, 1360,
        1360, 1368, 1368, 1368, 1360, 1368, 1384, 1376,
        1328, 1336, 1336, 1336, 1328, 1336, 1352, 1344,
        1368, 1376, 1376, 1376, 1368, 1376, 1392, 1384,
    ),
    (
        -40, 16, -32, -112, -56, -88, 0, -16, -40, 16, -32, -112, -56, -88, 0, -16,
        -56, 0, -48, -128, -72, -104, -16, -32, -40, 16, -32, -112, -56, -88, 0, -16,
        0, 56, 8, -72, -16, -48, 40, 24, 88, 144, 96, 16, 72, 40, 128, 112,
        72, 128, 80, 0, 56, 24, 112, 96, 32, 88, 40, -40, 16, -16, 72, 56,
    ),
)

_EG_PIECE_SQUARE: tuple[tuple[int, ...], ...] = (
    (),
    (
        120, 112, 96, 80, 88, 88, 96, 88, 176, 168, 152, 136, 144, 144, 152, 144,
        168, 160, 144, 128, 136, 136, 144, 136, 168, 160, 144, 128, 136, 136, 144, 136,
        184, 176, 160, 144, 152, 152, 160, 152, 288, 280, 264, 248, 256, 256, 264, 256,
        392, 384, 368, 352, 360, 360, 368, 360, 120, 112, 96, 80, 88, 88, 96, 88,
    ),
    (
        368, 392, 424, 440, 440, 424, 400, 368, 400, 424, 456, 472, 472, 456, 432, 400,
        424, 448, 480, 496, 496, 480, 456, 424, 456, 480, 512, 528, 528, 512, 488, 456,
        464, 488, 520, 536, 536, 520, 496, 464, 440, 464, 496, 512, 512, 496, 472, 440,
        416, 440, 472, 488, 488, 472, 448, 416, 408, 432, 464, 480, 480, 464, 440, 408,
    ),
    (
        432, 440, 440, 448, 440, 440, 440, 432, 432, 440, 440, 448, 440, 440, 440, 432,
        440, 448, 448, 456, 448, 448, 448, 440, 448, 456, 456, 464, 456, 456, 456, 448,
        456, 464, 464, 472, 464, 464, 464, 456, 448, 456, 456, 464, 456, 456, 456, 448,
        448, 456, 456, 464, 456, 456, 456, 448, 456, 464, 464, 472, 464, 464, 464, 456,
    ),
    (
        824, 824, 824, 824, 824, 824, 824, 816, 816, 816, 816, 816, 816, 816, 816, 808,
        816, 816, 816, 816, 816, 816, 816, 808, 840, 840, 840, 840, 840, 840, 840, 832,
        848, 848, 848, 848, 848, 848, 848, 840, 848, 848, 848, 848, 848, 848, 848, 840,
        856, 856, 856, 856, 856, 856, 856, 848, 856, 856, 856, 856, 856, 856, 856, 848,
    ),
    (
        1384, 1400, 1408, 1424, 1440, 1440, 1424, 1424,
        1400, 1416, 1424, 1440, 1456, 1456, 1440, 1440,
        1448, 1464, 1472, 1488, 1504, 1504, 1488, 1488,
        1496, 1512, 1520, 1536, 1552, 1552, 1536, 1536,
        1528, 1544, 1552, 1568, 1584, 1584, 1568, 1568,
        1520, 1536, 1544, 1560, 1576, 1576, 1560, 1560,
        1528, 1544, 1552, 1568, 1584, 1584, 1568, 1568,
        1496, 1512, 1520, 1536, 1552, 1552, 1536, 1536,
    ),
    (
        0, -16, 0, 0, 0, 0, 0, -48, 0, -16, 0, 0, 0, 0, 0, -48,
        0, -16, 0, 0, 0, 0, 0, -48, 0, -16, 0, 0, 0, 0, 0, -48,
        32, 16, 32, 32, 32, 32, 32, -16, 24, 8, 24, 24, 24, 24, 24, -24,
        16, 0, 16, 16, 16, 16, 16, -32, -48, -64, -48, -48, -48, -48, -48, -96,
    ),
)

_MG_MOBILITY = (0, 0, 0, 6, 3, 3, -10)
_EG_MOBILITY = (0, 0, 0, 7, 4, 3, 0)
_MG_KING_PRESSURE = (0, 0, 9, 16, 36, 23, -114)
_EG_KING_PRESSURE = (0, 0, -2, 0, -10, 18, 0)
_MG_OPEN_FILE = (0, 16, 5, 1, 31, 3, -30)
_EG_OPEN_FILE = (0, 26, -4, 4, 13, 21, 0)

type TTEntry = tuple[int, int, int, int, chess.Move | None]


def _divide_toward_zero(numerator: int, denominator: int) -> int:
    quotient = abs(numerator) // denominator
    return -quotient if numerator < 0 else quotient


_TT: list[TTEntry | None] = [None] * _TT_SIZE
_TT_AGE = [0] * _TT_SIZE
_TT_GENERATION = 0
_HISTORY = [0] * 4096
_GAME_BOARD: chess.Board | None = None


class _SearchTimeout(Exception):
    pass


def _move_key(move: chess.Move) -> int:
    return move.from_square | (move.to_square << 6)


def _tt_key(board: chess.Board, *, selective: bool) -> int:
    """Hash the position and the draw-clock state relevant to its score."""
    position_key = chess.polyglot.zobrist_hash(board)
    rule_50_count = min(board.halfmove_clock, 100)
    rule_50_key = ((rule_50_count + 1) * _RULE_50_KEY) & _U64_MASK
    mode_key = _SELECTIVE_TT_KEY if selective else 0
    return position_key ^ rule_50_key ^ mode_key


def _score_to_tt(score: int, ply: int) -> int:
    """Store mate scores independently of the path used to reach the node."""
    if score >= _MATE_TT_THRESHOLD:
        return score + ply
    if score <= -_MATE_TT_THRESHOLD:
        return score - ply
    return score


def _score_from_tt(score: int, ply: int) -> int:
    if score >= _MATE_TT_THRESHOLD:
        return score - ply
    if score <= -_MATE_TT_THRESHOLD:
        return score + ply
    return score


def _is_search_draw(board: chess.Board, ply: int) -> bool:
    # At the root, a twofold position is still a live position. Treating it as
    # an immediate draw leaves iterative deepening without a selected move.
    if board.is_insufficient_material():
        return True
    if board.halfmove_clock >= 100:
        # Checkmate takes precedence when the mating move also reaches the
        # fifty-move threshold.
        return not board.is_checkmate()
    return ply > 0 and board.is_repetition(2)


def _null_move_safe(board: chess.Board) -> bool:
    """Avoid null-move pruning when the mover has zugzwang-prone material."""
    color = board.turn
    majors = board.pieces_mask(chess.ROOK, color) | board.pieces_mask(chess.QUEEN, color)
    minors = board.pieces_mask(chess.KNIGHT, color) | board.pieces_mask(chess.BISHOP, color)
    return bool(majors) or minors.bit_count() >= 2


def _gives_check(board: chess.Board, move: chess.Move) -> bool:
    """Probe checks without the push/pop performed by ``Board.gives_check``."""
    color = board.turn
    king = board.king(not color)
    if king is None:
        return False

    from_mask = chess.BB_SQUARES[move.from_square]
    to_mask = chess.BB_SQUARES[move.to_square]
    occupied = (board.occupied & ~from_mask) | to_mask
    if board.is_en_passant(move):
        captured_square = move.to_square - 8 if color else move.to_square + 8
        occupied &= ~chess.BB_SQUARES[captured_square]

    piece_type = move.promotion or board.piece_type_at(move.from_square)
    if piece_type == chess.PAWN:
        direct_attacks = chess.BB_PAWN_ATTACKS[color][move.to_square]
    elif piece_type == chess.KNIGHT:
        direct_attacks = chess.BB_KNIGHT_ATTACKS[move.to_square]
    elif piece_type == chess.BISHOP:
        direct_attacks = chess.BB_DIAG_ATTACKS[move.to_square][
            chess.BB_DIAG_MASKS[move.to_square] & occupied
        ]
    elif piece_type == chess.ROOK:
        direct_attacks = (
            chess.BB_RANK_ATTACKS[move.to_square][
                chess.BB_RANK_MASKS[move.to_square] & occupied
            ]
            | chess.BB_FILE_ATTACKS[move.to_square][
                chess.BB_FILE_MASKS[move.to_square] & occupied
            ]
        )
    elif piece_type == chess.QUEEN:
        direct_attacks = (
            chess.BB_DIAG_ATTACKS[move.to_square][
                chess.BB_DIAG_MASKS[move.to_square] & occupied
            ]
            | chess.BB_RANK_ATTACKS[move.to_square][
                chess.BB_RANK_MASKS[move.to_square] & occupied
            ]
            | chess.BB_FILE_ATTACKS[move.to_square][
                chess.BB_FILE_MASKS[move.to_square] & occupied
            ]
        )
    else:
        direct_attacks = chess.BB_KING_ATTACKS[move.to_square]

    if direct_attacks & chess.BB_SQUARES[king]:
        return True
    if board.is_castling(move):
        return board.gives_check(move)
    if not board.is_en_passant(move) and not chess.ray(move.from_square, king):
        return False

    discovered_attackers = board.attackers_mask(color, king, occupied) & ~from_mask
    return bool(discovered_attackers)


def _recover_game_board(fen: str) -> chess.Board:
    """Recover the opponent's ply so repetition checks retain history."""
    global _GAME_BOARD

    requested = chess.Board(fen)
    if _GAME_BOARD is None:
        return requested

    requested_fen = requested.fen()
    for reply in _GAME_BOARD.legal_moves:
        continued = _GAME_BOARD.copy(stack=True)
        continued.push(reply)
        if continued.fen() == requested_fen:
            return continued
    return requested


def _evaluate(board: chess.Board) -> tuple[int, int]:
    """Return a tapered score from the side-to-move perspective and phase."""
    middle_game = 15
    end_game = 0
    phase = 0

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == board.turn else -1
        own_pieces = board.occupied_co[color]
        pawns = board.pieces_mask(chess.PAWN, color)

        enemy_king = board.king(not color)
        king_zone = chess.BB_KING_ATTACKS[enemy_king] if enemy_king is not None else 0

        for piece_type in range(chess.PAWN, chess.KING + 1):
            pieces = board.pieces_mask(piece_type, color)
            phase += _PHASE_WEIGHT[piece_type] * pieces.bit_count()

            while pieces:
                square = chess.lsb(pieces)
                pieces &= pieces - 1
                relative_square = (
                    square if color == chess.WHITE else chess.square_mirror(square)
                )
                middle_game += sign * _MG_PIECE_SQUARE[piece_type][relative_square]
                end_game += sign * _EG_PIECE_SQUARE[piece_type][relative_square]

                same_file_pawns = pawns & chess.BB_FILES[chess.square_file(square)]
                if same_file_pawns & ~chess.BB_SQUARES[square] == 0:
                    middle_game += sign * _MG_OPEN_FILE[piece_type]
                    end_game += sign * _EG_OPEN_FILE[piece_type]

                if piece_type > chess.KNIGHT:
                    attacks = board.attacks_mask(square) & ~own_pieces
                    mobility = attacks.bit_count()
                    king_pressure = (attacks & king_zone).bit_count()
                    middle_game += sign * (
                        _MG_MOBILITY[piece_type] * mobility
                        + _MG_KING_PRESSURE[piece_type] * king_pressure
                    )
                    end_game += sign * (
                        _EG_MOBILITY[piece_type] * mobility
                        + _EG_KING_PRESSURE[piece_type] * king_pressure
                    )

    phase = min(24, phase)
    tapered = _divide_toward_zero(
        middle_game * phase + end_game * (24 - phase),
        24,
    )
    return tapered, phase


class _Searcher:
    def __init__(
        self,
        board: chess.Board,
        time_left_ms: int,
        *,
        selective: bool = True,
        use_tt: bool = True,
    ) -> None:
        self.board = board
        self.selective = selective
        self.use_tt = use_tt
        self.started = time.perf_counter()
        remaining_ms = max(1, time_left_ms)
        reserve_ms = min(50, max(2, remaining_ms // 20))
        self.hard_budget_ms = min(
            max(1, remaining_ms // 6),
            max(1, remaining_ms - reserve_ms),
        )
        self.soft_budget_ms = max(1, self.hard_budget_ms // 5)
        self.nodes = 0
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(_MAX_PLY)]
        self.iteration_root_move: chess.Move | None = None
        self.iteration_root_moves_completed = 0
        self.completed_depth = 0
        self.completed_score = 0

    def _elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0

    def _check_time(self, *, force: bool = False) -> None:
        if (force or self.nodes & 127 == 0) and (self._elapsed_ms() >= self.hard_budget_ms):
            raise _SearchTimeout

    def _store_tt(
        self,
        key: int,
        depth: int,
        score: int,
        bound: int,
        move: chess.Move | None,
        ply: int,
    ) -> None:
        if not self.use_tt:
            return
        index = key & _TT_MASK
        current = _TT[index]
        if current is not None and current[1] > depth:
            same_position = current[0] == key
            current_generation = _TT_AGE[index] == _TT_GENERATION
            if same_position or current_generation:
                return
        _TT[index] = (key, depth, _score_to_tt(score, ply), bound, move)
        _TT_AGE[index] = _TT_GENERATION

    def _move_order_score(
        self,
        move: chess.Move,
        tt_move: chess.Move | None,
        ply: int,
    ) -> int:
        if move == tt_move:
            return 20_000_000
        if move.promotion is not None:
            return 10_000_000 + move.promotion * 100_000
        if self.board.is_capture(move):
            captured = (
                chess.PAWN
                if self.board.is_en_passant(move)
                else self.board.piece_type_at(move.to_square) or chess.PAWN
            )
            attacker = self.board.piece_type_at(move.from_square) or chess.PAWN
            return 8_000_000 + captured * 100_000 - attacker * 1_000
        if _gives_check(self.board, move):
            return 7_500_000

        if not self.selective:
            return 0

        first_killer, second_killer = self.killers[min(ply, _MAX_PLY - 1)]
        if move == first_killer:
            return 7_000_000
        if move == second_killer:
            return 6_900_000
        return _HISTORY[_move_key(move)]

    def _ordered_moves(
        self,
        tt_move: chess.Move | None,
        ply: int,
        *,
        tactical_only: bool,
    ) -> tuple[list[chess.Move], bool]:
        if tactical_only:
            moves = list(self.board.generate_legal_captures())
            promotion_rank = chess.BB_RANK_7 if self.board.turn else chess.BB_RANK_2
            promotion_pawns = (
                self.board.pieces_mask(chess.PAWN, self.board.turn) & promotion_rank
            )
            if promotion_pawns:
                moves.extend(
                    move
                    for move in self.board.generate_legal_moves(from_mask=promotion_pawns)
                    if move.promotion is not None and not self.board.is_capture(move)
                )
            has_legal_move = bool(moves) or any(self.board.generate_legal_moves())
        else:
            moves = list(self.board.legal_moves)
            has_legal_move = bool(moves)
        moves.sort(
            key=lambda move: self._move_order_score(move, tt_move, ply),
            reverse=True,
        )
        return moves, has_legal_move

    def _selective_move_order_score(
        self,
        move: chess.Move,
        tt_move: chess.Move | None,
        ply: int,
    ) -> int:
        if move == tt_move:
            return 20_000_000
        if self.board.is_capture(move):
            captured = (
                chess.PAWN
                if self.board.is_en_passant(move)
                else self.board.piece_type_at(move.to_square) or chess.PAWN
            )
            attacker = self.board.piece_type_at(move.from_square) or chess.PAWN
            return 8_000_000 + captured * 100_000 - attacker
        if _gives_check(self.board, move):
            return 7_500_000
        if move == self.killers[min(ply, _MAX_PLY - 1)][0]:
            return 7_000_000
        return _HISTORY[_move_key(move)]

    def _search_selective(
        self,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null: bool,
    ) -> int:
        """Fast competition search, kept semantically aligned with experiment."""
        self.nodes += 1
        if ply >= _MAX_PLY:
            return _evaluate(self.board)[0]
        if self.board.halfmove_clock >= 100:
            return 0 if not self.board.is_checkmate() else -_MATE_SCORE + ply
        if not (self.board.pawns | self.board.rooks | self.board.queens) and (
            self.board.is_insufficient_material()
        ):
            return 0
        if allow_null and self.board.is_repetition(2):
            return 0

        in_check = self.board.is_check()
        if in_check:
            depth += 1

        in_qsearch = depth <= 0
        null_window = beta == alpha + 1
        best_score = -_INFINITY
        static_score, phase = _evaluate(self.board)
        key = chess.polyglot.zobrist_hash(self.board)
        index = key & _TT_MASK
        entry = _TT[index] if self.use_tt else None
        tt_move: chess.Move | None = None
        tt_score: int | None = None
        tt_bound: int | None = None

        if entry is not None and entry[0] == key:
            _, tt_depth, raw_score, tt_bound, tt_move = entry
            # The selective search's mate values are already root-relative.
            # Re-normalizing them changes the experiment algorithm's bounds.
            tt_score = raw_score
            if tt_depth >= depth and null_window:
                usable_lower = tt_bound in (_EXACT, _LOWER) and tt_score >= beta
                usable_upper = tt_bound in (_EXACT, _UPPER) and tt_score <= alpha
                if usable_lower or usable_upper:
                    return tt_score

            raises_static = tt_bound in (_EXACT, _LOWER) and tt_score > static_score
            lowers_static = tt_bound in (_EXACT, _UPPER) and tt_score < static_score
            if raises_static or lowers_static:
                static_score = tt_score
        elif depth > 3:
            depth -= 1

        if in_qsearch and not in_check:
            if static_score >= beta:
                if any(self.board.generate_legal_moves()):
                    return static_score
                return -_MATE_SCORE + ply if in_check else 0
            alpha = max(alpha, static_score)
            best_score = static_score
        elif null_window and not in_check:
            if depth < 7 and static_score - depth * 75 > beta:
                return static_score if any(self.board.generate_legal_moves()) else 0

            if allow_null and static_score >= beta and depth > 2 and phase != 0:
                if not any(self.board.generate_legal_moves()):
                    return 0
                self.board.push(chess.Move.null())
                try:
                    null_score = -self._search_selective(
                        depth - (4 + depth // 6),
                        -beta,
                        -alpha,
                        ply + 1,
                        False,
                    )
                finally:
                    self.board.pop()
                if null_score >= beta:
                    return beta

        if in_qsearch and not in_check:
            moves = list(self.board.generate_legal_captures())
            promotion_rank = chess.BB_RANK_7 if self.board.turn else chess.BB_RANK_2
            promotion_pawns = (
                self.board.pieces_mask(chess.PAWN, self.board.turn) & promotion_rank
            )
            if promotion_pawns:
                moves.extend(
                    move
                    for move in self.board.generate_legal_moves(from_mask=promotion_pawns)
                    if move.promotion is not None and not self.board.is_capture(move)
                )
        else:
            moves = list(self.board.legal_moves)
        moves.sort(
            key=lambda move: self._selective_move_order_score(move, tt_move, ply),
            reverse=True,
        )

        hash_move = tt_move
        quiets: list[chess.Move] = []
        moves_searched = 0
        prunable_quiets = 0
        tt_bound = _UPPER

        for move in moves:
            is_quiet = not self.board.is_capture(move) and move.promotion is None
            gives_check = is_quiet and _gives_check(self.board, move)
            is_killer = move == self.killers[min(ply, _MAX_PLY - 1)][0]
            self.board.push(move)
            try:
                if in_qsearch or moves_searched == 0:
                    score = -self._search_selective(
                        depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        True,
                    )
                else:
                    reduction = 0
                    if (
                        depth > 2
                        and moves_searched > 4
                        and is_quiet
                        and not gives_check
                        and not is_killer
                    ):
                        history = _HISTORY[_move_key(move)]
                        history_sign = int(history > 0) - int(history < 0)
                        reduction = (
                            2
                            + depth // 8
                            + moves_searched // 16
                            + int(null_window and not in_check)
                            - history_sign
                        )
                        score = -self._search_selective(
                            depth - reduction,
                            -alpha - 1,
                            -alpha,
                            ply + 1,
                            True,
                        )
                    else:
                        score = alpha

                    if reduction == 0 or score > alpha:
                        score = -self._search_selective(
                            depth - 1,
                            -alpha - 1,
                            -alpha,
                            ply + 1,
                            True,
                        )
                        if alpha < score < beta:
                            score = -self._search_selective(
                                depth - 1,
                                -beta,
                                -alpha,
                                ply + 1,
                                True,
                            )
            finally:
                self.board.pop()

            # A child that returned after the hard deadline may itself contain
            # a partial subtree. Do not let that score replace a completed move.
            if depth > 2 and self._elapsed_ms() > self.hard_budget_ms:
                return best_score

            moves_searched += 1
            if ply == 0:
                self.iteration_root_moves_completed += 1
            if score > best_score:
                best_score = score
                if score > alpha:
                    alpha = score
                    hash_move = move
                    tt_bound = _EXACT
                    if ply == 0:
                        self.iteration_root_move = move
                    if alpha >= beta:
                        tt_bound = _LOWER
                        if is_quiet:
                            bonus = depth * depth
                            _HISTORY[_move_key(move)] += bonus
                            killer_ply = min(ply, _MAX_PLY - 1)
                            self.killers[killer_ply][0] = move
                            for prior_move in quiets:
                                _HISTORY[_move_key(prior_move)] -= bonus
                        break

            if is_quiet:
                quiets.append(move)
                if not gives_check:
                    prunable_quiets += 1
                    if (
                        null_window
                        and not in_check
                        and prunable_quiets > 3 + depth * depth
                    ):
                        break

        if moves_searched == 0:
            if in_check:
                return -_MATE_SCORE + ply
            if in_qsearch:
                return best_score
            return 0

        if self.use_tt:
            _TT[index] = (
                key,
                0 if in_qsearch else depth,
                best_score,
                tt_bound,
                hash_move,
            )
            _TT_AGE[index] = _TT_GENERATION
        return best_score

    def _quiescence(self, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self._check_time()

        if ply >= _MAX_PLY:
            return _evaluate(self.board)[0]
        if _is_search_draw(self.board, ply):
            return 0

        original_alpha = alpha
        original_beta = beta
        key = 0
        tt_move: chess.Move | None = None
        if self.use_tt:
            key = _tt_key(self.board, selective=self.selective)
            entry = _TT[key & _TT_MASK]
            if entry is not None and entry[0] == key and entry[1] == 0:
                _, _, raw_score, bound, tt_move = entry
                tt_score = _score_from_tt(raw_score, ply)
                if bound == _EXACT:
                    return tt_score
                if bound == _LOWER:
                    alpha = max(alpha, tt_score)
                else:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        in_check = self.board.is_check()
        stand_pat = _evaluate(self.board)[0]
        search_all_evasions = in_check and not self.selective
        if not search_all_evasions:
            if stand_pat >= beta:
                score = stand_pat if any(self.board.generate_legal_moves()) else 0
                bound = _LOWER if score >= original_beta else _EXACT
                self._store_tt(key, 0, score, bound, None, ply)
                return score
            alpha = max(alpha, stand_pat)

        moves, has_legal_move = self._ordered_moves(
            tt_move,
            ply,
            tactical_only=not search_all_evasions,
        )
        if not moves:
            if in_check and not has_legal_move:
                score = -_MATE_SCORE + ply
            else:
                score = stand_pat if has_legal_move else 0
            bound = _EXACT
            if score <= original_alpha:
                bound = _UPPER
            elif score >= original_beta:
                bound = _LOWER
            self._store_tt(key, 0, score, bound, None, ply)
            return score

        best_score = -_INFINITY if search_all_evasions else stand_pat
        best_move: chess.Move | None = None
        for move in moves:
            self.board.push(move)
            try:
                score = -self._quiescence(-beta, -alpha, ply + 1)
            finally:
                self.board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break

        bound = _EXACT
        if best_score <= original_alpha:
            bound = _UPPER
        elif best_score >= original_beta:
            bound = _LOWER
        self._store_tt(key, 0, best_score, bound, best_move, ply)
        return best_score

    def _search(
        self,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null: bool,
    ) -> int:
        if self.selective:
            return self._search_selective(depth, alpha, beta, ply, allow_null)

        self.nodes += 1
        self._check_time()

        if ply >= _MAX_PLY:
            return _evaluate(self.board)[0]
        if _is_search_draw(self.board, ply):
            return 0

        in_check = self.board.is_check()
        if in_check:
            depth += 1
        if depth <= 0:
            return self._quiescence(alpha, beta, ply)

        original_alpha = alpha
        original_beta = beta
        key = 0
        entry: TTEntry | None = None
        if self.use_tt:
            key = _tt_key(self.board, selective=self.selective)
            entry = _TT[key & _TT_MASK]
        tt_move: chess.Move | None = None
        tt_score: int | None = None
        tt_bound: int | None = None
        tt_depth = -1

        if entry is not None and entry[0] == key:
            _, tt_depth, raw_score, tt_bound, tt_move = entry
            tt_score = _score_from_tt(raw_score, ply)
            if ply == 0 and tt_move is not None and self.board.is_legal(tt_move):
                self.iteration_root_move = tt_move
            if tt_depth >= depth:
                if tt_bound == _EXACT:
                    return tt_score
                if tt_bound == _LOWER:
                    alpha = max(alpha, tt_score)
                else:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        # Internal iterative reduction is useful when move ordering has no TT
        # move. It is deliberately excluded from exact verification searches.
        if self.selective and ply > 0 and depth > 3 and tt_move is None:
            depth -= 1

        static_score, _ = _evaluate(self.board)
        if self.selective and tt_score is not None and tt_depth >= depth:
            raises_static = tt_bound in (_EXACT, _LOWER) and tt_score > static_score
            lowers_static = tt_bound in (_EXACT, _UPPER) and tt_score < static_score
            if raises_static or lowers_static:
                static_score = tt_score
        null_window = beta == alpha + 1
        if self.selective and null_window and not in_check:
            if depth <= 6 and static_score - depth * 75 >= beta:
                return static_score if any(self.board.generate_legal_moves()) else 0

            if allow_null and depth >= 3 and _null_move_safe(self.board) and static_score >= beta:
                if not any(self.board.generate_legal_moves()):
                    return 0
                reduction = 3 + depth // 6
                self.board.push(chess.Move.null())
                try:
                    null_score = -self._search(
                        depth - reduction - 1,
                        -beta,
                        -beta + 1,
                        ply + 1,
                        False,
                    )
                finally:
                    self.board.pop()
                if null_score >= beta:
                    # A null move is only a pruning proof, not a legal line whose
                    # fail-soft score may be propagated through the real tree.
                    return beta

        moves, _ = self._ordered_moves(tt_move, ply, tactical_only=False)
        if not moves:
            return -_MATE_SCORE + ply if in_check else 0

        best_score = -_INFINITY
        best_move: chess.Move | None = None
        quiets: list[chess.Move] = []
        prunable_quiets = 0

        for move_index, move in enumerate(moves):
            is_quiet = not self.board.is_capture(move) and move.promotion is None
            self.board.push(move)
            try:
                child_depth = depth - 1
                if move_index == 0:
                    score = -self._search(
                        child_depth,
                        -beta,
                        -alpha,
                        ply + 1,
                        True,
                    )
                else:
                    reduction = 0
                    reduced_depth = child_depth
                    if (
                        self.selective
                        and is_quiet
                        and depth >= 3
                        and move_index > 4
                        and not in_check
                    ):
                        history = _HISTORY[_move_key(move)]
                        history_sign = int(history > 0) - int(history < 0)
                        reduction = (
                            2
                            + depth // 8
                            + move_index // 16
                            + int(null_window)
                            - history_sign
                        )
                        reduced_depth = max(0, depth - reduction)

                    score = -self._search(
                        child_depth if reduction == 0 else reduced_depth,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        True,
                    )
                    if score > alpha and reduction:
                        score = -self._search(
                            child_depth,
                            -alpha - 1,
                            -alpha,
                            ply + 1,
                            True,
                        )
                    if alpha < score < beta:
                        score = -self._search(
                            child_depth,
                            -beta,
                            -alpha,
                            ply + 1,
                            True,
                        )
            finally:
                self.board.pop()

            if ply == 0:
                self.iteration_root_moves_completed += 1
            if score > best_score:
                best_score = score
                best_move = move
                if ply == 0:
                    self.iteration_root_move = move

            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if self.selective and is_quiet:
                        bonus = depth * depth
                        _HISTORY[_move_key(move)] += bonus
                        killer_ply = min(ply, _MAX_PLY - 1)
                        first_killer = self.killers[killer_ply][0]
                        if move != first_killer:
                            self.killers[killer_ply] = [move, first_killer]
                        for prior_move in quiets:
                            _HISTORY[_move_key(prior_move)] -= bonus
                    break

            if is_quiet:
                quiets.append(move)
                prunable_quiets += 1
                if (
                    self.selective
                    and null_window
                    and prunable_quiets > 3 + depth * depth
                ):
                    break
            self._check_time(force=ply == 0)

        bound = _EXACT
        if best_score <= original_alpha:
            bound = _UPPER
        elif best_score >= original_beta:
            bound = _LOWER
        self._store_tt(key, depth, best_score, bound, best_move, ply)
        return best_score

    def search_depth(self, depth: int) -> tuple[chess.Move, int]:
        """Run one full-window iteration for deterministic verification."""
        if depth < 1:
            raise ValueError("depth must be positive")
        if self.board.is_game_over(claim_draw=True):
            raise ValueError("cannot search a finished position")

        self.iteration_root_move = None
        score = self._search(depth, -_INFINITY, _INFINITY, 0, False)
        if self.iteration_root_move is None:
            raise RuntimeError("completed root search did not select a move")
        return self.iteration_root_move, score

    def _opponent_has_mate_in_one(self) -> bool:
        for reply in self.board.legal_moves:
            if not _gives_check(self.board, reply):
                continue
            self.board.push(reply)
            try:
                if self.board.is_checkmate():
                    return True
            finally:
                self.board.pop()
        return False

    def _fallback_score(self) -> int:
        if self.board.is_game_over(claim_draw=True):
            return 0
        if self._opponent_has_mate_in_one():
            return -_MATE_SCORE + 2
        return -_evaluate(self.board)[0]

    def _fallback_move(self, legal_moves: list[chess.Move]) -> chess.Move:
        """Choose a deterministic static fallback before a timed iteration."""
        best_move = legal_moves[0]
        best_score = -_INFINITY
        for move in legal_moves:
            self.board.push(move)
            try:
                if self.board.is_checkmate():
                    return move
                score = self._fallback_score()
            finally:
                self.board.pop()
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _selective_fallback(self, legal_moves: list[chess.Move]) -> chess.Move:
        """Find a mate-safe emergency move without spending the search budget."""
        for move in legal_moves:
            self.board.push(move)
            try:
                if self.board.is_checkmate():
                    return move
            finally:
                self.board.pop()

        for move in legal_moves:
            self.board.push(move)
            try:
                safe = not self._opponent_has_mate_in_one()
            finally:
                self.board.pop()
            if safe:
                return move
        return legal_moves[0]

    def choose_move(self) -> chess.Move:
        legal_moves = list(self.board.legal_moves)
        if len(legal_moves) == 1:
            self.completed_depth = 0
            return legal_moves[0]
        if self.selective:
            fallback = self._selective_fallback(legal_moves)
            score = 0
            depth = 1
            while self._elapsed_ms() <= self.soft_budget_ms:
                window = 40
                while True:
                    alpha = score - window
                    beta = score + window
                    self.iteration_root_moves_completed = 0
                    score = self._search_selective(depth, alpha, beta, 0, False)
                    if self._elapsed_ms() > self.hard_budget_ms:
                        break
                    if alpha < score < beta:
                        self.completed_depth = depth
                        self.completed_score = score
                        break
                    window *= 2
                depth += 1
            return self.iteration_root_move or fallback

        completed_best = self._fallback_move(legal_moves)
        previous_score = 0
        stable_iterations = 0
        score_change = _INFINITY
        aspiration_window = 35

        for depth in range(1, 65):
            if depth > 1 and self._elapsed_ms() >= self.soft_budget_ms:
                stable = stable_iterations >= 2 and score_change < aspiration_window
                extension_limit = min(self.hard_budget_ms, self.soft_budget_ms * 3)
                if stable or self._elapsed_ms() >= extension_limit:
                    break

            window = aspiration_window
            try:
                while True:
                    self.iteration_root_move = None
                    self.iteration_root_moves_completed = 0
                    alpha = previous_score - window
                    beta = previous_score + window
                    score = self._search(depth, alpha, beta, 0, False)

                    if score <= alpha or score >= beta:
                        window *= 2
                        continue
                    break
            except _SearchTimeout:
                if (
                    self.completed_depth > 0
                    and self.iteration_root_moves_completed > 0
                    and self.iteration_root_move is not None
                ):
                    completed_best = self.iteration_root_move
                break

            if self.iteration_root_move is not None:
                stable_iterations = (
                    stable_iterations + 1
                    if self.iteration_root_move == completed_best
                    else 0
                )
                score_change = abs(score - previous_score)
                completed_best = self.iteration_root_move
                previous_score = score
                self.completed_depth = depth
                self.completed_score = score

        return completed_best


def get_move(fen: str, time_left_ms: int) -> str:
    """Choose a legal UCI move for the side to move in the supplied FEN."""
    global _GAME_BOARD, _TT_GENERATION

    board = _recover_game_board(fen)
    if board.is_game_over(claim_draw=True):
        return "0000"

    _TT_GENERATION += 1
    for index, value in enumerate(_HISTORY):
        _HISTORY[index] = _divide_toward_zero(value, 8)

    move = _Searcher(board, time_left_ms).choose_move()
    board.push(move)
    _GAME_BOARD = board
    return move.uci()
