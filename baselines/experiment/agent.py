from __future__ import annotations

import time

import chess
import chess.polyglot

_INFINITY = 2_000_000
_MATE = 1_000_000
_TT_SIZE = 2_097_152
_TT_MASK = _TT_SIZE - 1

_EVAL_VALUES = (0, 0, 1, 1, 2, 4, 0, 0, 0, 458756, 393221, 393221,
                 524295, 1376266, 2228250,
                 0, 1114134, 1376281, 1572891, 1835037, 1900574,
                   1703971, 1507359, 1441800,
                   1376282, 1376284, 1441820, 1507356, 1572892, 
                   1507358, 1507354, 1572883, 
                   3014694, 2949155, 2949156, 3145764, 3211303,
                     3211307, 3276843, 3276844, 
                   5242956, 5374027, 5767241, 6160455, 6422598,
                     6357064, 6422596, 6160457,
                     -3, -3, -5, -3, 262146, 196621, 131083,
                       -393210, 0, 0, 0, 0, 0, 0, 0, 0, 
                983044, 917510, 786439, 655369, 720906, 720909,
                  786444, 720901, 1900571, 
                2097183, 2359329, 2490402, 2490402, 2359330,
                  2162721, 1900574, 2162720, 
                2228258, 2228258, 2293794, 2228258, 2228257,
                  2228259, 2162721, 3735588, 3735588, 3735590, 
                3735591, 3735590, 3735590, 3735588,
                  3670052, 6094946, 6226019, 6291555, 6422627, 6553698, 
                6553699, 6422629, 6422628,
                  -2, -131067, -1, -11, -4, -8, 3, -393215, 0, 0, 0, 458758, 262147, 
                196611, -10, 0, 0, -131063,
                  16, -655324, 1179671, -114, 0, 1703952,
                    -262139, 262145, 851999, 1376259,
                    -30, 0, 0, 0, 0, 0)

type TTEntry = tuple[int, chess.Move | None, int, int, int]
_TT: list[TTEntry | None] = [None] * _TT_SIZE
_QUIET_HISTORY = [0] * 4096
_GAME_BOARD: chess.Board | None = None


def _short(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def _trunc_div(numerator: int, denominator: int) -> int:
    quotient = abs(numerator) // denominator
    return -quotient if numerator < 0 else quotient


def _move_key(move: chess.Move) -> int:
    # The low 12 bits of ChessChallenge.API.Move.RawValue are source and target.
    return move.from_square | (move.to_square << 6)


def _restore_history(fen: str) -> chess.Board:
    global _GAME_BOARD
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


class _Searcher:
    def __init__(self, board: chess.Board, time_left_ms: int) -> None:
        self.board = board
        self.started = time.perf_counter()
        self.allocated_ms = max(0, time_left_ms // 8)
        self.root_best: chess.Move | None = None
        self.killers: list[chess.Move | None] = [None] * 256

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0

    def evaluate(self) -> tuple[int, int]:
        score = 15
        phase = 0
        for color in (not self.board.turn, self.board.turn):
            score = -score
            own = self.board.occupied_co[color]
            enemy_king = self.board.king(not color)
            king_attacks = chess.BB_KING_ATTACKS[enemy_king] if enemy_king is not None else 0
            pawns = self.board.pieces_mask(chess.PAWN, color)
            for piece_type in range(chess.PAWN, chess.KING + 1):
                pieces = self.board.pieces_mask(piece_type, color)
                while pieces:
                    square = chess.lsb(pieces)
                    pieces &= pieces - 1
                    file_mask = chess.BB_FILES[chess.square_file(square)]
                    if file_mask & ~chess.BB_SQUARES[square] & pawns == 0:
                        score += _EVAL_VALUES[126 + piece_type]
                    if piece_type > chess.KNIGHT:
                        mobility = self.board.attacks_mask(square) & ~own
                        score += (
                            _EVAL_VALUES[112 + piece_type] * mobility.bit_count()
                            + _EVAL_VALUES[119 + piece_type]
                            * (mobility & king_attacks).bit_count()
                        )
                    relative_square = square if color == chess.WHITE else square ^ 56
                    phase += _EVAL_VALUES[piece_type]
                    score += (
                        _EVAL_VALUES[piece_type * 8 + relative_square // 8]
                        + _EVAL_VALUES[56 + piece_type * 8 + relative_square % 8]
                    ) << 3
        middle_game = _short(score)
        end_game = (score + 0x8000) >> 16
        return _trunc_div(
            middle_game * phase + end_game * (24 - phase),
            24,
        ), phase

    def _move_order_score(
        self,
        move: chess.Move,
        tt_move: chess.Move | None,
        ply: int,
    ) -> int:
        if move == tt_move:
            return 9_000_000_000_000_000_000
        if self.board.is_capture(move):
            captured = (
                chess.PAWN
                if self.board.is_en_passant(move)
                else self.board.piece_type_at(move.to_square) or 0
            )
            mover = self.board.piece_type_at(move.from_square) or 0
            return 1_000_000_000_000_000_000 * captured - mover
        if ply < len(self.killers) and move == self.killers[ply]:
            return 500_000_000_000_000_000
        return _QUIET_HISTORY[_move_key(move)]

    def search(
        self,
        ply: int,
        depth: int,
        alpha: int,
        beta: int,
        null_allowed: bool,
    ) -> int:
        if null_allowed and self.board.is_repetition(2):
            return 0

        in_check = self.board.is_check()
        if in_check:
            depth += 1

        key = chess.polyglot.zobrist_hash(self.board)
        in_qsearch = depth <= 0
        best_score = -_INFINITY
        do_pruning = alpha == beta - 1 and not in_check
        score, phase = self.evaluate()

        entry = _TT[key & _TT_MASK]
        tt_move: chess.Move | None = None
        tt_flag = 0
        if entry is not None and entry[0] == key:
            _, tt_move, tt_depth, tt_score, tt_flag = entry
            expected_bound = 0 if tt_score >= beta else 2
            if alpha == beta - 1 and tt_depth >= depth and tt_flag != expected_bound:
                return tt_score
            expected_static_bound = 0 if tt_score > score else 2
            if tt_flag != expected_static_bound:
                score = tt_score
        elif depth > 3:
            depth -= 1

        if in_qsearch:
            if score >= beta:
                return score
            if score > alpha:
                alpha = score
            best_score = score
        elif do_pruning:
            if depth < 7 and score - depth * 75 > beta:
                return score
            if null_allowed and score >= beta and depth > 2 and phase != 0:
                self.board.push(chess.Move.null())
                score = -self.search(
                    ply + 1,
                    depth - (4 + depth // 6),
                    -beta,
                    -alpha,
                    False,
                )
                self.board.pop()
                if score >= beta:
                    return beta

        moves = (
            list(self.board.generate_legal_captures())
            if in_qsearch
            else list(self.board.legal_moves)
        )
        moves.sort(
            key=lambda move: self._move_order_score(move, tt_move, ply),
            reverse=True,
        )
        quiets_evaluated: list[chess.Move] = []
        moves_evaluated = 0
        tt_flag = 0

        for move in moves:
            is_quiet = not self.board.is_capture(move)
            self.board.push(move)

            if in_qsearch or moves_evaluated == 0:
                score = -self.search(ply + 1, depth - 1, -beta, -alpha, True)
            else:
                skip_move = False
                no_reduction = depth <= 2 or moves_evaluated <= 4 or not is_quiet
                if not no_reduction:
                    history = _QUIET_HISTORY[_move_key(move)]
                    history_sign = int(history > 0) - int(history < 0)
                    reduction = (
                        2
                        + depth // 8
                        + moves_evaluated // 16
                        + int(do_pruning)
                        - history_sign
                    )
                    score = -self.search(
                        ply + 1,
                        depth - reduction,
                        -(alpha + 1),
                        -alpha,
                        True,
                    )
                    skip_move = score <= alpha

                if not skip_move:
                    score = -self.search(
                        ply + 1,
                        depth - 1,
                        -(alpha + 1),
                        -alpha,
                        True,
                    )
                    if alpha < score < beta:
                        score = -self.search(
                            ply + 1,
                            depth - 1,
                            -beta,
                            -alpha,
                            True,
                        )

            self.board.pop()

            if depth > 2 and self.elapsed_ms() > self.allocated_ms:
                return best_score

            moves_evaluated += 1
            if score > best_score:
                best_score = score
                if score > alpha:
                    tt_move = move
                    if ply == 0:
                        self.root_best = move
                    alpha = score
                    tt_flag = 1
                    if score >= beta:
                        if is_quiet:
                            bonus = depth * depth
                            _QUIET_HISTORY[_move_key(move)] += bonus
                            for previous in quiets_evaluated:
                                _QUIET_HISTORY[_move_key(previous)] -= bonus
                            if ply < len(self.killers):
                                self.killers[ply] = move
                        tt_flag += 1
                        break

            if is_quiet:
                quiets_evaluated.append(move)
            if do_pruning and len(quiets_evaluated) > 3 + depth * depth:
                break

        if moves_evaluated == 0:
            if in_qsearch:
                return best_score
            return ply - _MATE if in_check else 0

        _TT[key & _TT_MASK] = (key, tt_move, 0 if in_qsearch else depth, best_score, tt_flag)
        return best_score

    def choose_move(self) -> chess.Move:
        fallback = next(iter(self.board.legal_moves))
        score = 0
        depth = 1
        soft_limit = self.allocated_ms // 5
        while self.elapsed_ms() <= soft_limit:
            window = 40
            while True:
                alpha = score - window
                beta = score + window
                score = self.search(0, depth, alpha, beta, False)
                if self.elapsed_ms() > self.allocated_ms:
                    break
                if alpha < score < beta:
                    break
                window *= 2
            depth += 1
        return self.root_best or fallback


def get_move(fen: str, time_left_ms: int) -> str:
    """Return Smol's selected move in the harness UCI contract."""
    global _GAME_BOARD
    board = _restore_history(fen)
    if board.is_game_over(claim_draw=True):
        return "0000"
    for index, value in enumerate(_QUIET_HISTORY):
        _QUIET_HISTORY[index] = _trunc_div(value, 8)
    move = _Searcher(board, time_left_ms).choose_move()
    board.push(move)
    _GAME_BOARD = board
    return move.uci()
