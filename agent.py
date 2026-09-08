from __future__ import annotations

import time

import chess
import chess.polyglot

_INFINITY = 2_000_000
_MATE_SCORE = 1_000_000
_MAX_PLY = 192
_TT_SIZE = 1 << 20
_TT_MASK = _TT_SIZE - 1

_EXACT = 0
_LOWER = 1
_UPPER = 2

# Packed middle-game/endgame parameters shared with baselines/experiment.
_PACKED_EVALUATION = (
    0,
    0,
    1,
    1,
    2,
    4,
    0,
    0,
    0,
    458756,
    393221,
    393221,
    524295,
    1376266,
    2228250,
    0,
    1114134,
    1376281,
    1572891,
    1835037,
    1900574,
    1703971,
    1507359,
    1441800,
    1376282,
    1376284,
    1441820,
    1507356,
    1572892,
    1507358,
    1507354,
    1572883,
    3014694,
    2949155,
    2949156,
    3145764,
    3211303,
    3211307,
    3276843,
    3276844,
    5242956,
    5374027,
    5767241,
    6160455,
    6422598,
    6357064,
    6422596,
    6160457,
    -3,
    -3,
    -5,
    -3,
    262146,
    196621,
    131083,
    -393210,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    983044,
    917510,
    786439,
    655369,
    720906,
    720909,
    786444,
    720901,
    1900571,
    2097183,
    2359329,
    2490402,
    2490402,
    2359330,
    2162721,
    1900574,
    2162720,
    2228258,
    2228258,
    2293794,
    2228258,
    2228257,
    2228259,
    2162721,
    3735588,
    3735588,
    3735590,
    3735591,
    3735590,
    3735590,
    3735588,
    3670052,
    6094946,
    6226019,
    6291555,
    6422627,
    6553698,
    6553699,
    6422629,
    6422628,
    -2,
    -131067,
    -1,
    -11,
    -4,
    -8,
    3,
    -393215,
    0,
    0,
    0,
    458758,
    262147,
    196611,
    -10,
    0,
    0,
    -131063,
    16,
    -655324,
    1179671,
    -114,
    0,
    1703952,
    -262139,
    262145,
    851999,
    1376259,
    -30,
    0,
    0,
    0,
    0,
    0,
)

type ScorePair = tuple[int, int]
type TTEntry = tuple[int, int, int, int, chess.Move | None]


def _signed_short(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def _decode_score(value: int) -> ScorePair:
    return _signed_short(value), (value + 0x8000) >> 16


def _divide_toward_zero(numerator: int, denominator: int) -> int:
    quotient = abs(numerator) // denominator
    return -quotient if numerator < 0 else quotient


_EVALUATION: tuple[ScorePair, ...] = tuple(_decode_score(value) for value in _PACKED_EVALUATION)
_TT: list[TTEntry | None] = [None] * _TT_SIZE
_HISTORY = [0] * 4096
_GAME_BOARD: chess.Board | None = None


class _SearchTimeout(Exception):
    pass


def _move_key(move: chess.Move) -> int:
    return move.from_square | (move.to_square << 6)


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
        own = board.occupied_co[color]
        pawns = board.pieces_mask(chess.PAWN, color)
        enemy_king = board.king(not color)
        king_zone = chess.BB_KING_ATTACKS[enemy_king] if enemy_king is not None else 0

        for piece_type in range(chess.PAWN, chess.KING + 1):
            pieces = board.pieces_mask(piece_type, color)
            phase += _PACKED_EVALUATION[piece_type] * pieces.bit_count()

            while pieces:
                square = chess.lsb(pieces)
                pieces &= pieces - 1
                relative_square = square if color == chess.WHITE else square ^ 56

                rank_score = _EVALUATION[piece_type * 8 + relative_square // 8]
                file_score = _EVALUATION[56 + piece_type * 8 + relative_square % 8]
                middle_game += sign * 8 * (rank_score[0] + file_score[0])
                end_game += sign * 8 * (rank_score[1] + file_score[1])

                file_mask = chess.BB_FILES[chess.square_file(square)]
                other_pawns = pawns & file_mask & ~chess.BB_SQUARES[square]
                if other_pawns == 0:
                    open_file_score = _EVALUATION[126 + piece_type]
                    middle_game += sign * open_file_score[0]
                    end_game += sign * open_file_score[1]

                if piece_type > chess.KNIGHT:
                    attacks = board.attacks_mask(square) & ~own
                    mobility = attacks.bit_count()
                    king_pressure = (attacks & king_zone).bit_count()
                    mobility_score = _EVALUATION[112 + piece_type]
                    pressure_score = _EVALUATION[119 + piece_type]
                    middle_game += sign * (
                        mobility_score[0] * mobility + pressure_score[0] * king_pressure
                    )
                    end_game += sign * (
                        mobility_score[1] * mobility + pressure_score[1] * king_pressure
                    )

    phase = min(24, phase)
    tapered = _divide_toward_zero(
        middle_game * phase + end_game * (24 - phase),
        24,
    )
    return tapered, phase


class _Searcher:
    def __init__(self, board: chess.Board, time_left_ms: int) -> None:
        self.board = board
        self.started = time.perf_counter()
        remaining_ms = max(1, time_left_ms)
        reserve_ms = min(50, max(2, remaining_ms // 20))
        self.hard_budget_ms = min(
            max(1, remaining_ms // 8),
            max(1, remaining_ms - reserve_ms),
        )
        self.soft_budget_ms = max(1, self.hard_budget_ms // 5)
        self.nodes = 0
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(_MAX_PLY)]
        self.iteration_root_move: chess.Move | None = None

    def _elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0

    def _check_time(self, *, force: bool = False) -> None:
        if (force or self.nodes & 1023 == 0) and (self._elapsed_ms() >= self.hard_budget_ms):
            raise _SearchTimeout

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
    ) -> list[chess.Move]:
        if tactical_only:
            moves = [
                move
                for move in self.board.legal_moves
                if self.board.is_capture(move) or move.promotion is not None
            ]
        else:
            moves = list(self.board.legal_moves)
        moves.sort(
            key=lambda move: self._move_order_score(move, tt_move, ply),
            reverse=True,
        )
        return moves

    def _quiescence(self, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self._check_time()

        if ply >= _MAX_PLY:
            return _evaluate(self.board)[0]
        if self.board.is_repetition(2) or self.board.halfmove_clock >= 100:
            return 0

        in_check = self.board.is_check()
        stand_pat = _evaluate(self.board)[0]
        if not in_check:
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)

        moves = self._ordered_moves(None, ply, tactical_only=not in_check)
        if not moves:
            return -_MATE_SCORE + ply if in_check else stand_pat

        best_score = -_INFINITY if in_check else stand_pat
        for move in moves:
            self.board.push(move)
            try:
                score = -self._quiescence(-beta, -alpha, ply + 1)
            finally:
                self.board.pop()

            best_score = max(best_score, score)
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
        return best_score

    def _search(
        self,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null: bool,
    ) -> int:
        self.nodes += 1
        self._check_time()

        if ply >= _MAX_PLY:
            return _evaluate(self.board)[0]
        if self.board.is_repetition(2) or self.board.halfmove_clock >= 100:
            return 0

        in_check = self.board.is_check()
        if in_check:
            depth += 1
        if depth <= 0:
            return self._quiescence(alpha, beta, ply)

        original_alpha = alpha
        original_beta = beta
        key = chess.polyglot.zobrist_hash(self.board)
        entry = _TT[key & _TT_MASK]
        tt_move: chess.Move | None = None

        if entry is not None and entry[0] == key:
            _, stored_depth, stored_score, bound, tt_move = entry
            if stored_depth >= depth:
                if bound == _EXACT:
                    return stored_score
                if bound == _LOWER:
                    alpha = max(alpha, stored_score)
                else:
                    beta = min(beta, stored_score)
                if alpha >= beta:
                    return stored_score

        static_score, phase = _evaluate(self.board)
        null_window = beta == alpha + 1
        if null_window and not in_check:
            if depth <= 6 and static_score - depth * 75 >= beta:
                return static_score

            if allow_null and depth >= 3 and phase > 0 and static_score >= beta:
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
                    return null_score

        moves = self._ordered_moves(tt_move, ply, tactical_only=False)
        if not moves:
            return -_MATE_SCORE + ply if in_check else 0

        best_score = -_INFINITY
        best_move: chess.Move | None = None
        quiets: list[chess.Move] = []

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
                    if is_quiet and depth >= 3 and move_index >= 4 and not in_check:
                        history = _HISTORY[_move_key(move)]
                        reduction = 1 + depth // 7 + move_index // 12 - int(history > 0)
                        reduction = min(reduction, max(0, child_depth - 1))

                    score = -self._search(
                        child_depth - reduction,
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

            self._check_time(force=ply == 0)
            if score > best_score:
                best_score = score
                best_move = move
                if ply == 0:
                    self.iteration_root_move = move

            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if is_quiet:
                        bonus = min(2_000, depth * depth)
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
                if null_window and len(quiets) > 3 + depth * depth:
                    break

        bound = _EXACT
        if best_score <= original_alpha:
            bound = _UPPER
        elif best_score >= original_beta:
            bound = _LOWER
        _TT[key & _TT_MASK] = (key, depth, best_score, bound, best_move)
        return best_score

    def choose_move(self) -> chess.Move:
        legal_moves = list(self.board.legal_moves)
        completed_best = legal_moves[0]
        previous_score = 0

        for depth in range(1, 65):
            if depth > 1 and self._elapsed_ms() >= self.soft_budget_ms:
                break

            window = 35
            try:
                while True:
                    self.iteration_root_move = None
                    alpha = previous_score - window
                    beta = previous_score + window
                    score = self._search(depth, alpha, beta, 0, False)

                    if score <= alpha or score >= beta:
                        window *= 2
                        continue
                    break
            except _SearchTimeout:
                break

            if self.iteration_root_move is not None:
                completed_best = self.iteration_root_move
                previous_score = score

        return completed_best


def get_move(fen: str, time_left_ms: int) -> str:
    """Choose a legal UCI move for the side to move in the supplied FEN."""
    global _GAME_BOARD

    board = _recover_game_board(fen)
    if board.is_game_over(claim_draw=True):
        return "0000"

    for index, value in enumerate(_HISTORY):
        _HISTORY[index] = _divide_toward_zero(value, 8)

    move = _Searcher(board, time_left_ms).choose_move()
    board.push(move)
    _GAME_BOARD = board
    return move.uci()
