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
            max(1, remaining_ms // 8),
            max(1, remaining_ms - reserve_ms),
        )
        self.soft_budget_ms = max(1, self.hard_budget_ms // 5)
        self.nodes = 0
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(_MAX_PLY)]
        self.iteration_root_move: chess.Move | None = None
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
            return
        _TT[index] = (key, depth, _score_to_tt(score, ply), bound, move)

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
        if self.board.gives_check(move):
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
        legal_moves = list(self.board.legal_moves)
        if tactical_only:
            moves = [
                move
                for move in legal_moves
                if self.board.is_capture(move) or move.promotion is not None
            ]
        else:
            moves = legal_moves
        moves.sort(
            key=lambda move: self._move_order_score(move, tt_move, ply),
            reverse=True,
        )
        return moves, bool(legal_moves)

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
        if not in_check:
            if stand_pat >= beta:
                score = stand_pat if any(self.board.generate_legal_moves()) else 0
                bound = _LOWER if score >= original_beta else _EXACT
                self._store_tt(key, 0, score, bound, None, ply)
                return score
            alpha = max(alpha, stand_pat)

        moves, has_legal_move = self._ordered_moves(
            tt_move,
            ply,
            tactical_only=not in_check,
        )
        if not moves:
            score = (
                -_MATE_SCORE + ply
                if in_check
                else (stand_pat if has_legal_move else 0)
            )
            bound = _EXACT
            if score <= original_alpha:
                bound = _UPPER
            elif score >= original_beta:
                bound = _LOWER
            self._store_tt(key, 0, score, bound, None, ply)
            return score

        best_score = -_INFINITY if in_check else stand_pat
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

        if entry is not None and entry[0] == key:
            _, stored_depth, raw_score, tt_bound, tt_move = entry
            tt_score = _score_from_tt(raw_score, ply)
            if ply == 0 and tt_move is not None and self.board.is_legal(tt_move):
                self.iteration_root_move = tt_move
            if stored_depth >= depth:
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
        if self.selective and tt_score is not None:
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
                    return null_score

        moves, _ = self._ordered_moves(tt_move, ply, tactical_only=False)
        if not moves:
            return -_MATE_SCORE + ply if in_check else 0

        best_score = -_INFINITY
        best_move: chess.Move | None = None
        quiets: list[chess.Move] = []
        prunable_quiets = 0

        for move_index, move in enumerate(moves):
            is_quiet = not self.board.is_capture(move) and move.promotion is None
            gives_check = self.board.gives_check(move)
            killer_ply = min(ply, _MAX_PLY - 1)
            is_killer = move in self.killers[killer_ply]
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
                    if (
                        self.selective
                        and is_quiet
                        and depth >= 3
                        and move_index > 4
                        and not in_check
                        and not gives_check
                        and not is_killer
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
                    if self.selective and is_quiet:
                        bonus = depth * depth
                        _HISTORY[_move_key(move)] += bonus
                        first_killer = self.killers[killer_ply][0]
                        if move != first_killer:
                            self.killers[killer_ply] = [move, first_killer]
                        for prior_move in quiets:
                            _HISTORY[_move_key(prior_move)] -= bonus
                    break

            if is_quiet:
                quiets.append(move)
                if not gives_check:
                    prunable_quiets += 1
                    if (
                        self.selective
                        and null_window
                        and prunable_quiets > 3 + depth * depth
                    ):
                        break

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
            if not self.board.gives_check(reply):
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

    def choose_move(self) -> chess.Move:
        legal_moves = list(self.board.legal_moves)
        if len(legal_moves) == 1:
            self.completed_depth = 0
            return legal_moves[0]
        completed_best = self._fallback_move(legal_moves)
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
                self.completed_depth = depth
                self.completed_score = score

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
