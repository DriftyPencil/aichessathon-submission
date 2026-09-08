"""Pure neural PUCT agent using calibrated team-trained ONNX inference."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from pathlib import Path

import chess
import chess.polyglot
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from chess_encoding import action_index, encode_board

MAX_SIMULATIONS = 131_072
SEARCH_BATCH_SIZE = 4
MAX_TREE_NODES = 200_000
PUCT = 1.55
FPU_REDUCTION = 0.2
POLICY_FLOOR_MIX = 0.03
ROOT_REPLY_FORCING_MIX = 0.25
MAX_CACHED_EVALUATIONS = 750_000


class SearchNode:
    __slots__ = (
        "_children",
        "_moves",
        "_priors",
        "expanded",
        "forcing_mixed",
        "prior",
        "proven",
        "size",
        "value_sum",
        "visits",
    )

    def __init__(self, prior: float = 0.0) -> None:
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self._children: dict[chess.Move, SearchNode] | None = None
        self._moves: list[chess.Move] | None = None
        self._priors: np.ndarray | None = None
        self.expanded = False
        self.forcing_mixed = False
        self.proven: float | None = None
        self.size = 1

    @property
    def children(self) -> dict[chess.Move, SearchNode]:
        if self._children is None:
            self._children = {}
        return self._children

    @children.setter
    def children(self, value: dict[chess.Move, SearchNode]) -> None:
        self._children = value

    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


_OPTIONS = ort.SessionOptions()
_OPTIONS.intra_op_num_threads = 1
_OPTIONS.inter_op_num_threads = 1
_OPTIONS.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
_OPTIONS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_MODEL_PATH = Path(__file__).with_name("weights") / "az_lite-int8.onnx"
_BOOK_PATH = Path(__file__).with_name("weights") / "opening_book.npz"
try:
    _INFERENCE = ort.InferenceSession(
        _MODEL_PATH,
        sess_options=_OPTIONS,
        providers=["CPUExecutionProvider"],
    )
    _INFERENCE.run(None, {"state": np.zeros((1, 18, 8, 8), dtype=np.float32)})
except (OSError, RuntimeError, ValueError) as error:
    raise RuntimeError(f"could not load trained AlphaZero-lite ONNX model: {error}") from error
if _BOOK_PATH.exists():
    with np.load(_BOOK_PATH) as archive:
        _BOOK_KEYS = archive["keys"].astype(np.uint64)
        _BOOK_ACTIONS = archive["actions"].astype(np.uint16)
else:
    _BOOK_KEYS = np.empty(0, dtype=np.uint64)
    _BOOK_ACTIONS = np.empty(0, dtype=np.uint16)

_GAME_BOARD: chess.Board | None = None
_GAME_ROOT: SearchNode | None = None
_EVALUATIONS: OrderedDict[tuple[int, ...], tuple[np.ndarray, float]] = OrderedDict()
_CACHE_HITS = 0
_CACHE_MISSES = 0


def get_move(fen: str, time_left_ms: int) -> str:
    """Return the highest-visit legal move from neural PUCT search."""
    global _GAME_BOARD, _GAME_ROOT
    started = time.perf_counter()
    board = _restore_history(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"
    if len(moves) == 1 or time_left_ms < 20:
        chosen = moves[0]
    elif mate := _mate_in_one(board, moves):
        chosen = mate
    elif book_move := _book_move(board, moves):
        chosen = book_move
    else:
        chosen = _mcts_move(board, started + _move_budget_ms(time_left_ms) / 1000.0)
    _GAME_ROOT = _GAME_ROOT.children.get(chosen) if _GAME_ROOT is not None else None
    board.push(chosen)
    _GAME_BOARD = board
    return chosen.uci()


def _restore_history(fen: str) -> chess.Board:
    global _GAME_ROOT
    target = chess.Board(fen)
    if _GAME_BOARD is None:
        _GAME_ROOT = None
        return target
    target_fen = target.fen()
    for reply in _GAME_BOARD.legal_moves:
        candidate = _GAME_BOARD.copy(stack=True)
        candidate.push(reply)
        if candidate.fen() == target_fen:
            _GAME_ROOT = _GAME_ROOT.children.get(reply) if _GAME_ROOT is not None else None
            return candidate
    _GAME_ROOT = None
    return target


def _move_budget_ms(time_left_ms: int) -> float:
    if time_left_ms <= 500:
        return max(0.0, min(time_left_ms * 0.03, time_left_ms - 10.0))
    if time_left_ms <= 3_000:
        return min(200.0, max(35.0, time_left_ms * 0.08))
    desired = min(8_000.0, max(80.0, time_left_ms / 14.0))
    return min(desired, time_left_ms - 500.0)


def _mate_in_one(board: chess.Board, moves: list[chess.Move]) -> chess.Move | None:
    for move in moves:
        board.push(move)
        try:
            if board.is_checkmate():
                return move
        finally:
            board.pop()
    return None


def _book_move(board: chess.Board, moves: list[chess.Move]) -> chess.Move | None:
    if not len(_BOOK_KEYS):
        return None
    key = np.uint64(chess.polyglot.zobrist_hash(board))
    row = int(np.searchsorted(_BOOK_KEYS, key))
    if row >= len(_BOOK_KEYS) or _BOOK_KEYS[row] != key:
        return None
    target = int(_BOOK_ACTIONS[row])
    return next((move for move in moves if action_index(board, move) == target), None)


def _terminal_value(board: chess.Board, moves: list[chess.Move]) -> float | None:
    if not moves:
        return -1.0 if board.is_check() else 0.0
    if board.is_insufficient_material():
        return 0.0
    if board.halfmove_clock >= 150 or board.is_fivefold_repetition():
        return 0.0
    if board.halfmove_clock >= 99 and board.can_claim_fifty_moves():
        return 0.0
    if board.is_repetition(2) and board.can_claim_threefold_repetition():
        return 0.0
    return None


def _mcts_move(board: chess.Board, deadline: float) -> chess.Move:
    global _GAME_ROOT
    if time.perf_counter() >= deadline:
        moves = list(board.legal_moves)
        if mate := _mate_in_one(board, moves):
            return mate
        return moves[0]
    root = _GAME_ROOT
    if root is None or root.visits > 2 * MAX_SIMULATIONS or root.size > MAX_TREE_NODES // 2:
        root = SearchNode()
    _GAME_ROOT = root
    if not root.expanded:
        root_moves = list(board.legal_moves)
        _expand_node(
            root,
            root_moves,
            _evaluate_many([(board, root_moves)])[0],
            materialize=True,
        )
    _prepare_root_tactics(root, board)
    simulations = 0
    search_board = board.copy(stack=True)
    root_stack_size = len(search_board.move_stack)
    while (
        root.proven is None
        and root.size < MAX_TREE_NODES
        and simulations < MAX_SIMULATIONS
        and time.perf_counter() < deadline
    ):
        pending: list[
            tuple[SearchNode, chess.Board, list[SearchNode], list[chess.Move], int]
        ] = []
        queued: set[SearchNode] = set()
        try:
            for _ in range(min(SEARCH_BATCH_SIZE, MAX_SIMULATIONS - simulations)):
                if root.proven is not None or time.perf_counter() >= deadline:
                    break
                node = root
                path = [root]
                added_nodes = 0
                try:
                    while node.expanded and node._moves and node.proven is None:
                        parent = node
                        child_count = len(parent._children) if parent._children is not None else 0
                        move, node = _select_child(parent)
                        if len(parent.children) > child_count:
                            added_nodes += 1
                        search_board.push(move)
                        path.append(node)
                    if node in queued:
                        break
                    leaf_moves = list(search_board.legal_moves)
                    if node.proven is None:
                        node.proven = _terminal_value(search_board, leaf_moves)
                    if node.proven is not None:
                        _backpropagate(path, node.proven, added_nodes=added_nodes)
                        simulations += 1
                        continue
                    pending.append(
                        (node, search_board.copy(stack=False), path, leaf_moves, added_nodes)
                    )
                    queued.add(node)
                    for ancestor in path:
                        ancestor.visits += 1
                        ancestor.value_sum += 1.0
                finally:
                    while len(search_board.move_stack) > root_stack_size:
                        search_board.pop()
            predictions = _evaluate_many([(item[1], item[3]) for item in pending])
        finally:
            for _, _, path, _, _ in pending:
                for ancestor in path:
                    ancestor.visits -= 1
                    ancestor.value_sum -= 1.0
        for (node, leaf_board, path, moves, added_nodes), prediction in zip(
            pending, predictions, strict=True
        ):
            forcing_board = leaf_board if len(path) == 2 else None
            leaf_value = _expand_node(node, moves, prediction, forcing_board)
            _backpropagate(path, leaf_value, added_nodes=added_nodes)
            simulations += 1
    if not root.children:
        return next(iter(board.legal_moves))
    return _best_move(root)


def _best_move(root: SearchNode) -> chess.Move:
    def rank(item: tuple[chess.Move, SearchNode]) -> tuple[int, int, float]:
        child = item[1]
        proof_rank = 2 if child.proven == -1.0 else (0 if child.proven == 1.0 else 1)
        return proof_rank, child.visits, child.prior

    return max(root.children.items(), key=rank)[0]


def _prepare_root_tactics(root: SearchNode, board: chess.Board) -> None:
    for move, child in root.children.items():
        if child.proven is not None:
            continue
        board.push(move)
        try:
            if child.expanded and not child.forcing_mixed:
                assert child._moves is not None and child._priors is not None
                replies = child._moves
                priors = child._priors
                _mix_forcing_priors(board, replies, priors)
                if child._children is not None:
                    for reply, prior in zip(replies, priors, strict=True):
                        materialized = child._children.get(reply)
                        if materialized is not None:
                            materialized.prior = float(prior)
                child.forcing_mixed = True
            if _mate_in_one(board, list(board.legal_moves)) is not None:
                child.proven = 1.0
        finally:
            board.pop()


def _select_child(node: SearchNode) -> tuple[chess.Move, SearchNode]:
    assert node._moves is not None and node._priors is not None
    parent_scale = math.sqrt(node.visits if node.visits > 0 else 1)
    parent_value = node.value_sum / node.visits if node.visits else 0.0
    best_move: chess.Move | None = None
    best_child: SearchNode | None = None
    best_prior = 0.0
    best_score = -math.inf
    children = node._children
    for move, raw_prior in zip(node._moves, node._priors, strict=True):
        child = children.get(move) if children is not None else None
        prior = float(raw_prior)
        if child is not None and child.proven is not None:
            score = 0.0 if child.proven == 0.0 else -math.inf
        else:
            exploitation = (
                -child.value_sum / child.visits
                if child is not None and child.visits
                else parent_value - FPU_REDUCTION
            )
            visits = child.visits if child is not None else 0
            score = exploitation + PUCT * prior * parent_scale / (1 + visits)
        if best_move is None or score > best_score:
            best_move = move
            best_child = child
            best_prior = prior
            best_score = score
    assert best_move is not None
    if best_child is None:
        best_child = SearchNode(best_prior)
        node.children[best_move] = best_child
    return best_move, best_child


def _expand_node(
    node: SearchNode,
    moves: list[chess.Move],
    prediction: tuple[np.ndarray, float],
    forcing_board: chess.Board | None = None,
    materialize: bool = False,
) -> float:
    priors, value = prediction
    priors = priors.copy()
    if forcing_board is not None:
        _mix_forcing_priors(forcing_board, moves, priors)
        node.forcing_mixed = True
    node._moves = moves
    node._priors = priors
    if materialize:
        node.children = {
            move: SearchNode(float(prior)) for move, prior in zip(moves, priors, strict=True)
        }
        node.size = 1 + len(node.children)
    node.expanded = True
    return value


def _mix_forcing_priors(
    board: chess.Board,
    moves: list[chess.Move],
    priors: np.ndarray,
) -> None:
    forcing = np.fromiter(
        (
            board.gives_check(move) or board.is_capture(move) or move.promotion is not None
            for move in moves
        ),
        dtype=np.bool_,
        count=len(moves),
    )
    forcing_count = int(forcing.sum())
    if forcing_count:
        priors *= 1.0 - ROOT_REPLY_FORCING_MIX
        priors[forcing] += ROOT_REPLY_FORCING_MIX / forcing_count


def _predict(board: chess.Board) -> tuple[np.ndarray, float]:
    return _predict_many([board])[0]


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
        board.promoted,
        -1 if board.ep_square is None else board.ep_square,
        min(board.halfmove_clock, 100),
    )


def _predict_many(boards: list[chess.Board]) -> list[tuple[np.ndarray, float]]:
    move_lists = [list(board.legal_moves) for board in boards]
    return _evaluate_many(list(zip(boards, move_lists, strict=True)))


def _evaluate_many(
    positions: list[tuple[chess.Board, list[chess.Move]]],
) -> list[tuple[np.ndarray, float]]:
    global _CACHE_HITS, _CACHE_MISSES
    if not positions:
        return []
    boards = [position[0] for position in positions]
    keys = [_input_key(board) for board in boards]
    resolved: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
    missing: dict[tuple[int, ...], tuple[chess.Board, list[chess.Move]]] = {}
    for key, position in zip(keys, positions, strict=True):
        cached = _EVALUATIONS.get(key)
        if cached is not None:
            _EVALUATIONS.move_to_end(key)
            _CACHE_HITS += 1
            resolved[key] = cached
        elif key not in missing:
            missing[key] = position
    if missing:
        states = np.stack([encode_board(board) for board, _ in missing.values()])
        if isinstance(_INFERENCE, ort.InferenceSession):
            raw_policies, raw_values = _INFERENCE.run(None, {"state": states})
        else:
            raw_policies, raw_values = _INFERENCE(states)
        torch_like = hasattr(raw_policies, "detach")
        if not torch_like:
            policies = np.asarray(raw_policies)
            values = np.asarray(raw_values)
        for row, (key, (board, moves)) in enumerate(missing.items()):
            if torch_like:
                policy = raw_policies[row].detach().cpu().numpy()
                value = float(raw_values[row])
            else:
                policy = policies[row]
                value = float(values[row])
            indices = np.fromiter(
                (action_index(board, move) for move in moves),
                dtype=np.int64,
                count=len(moves),
            )
            selected = np.asarray(policy[indices], dtype=np.float32)
            if len(selected):
                priors = np.exp(selected - float(selected.max()))
                priors /= priors.sum()
                priors *= 1.0 - POLICY_FLOOR_MIX
                priors += POLICY_FLOOR_MIX / len(priors)
            else:
                priors = selected
            prediction = priors, value
            _CACHE_MISSES += 1
            if len(_EVALUATIONS) >= MAX_CACHED_EVALUATIONS:
                _EVALUATIONS.popitem(last=False)
            _EVALUATIONS[key] = prediction
            resolved[key] = prediction
    return [resolved[key] for key in keys]


def _backpropagate(path: list[SearchNode], leaf_value: float, added_nodes: int = 0) -> None:
    value = leaf_value
    proof_can_propagate = path[-1].proven is not None
    for node in reversed(path):
        if node is not path[-1]:
            node.size += added_nodes
        children = node._children
        if node is not path[-1] and proof_can_propagate and node.proven is None and children:
            proofs = [child.proven for child in children.values()]
            if -1.0 in proofs:
                node.proven = 1.0
            elif (
                node._moves is not None
                and len(children) == len(node._moves)
                and all(proof is not None for proof in proofs)
            ):
                node.proven = max(-proof for proof in proofs if proof is not None)
        proof_can_propagate = node.proven is not None
        if node.proven is not None:
            value = node.proven
        node.visits += 1
        node.value_sum += value
        value = -value
