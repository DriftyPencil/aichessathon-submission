"""Small policy/value network and encodings shared by training and the agent."""

from __future__ import annotations

import chess
import numpy as np
import torch
from torch import Tensor, nn

INPUT_PLANES = 18
POLICY_PLANES = 73
POLICY_SIZE = 64 * POLICY_PLANES
MODEL_CHANNELS = 48
MODEL_BLOCKS = 3

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


def mirror_state(state: np.ndarray) -> np.ndarray:
    """Reflect a canonical board across the vertical file axis."""
    mirrored = state[:, :, ::-1].copy()
    mirrored[12] = state[13, :, ::-1]
    mirrored[13] = state[12, :, ::-1]
    mirrored[14] = state[15, :, ::-1]
    mirrored[15] = state[14, :, ::-1]
    return mirrored


def mirror_policy(policy: np.ndarray) -> np.ndarray:
    """Reflect an 8 x 8 x 73 policy tensor across the vertical file axis."""
    return policy.reshape(-1)[MIRROR_ACTION_INDICES].reshape(policy.shape)


def mirror_action_index(index: int) -> int:
    """Reflect one encoded policy action across the vertical file axis."""
    source_square, plane = divmod(index, POLICY_PLANES)
    rank, file_index = divmod(source_square, 8)
    target_square = rank * 8 + 7 - file_index
    if plane < 56:
        direction_index, distance = divmod(plane, 7)
        file_delta, rank_delta = _QUEEN_DIRECTIONS[direction_index]
        mirrored_direction = _QUEEN_DIRECTIONS.index((-file_delta, rank_delta))
        target_plane = mirrored_direction * 7 + distance
    elif plane < 64:
        direction_index = plane - 56
        file_delta, rank_delta = _KNIGHT_DIRECTIONS[direction_index]
        mirrored_direction = _KNIGHT_DIRECTIONS.index((-file_delta, rank_delta))
        target_plane = 56 + mirrored_direction
    else:
        promotion_direction, promotion = divmod(plane - 64, 3)
        target_plane = 64 + (2 - promotion_direction) * 3 + promotion
    return target_square * POLICY_PLANES + target_plane


MIRROR_ACTION_INDICES = np.asarray(
    [mirror_action_index(index) for index in range(POLICY_SIZE)], dtype=np.int64
)


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


def action_index(board: chess.Board, move: chess.Move) -> int:
    """Map a legal move to one of AlphaZero's 8 x 8 x 73 policy actions."""
    turn = board.turn
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
        raise ValueError(f"move has no displacement: {move.uci()}")
    queen_direction = (file_delta // distance, rank_delta // distance)
    try:
        direction_index = _QUEEN_DIRECTIONS.index(queen_direction)
    except ValueError as error:
        raise ValueError(f"cannot encode move: {move.uci()}") from error
    plane = direction_index * 7 + distance - 1
    return from_square * POLICY_PLANES + plane


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return torch.relu(outputs + residual)


class AlphaZeroLite(nn.Module):
    """A compact residual network sized for single-core CPU inference."""

    def __init__(self, channels: int = MODEL_CHANNELS, blocks: int = MODEL_BLOCKS) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy_head = nn.Conv2d(channels, POLICY_PLANES, 1)
        self.value_conv = nn.Conv2d(channels, 8, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU(inplace=True)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.value_fc1.weight)
        nn.init.zeros_(self.value_fc1.bias)
        nn.init.xavier_uniform_(self.value_fc2.weight)
        nn.init.zeros_(self.value_fc2.bias)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        features = self.tower(self.stem(inputs))
        policy = self.policy_head(features)
        policy = policy.permute(0, 2, 3, 1).reshape(inputs.shape[0], POLICY_SIZE)
        value = self.relu(self.value_bn(self.value_conv(features)))
        value = self.relu(self.value_fc1(value.flatten(1)))
        return policy, torch.tanh(self.value_fc2(value)).squeeze(1)


def model_from_state(state: dict[str, Tensor]) -> AlphaZeroLite:
    """Recover the width and depth of a team-trained state dictionary."""
    channels = state["stem.0.weight"].shape[0]
    blocks = len({name.split(".")[1] for name in state if name.startswith("tower.")})
    model = AlphaZeroLite(channels=channels, blocks=blocks)
    model.load_state_dict(state)
    return model
