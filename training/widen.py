"""Expand a team-trained residual network without discarding its learned function."""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import numpy as np
import torch
from torch import Tensor

from az_model import INPUT_PLANES, POLICY_PLANES, AlphaZeroLite, encode_board, model_from_state


def widen_model(source: AlphaZeroLite, channels: int, blocks: int) -> AlphaZeroLite:
    old = source.state_dict()
    old_channels = old["stem.0.weight"].shape[0]
    old_blocks = len(source.tower)
    if channels < old_channels or blocks < old_blocks:
        raise ValueError("widening cannot remove channels or residual blocks")
    target = AlphaZeroLite(channels=channels, blocks=blocks)
    state = target.state_dict()
    mapping = torch.arange(channels) % old_channels

    def convolution(name: str, inputs: Tensor, outputs: Tensor) -> None:
        weight = old[f"{name}.weight"]
        counts = torch.bincount(inputs, minlength=weight.shape[1])
        # Replicated inputs share their original outgoing weight equally.
        state[f"{name}.weight"] = (
            weight[outputs][:, inputs] / counts[inputs].reshape(1, -1, 1, 1)
        )
        if f"{name}.bias" in old:
            state[f"{name}.bias"] = old[f"{name}.bias"][outputs]

    def batch_norm(name: str, outputs: Tensor) -> None:
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            state[f"{name}.{suffix}"] = old[f"{name}.{suffix}"][outputs]
        state[f"{name}.num_batches_tracked"] = old[f"{name}.num_batches_tracked"].clone()

    convolution("stem.0", torch.arange(INPUT_PLANES), mapping)
    batch_norm("stem.1", mapping)
    for block in range(old_blocks):
        for layer in (1, 2):
            convolution(f"tower.{block}.conv{layer}", mapping, mapping)
            batch_norm(f"tower.{block}.bn{layer}", mapping)
    for block in range(old_blocks, blocks):
        # A zero residual preserves the existing nonnegative feature stream.
        state[f"tower.{block}.bn2.weight"].zero_()
        state[f"tower.{block}.bn2.bias"].zero_()
    convolution("policy_head", mapping, torch.arange(POLICY_PLANES))
    convolution("value_conv", mapping, torch.arange(8))
    batch_norm("value_bn", torch.arange(8))
    for name in ("value_fc1", "value_fc2"):
        for suffix in ("weight", "bias"):
            state[f"{name}.{suffix}"] = old[f"{name}.{suffix}"].clone()
    target.load_state_dict(state)
    return target.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("weights/az_lite.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20260906)
    source = model_from_state(torch.load(args.source, map_location="cpu", weights_only=True)).eval()
    target = widen_model(source, args.channels, args.blocks)
    board = chess.Board()
    positions = [encode_board(board)]
    rng = np.random.default_rng(20260906)
    for _ in range(48):
        if board.is_game_over(claim_draw=True):
            board.reset()
        legal = list(board.legal_moves)
        board.push(legal[int(rng.integers(len(legal)))])
        positions.append(encode_board(board))
    with torch.inference_mode():
        inputs = torch.from_numpy(np.stack(positions))
        expected = source(inputs)
        actual = target(inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target.state_dict(), args.output)
    print(
        f"Preserved outputs on {len(positions)} positions; saved "
        f"{args.channels} channels, {args.blocks} blocks to {args.output} "
        f"({args.output.stat().st_size / 1_000_000:.2f} MB)"
    )


if __name__ == "__main__":
    main()
