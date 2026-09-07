"""Export a team-trained policy/value checkpoint for the preinstalled ONNX Runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from az_model import INPUT_PLANES, POLICY_SIZE, model_from_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/az_lite.pt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = model_from_state(torch.load(args.weights, map_location="cpu", weights_only=True)).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.zeros(1, INPUT_PLANES, 8, 8),),
        args.output,
        input_names=["state"],
        output_names=["policy", "value"],
        dynamic_axes={
            "state": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        dynamo=False,
        opset_version=18,
    )
    print(f"saved {args.output} ({args.output.stat().st_size / 1_000_000:.2f} MB)")
    print(f"policy actions: {POLICY_SIZE}")


if __name__ == "__main__":
    main()
