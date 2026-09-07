"""Compare one-core TorchScript and ONNX Runtime inference on the same checkpoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from az_model import INPUT_PLANES, model_from_state


def measure(name: str, run: object, inputs: np.ndarray, repeats: int) -> None:
    function = run
    assert callable(function)
    for _ in range(20):
        function(inputs)
    started = time.perf_counter()
    for _ in range(repeats):
        function(inputs)
    elapsed = time.perf_counter() - started
    print(
        f"{name:>11} batch={len(inputs):>2}: {elapsed * 1e6 / repeats:>7.0f} us/batch, "
        f"{elapsed * 1e6 / (repeats * len(inputs)):>6.0f} us/position"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/az_lite.pt"))
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=500)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = model_from_state(torch.load(args.weights, map_location="cpu", weights_only=True)).eval()
    traced = torch.jit.freeze(torch.jit.trace(model, torch.zeros(1, INPUT_PLANES, 8, 8)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        args.onnx,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    rng = np.random.default_rng(20260906)
    for batch in (1, 2, 4, 8, 16):
        inputs = rng.standard_normal((batch, INPUT_PLANES, 8, 8), dtype=np.float32)

        def torch_run(array: np.ndarray) -> object:
            with torch.inference_mode():
                return traced(torch.from_numpy(array))

        def onnx_run(array: np.ndarray) -> object:
            return session.run(None, {"state": array})

        measure("TorchScript", torch_run, inputs, args.repeats)
        measure("ONNX", onnx_run, inputs, args.repeats)


if __name__ == "__main__":
    main()
