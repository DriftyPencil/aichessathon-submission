"""Calibrate and quantize an exported team-trained network for CPU inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


class StateReader(CalibrationDataReader):
    def __init__(self, states: np.ndarray, batch_size: int) -> None:
        self._batches = iter(
            {"state": states[start : start + batch_size]}
            for start in range(0, len(states), batch_size)
        )

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._batches, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--format", choices=("qdq", "qoperator"), default="qdq")
    parser.add_argument("--activation", choices=("qint8", "quint8"), default="quint8")
    parser.add_argument(
        "--method", choices=("minmax", "entropy", "percentile"), default="minmax"
    )
    args = parser.parse_args()
    with np.load(args.dataset) as archive:
        available = len(archive["states"])
        count = min(args.positions, available)
        indices = np.random.default_rng(args.seed).choice(available, count, replace=False)
        states = archive["states"][np.sort(indices)].astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        args.model,
        args.output,
        StateReader(states, args.batch_size),
        quant_format=QuantFormat.QDQ if args.format == "qdq" else QuantFormat.QOperator,
        activation_type=QuantType.QInt8 if args.activation == "qint8" else QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method={
            "minmax": CalibrationMethod.MinMax,
            "entropy": CalibrationMethod.Entropy,
            "percentile": CalibrationMethod.Percentile,
        }[args.method],
    )
    print(f"saved {args.output} ({args.output.stat().st_size / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    main()
