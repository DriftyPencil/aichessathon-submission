"""Measure decision and value drift between two ONNX policy/value networks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def run(
    model: ort.InferenceSession, states: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    outputs = [
        model.run(None, {"state": states[start : start + batch_size]})
        for start in range(0, len(states), batch_size)
    ]
    return (
        np.concatenate([item[0] for item in outputs]),
        np.concatenate([item[1] for item in outputs]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    with np.load(args.dataset) as archive:
        states = archive["states"][-args.positions :].astype(np.float32)
        targets = archive["values"][-args.positions :].astype(np.float32)
        legal = archive["legal_actions"][-args.positions :].astype(np.int64)
        counts = archive["legal_counts"][-args.positions :].astype(np.int64)
    reference_policy, reference_value = run(session(args.reference), states, args.batch_size)
    candidate_policy, candidate_value = run(session(args.candidate), states, args.batch_size)
    matching = 0
    kls: list[float] = []
    for row, count in enumerate(counts):
        actions = legal[row, :count]
        first = reference_policy[row, actions]
        second = candidate_policy[row, actions]
        matching += int(first.argmax() == second.argmax())
        first_probability = np.exp(first - first.max())
        first_probability /= first_probability.sum()
        second_probability = np.exp(second - second.max())
        second_probability /= second_probability.sum()
        kls.append(
            float(
                np.sum(
                    first_probability
                    * np.log(
                        np.maximum(first_probability, 1e-12)
                        / np.maximum(second_probability, 1e-12)
                    )
                )
            )
        )
    drift = candidate_value - reference_value
    print(f"legal top-move agreement: {matching / len(states):.2%}")
    print(f"mean legal policy KL: {np.mean(kls):.6f}")
    print(f"value MAE / max: {np.mean(np.abs(drift)):.5f} / {np.max(np.abs(drift)):.5f}")
    print(f"reference target MSE: {np.mean((reference_value - targets) ** 2):.5f}")
    print(f"candidate target MSE: {np.mean((candidate_value - targets) ** 2):.5f}")


if __name__ == "__main__":
    main()
