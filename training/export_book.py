"""Export a compact opening table from team-generated teacher labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def majority_book(keys: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if len(keys) != len(actions):
        raise ValueError("book key/action arrays have different lengths")
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    actions = actions[order]
    unique_keys, starts, counts = np.unique(keys, return_index=True, return_counts=True)
    selected_actions = np.empty(len(unique_keys), dtype=np.uint16)
    conflicts = 0
    for row, (start, count) in enumerate(zip(starts, counts, strict=True)):
        candidates, votes = np.unique(actions[start : start + count], return_counts=True)
        selected_actions[row] = candidates[int(votes.argmax())]
        conflicts += int(len(candidates) > 1)
    return unique_keys, selected_actions, conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries: dict[int, int] = {}
    conflicts = 0
    for source in args.dataset:
        with np.load(source) as archive:
            if "position_keys" in archive and "best_actions" in archive:
                source_keys = archive["position_keys"]
                source_actions = archive["best_actions"]
            elif "keys" in archive and "actions" in archive:
                source_keys = archive["keys"]
                source_actions = archive["actions"]
            else:
                raise ValueError(f"dataset does not contain book labels: {source}")
            keys, actions, source_conflicts = majority_book(
                source_keys.astype(np.uint64),
                source_actions.astype(np.uint16),
            )
        conflicts += source_conflicts
        entries.update(zip(map(int, keys), map(int, actions), strict=True))
    unique_keys = np.asarray(sorted(entries), dtype=np.uint64)
    selected_actions = np.asarray([entries[int(key)] for key in unique_keys], dtype=np.uint16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, keys=unique_keys, actions=selected_actions)
    print(
        f"saved {len(unique_keys)} book positions to {args.output} "
        f"({args.output.stat().st_size / 1_000_000:.2f} MB, "
        f"{conflicts} majority-resolved conflicts)"
    )


if __name__ == "__main__":
    main()
