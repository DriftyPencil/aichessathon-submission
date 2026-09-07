"""Sharpen stored teacher policy targets using their recorded best actions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-mix", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.hard_mix <= 1.0:
        raise ValueError("--hard-mix must be between 0 and 1")

    with np.load(args.input) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if "best_actions" not in arrays:
        raise ValueError("dataset has no best_actions targets")
    policies = arrays["policies"].astype(np.float32, copy=True)
    best_actions = arrays["best_actions"].astype(np.int64)
    rows = np.arange(len(policies))
    policies *= 1.0 - args.hard_mix
    policies[rows, best_actions] += args.hard_mix
    arrays["policies"] = policies.astype(np.float16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(
        f"saved {args.output} ({args.output.stat().st_size / 1_000_000:.1f} MB, "
        f"hard-mix={args.hard_mix:.2f})"
    )


if __name__ == "__main__":
    main()
