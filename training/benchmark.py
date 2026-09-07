"""Paired opening matches through the unmodified harness, with retained evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import chess

from harness.package import members
from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import BASE_MS, INCREMENT_MS, PLY_CAP
from harness.sandbox import RUNNER, Agent

OPENINGS = (
    ("start", ""),
    ("italian", "e4 e5 Nf3 Nc6 Bc4 Bc5"),
    ("sicilian", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3"),
    ("queens-gambit", "d4 d5 c4 e6 Nc3 Nf6 Nf3 Be7"),
    ("french", "e4 e6 d4 d5 Nc3 Nf6"),
    ("caro-kann", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5"),
)


def seeded_agent(directory: Path, seed: int) -> Agent:
    code = (
        "import random, runpy, sys; "
        "random.seed(int(sys.argv.pop(1))); "
        "script = sys.argv.pop(1); "
        "runpy.run_path(script, run_name='__main__')"
    )
    return Agent([sys.executable, "-c", code, str(seed), str(RUNNER), str(directory.resolve())])


def fingerprint(directory: Path) -> dict[str, str]:
    paths = [source for source, _ in members(directory, ("weights",))]
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.exists()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/basic"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--base-ms", type=int, default=BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=INCREMENT_MS)
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument("--opening-offset", type=int, default=0)
    parser.add_argument("--opening-fens", type=Path)
    parser.add_argument("--max-losses", type=int, default=0)
    parser.add_argument("--max-draws", type=int, default=1)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error("--games must be a positive number of opening pairs")
    opening_fens: tuple[str, ...] = ()
    if args.opening_fens is not None:
        opening_fens = tuple(
            line.strip()
            for line in args.opening_fens.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if not opening_fens:
            parser.error("--opening-fens did not contain any positions")
        for fen in opening_fens:
            chess.Board(fen)
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "agent": fingerprint(args.agent),
        "opponent": fingerprint(args.opponent),
        "base_ms": args.base_ms,
        "increment_ms": args.increment_ms,
        "ply_cap": PLY_CAP,
        "seed": args.seed,
        "opening_offset": args.opening_offset,
        "opening_fens": str(args.opening_fens) if args.opening_fens is not None else None,
        "max_losses": args.max_losses,
        "max_draws": args.max_draws,
        "min_score": args.min_score,
        "games": [],
    }
    games: list[dict[str, object]] = []
    wins = draws = losses = failures = 0
    for index in range(args.games):
        opening_index = args.opening_offset + index // 2
        if opening_fens:
            rated_index = opening_index % len(opening_fens)
            name = f"rated-{rated_index + 1:02d}"
            board = chess.Board(opening_fens[rated_index])
        else:
            name, opening = OPENINGS[opening_index % len(OPENINGS)]
            board = chess.Board()
            for san in opening.split():
                board.push_san(san)
        as_white = index % 2 == 0
        seed = args.seed + index // 2
        white, black = (args.agent, args.opponent) if as_white else (args.opponent, args.agent)
        outcome = play_match(
            seeded_agent(white, seed),
            seeded_agent(black, seed),
            args.base_ms,
            args.increment_ms,
            start_fen=board.fen(),
        )
        failure = outcome.termination in FAILED_TERMINATIONS
        failures += int(failure)
        if outcome.result in ("draw", "void"):
            draws += 1
            result = "draw"
        elif (outcome.result == "white") == as_white:
            wins += 1
            result = "win"
        else:
            losses += 1
            result = "loss"
        pgn_name = f"{index + 1:03d}-{name}-{'white' if as_white else 'black'}.pgn"
        (args.output / pgn_name).write_text(outcome.pgn + "\n")
        games.append(
            {
                "opening": name,
                "fen": board.fen(),
                "as_white": as_white,
                "result": result,
                "termination": outcome.termination,
                "seed": seed,
                "pgn": pgn_name,
            }
        )
        score = (wins + draws / 2) / len(games)
        passed = (
            failures == 0
            and losses <= args.max_losses
            and draws <= args.max_draws
            and score >= args.min_score
        )
        report.update(
            games=games,
            wins=wins,
            draws=draws,
            losses=losses,
            score=score,
            failures=failures,
            complete=len(games) == args.games,
            passed=passed and len(games) == args.games,
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{index + 1}/{args.games} {name} {result} ({outcome.termination}): "
            f"+{wins} ={draws} -{losses}",
            flush=True,
        )
    print(f"score={score:.1%}, gate={'PASS' if passed else 'FAIL'}", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
