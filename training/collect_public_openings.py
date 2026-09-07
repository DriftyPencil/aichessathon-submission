"""Collect publicly visible rated starting positions from competition game pages."""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import re
import urllib.parse
import urllib.request
from pathlib import Path

import chess

BASE_URL = "https://aichessathon.com"
TEAM_PATTERN = re.compile(r'href="/team/([0-9a-f-]+)')
GAME_PATTERN = re.compile(r'href="/game/([0-9a-f-]+)')
FEN_PATTERN = re.compile(
    r'\[FEN\s+(?:\\?")([prnbqkPRNBQK1-8/]+\s+[wb]\s+(?:-|[KQkq]+)\s+'
    r'(?:-|[a-h][36])\s+\d+\s+\d+)'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams", type=int, default=40)
    parser.add_argument("--team-offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--output", type=Path, default=Path("training/rated_openings.txt")
    )
    return parser.parse_args()


def fetch(path: str, timeout: float) -> str:
    request = urllib.request.Request(
        BASE_URL + path,
        headers={"User-Agent": "AI-Chessathon-training-data-collector/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def unique_matches(pattern: re.Pattern[str], source: str) -> list[str]:
    return list(dict.fromkeys(pattern.findall(source)))


def fetch_many(paths: list[str], workers: int, timeout: float) -> list[str]:
    pages: list[str] = []
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, path, timeout): path for path in paths}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            try:
                pages.append(future.result())
            except (OSError, TimeoutError) as error:
                failures += 1
                print(f"fetch failed for {futures[future]}: {error}", flush=True)
            if completed % 50 == 0 or completed == len(futures):
                print(
                    f"fetched {completed}/{len(futures)} pages ({failures} failures)",
                    flush=True,
                )
    return pages


def extract_fen(source: str) -> str | None:
    decoded = urllib.parse.unquote(html.unescape(source))
    match = FEN_PATTERN.search(decoded)
    if match is None:
        return None
    fen = match.group(1)
    chess.Board(fen)
    return fen


def main() -> None:
    args = parse_args()
    if args.teams < 1 or args.workers < 1 or args.team_offset < 0:
        raise ValueError("--teams/workers must be positive and --team-offset non-negative")

    leaderboard = fetch("/leaderboard", args.timeout)
    all_team_ids = unique_matches(TEAM_PATTERN, leaderboard)
    team_ids = all_team_ids[args.team_offset : args.team_offset + args.teams]
    print(
        f"collecting games from {len(team_ids)} teams at offset {args.team_offset}",
        flush=True,
    )
    team_pages = fetch_many(
        [f"/team/{team_id}" for team_id in team_ids], args.workers, args.timeout
    )
    game_ids = list(
        dict.fromkeys(
            game_id for page in team_pages for game_id in unique_matches(GAME_PATTERN, page)
        )
    )
    print(f"found {len(game_ids)} distinct public games", flush=True)
    game_pages = fetch_many(
        [f"/game/{game_id}" for game_id in game_ids], args.workers, args.timeout
    )

    fens = {fen for page in game_pages if (fen := extract_fen(page)) is not None}
    if args.output.exists():
        fens.update(
            line.strip()
            for line in args.output.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    for fen in fens:
        board = chess.Board(fen)
        if board.outcome(claim_draw=True) is not None:
            raise ValueError(f"terminal opening found: {fen}")
    ordered = sorted(fens, key=lambda fen: (chess.Board(fen).fullmove_number, fen))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(ordered) + "\n")
    print(f"saved {len(ordered)} unique rated openings to {args.output}", flush=True)


if __name__ == "__main__":
    main()
