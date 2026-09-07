"""Run the identical frozen agent against every shipped starter and retain all evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harness.rules import BASE_MS, INCREMENT_MS
from training.benchmark import fingerprint

STARTERS = ("numba", "basic", "minimax", "greedy", "random")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=4901)
    parser.add_argument("--base-ms", type=int, default=BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=INCREMENT_MS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output directory; prior evidence is never overwritten")
    args.output.mkdir(parents=True)
    expected = fingerprint(args.agent)
    report: dict[str, object] = {
        "agent": expected,
        "workers": args.workers,
        "base_ms": args.base_ms,
        "increment_ms": args.increment_ms,
        "seed": args.seed,
        "complete": False,
        "passed": False,
        "results": {},
    }
    results: dict[str, object] = {}

    def run(index: int, opponent: str) -> tuple[str, dict[str, object]]:
        destination = args.output / opponent
        command = [
            sys.executable,
            "-u",
            "-m",
            "training.benchmark",
            "--agent",
            str(args.agent),
            "--opponent",
            str(Path("baselines") / opponent),
            "--output",
            str(destination),
            "--seed",
            str(args.seed + index * 1000),
            "--base-ms",
            str(args.base_ms),
            "--increment-ms",
            str(args.increment_ms),
        ]
        with (
            (args.output / f"{opponent}.log").open("w") as log,
            subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            as process,
        ):
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"{opponent}: {line.rstrip()}", flush=True)
            returncode = process.wait()
        result_path = destination / "report.json"
        if not result_path.exists():
            return opponent, {"complete": False, "passed": False, "returncode": returncode}
        result = json.loads(result_path.read_text())
        result["unchanged_agent"] = result["agent"] == expected == fingerprint(args.agent)
        result["passed"] = bool(result["passed"] and result["unchanged_agent"] and returncode == 0)
        return opponent, result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, index, opponent) for index, opponent in enumerate(STARTERS)]
        for future in as_completed(futures):
            opponent, result = future.result()
            results[opponent] = result
            report["results"] = results
            report["complete"] = len(results) == len(STARTERS)
            report["passed"] = report["complete"] and all(
                isinstance(value, dict) and value["passed"] for value in results.values()
            )
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Full starter suite: {'PASS' if report['passed'] else 'FAIL'}", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
