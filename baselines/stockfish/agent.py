"""Development-only Stockfish yardstick. This directory is never packaged for submission."""

import atexit
import os
import shutil

import chess
import chess.engine

_EXECUTABLE = shutil.which("stockfish")
if _EXECUTABLE is None:
    raise RuntimeError("Stockfish is not installed")

_NODES = int(os.environ.get("STOCKFISH_NODES", "1500"))
_ENGINE = chess.engine.SimpleEngine.popen_uci(_EXECUTABLE)
_ENGINE.configure({"Threads": 1, "Hash": 64})
atexit.register(_ENGINE.quit)


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    result = _ENGINE.play(board, chess.engine.Limit(nodes=_NODES))
    if result.move is None:
        return "0000"
    return result.move.uci()
