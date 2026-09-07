"""Development-only Stockfish opponent configured for a nominal 2400 UCI Elo."""

import atexit
import shutil

import chess
import chess.engine

_EXECUTABLE = shutil.which("stockfish")
if _EXECUTABLE is None:
    raise RuntimeError("Stockfish is not installed")

_ENGINE = chess.engine.SimpleEngine.popen_uci(_EXECUTABLE)
_ENGINE.configure(
    {
        "Threads": 1,
        "Hash": 64,
        "UCI_LimitStrength": True,
        "UCI_Elo": 2400,
    }
)
atexit.register(_ENGINE.quit)


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    seconds = max(0.01, time_left_ms / 1000.0)
    result = _ENGINE.play(
        board,
        chess.engine.Limit(
            white_clock=seconds,
            black_clock=seconds,
            white_inc=0.5,
            black_inc=0.5,
        ),
    )
    return "0000" if result.move is None else result.move.uci()
