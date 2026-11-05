"""Entry point for the HOMIE-base DreamWaQ deployment."""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
_DEPLOY = _ROOT / "deploy"
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))

from run_dreamwaq import main  # noqa: E402

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
