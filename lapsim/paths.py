"""Where the simulator keeps the files it writes.

Run from a clone, figures go in the repository's ``figures`` folder.  Packaged
as an executable with PyInstaller, the code runs from a temporary folder that is
deleted when the program closes, so figures go in a ``figures`` folder beside
the ``.exe`` instead, where they survive and can be found.

Deliberately free of heavy imports: the window reads this even when matplotlib
is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

IS_FROZEN = bool(getattr(sys, "frozen", False))
"""True when running as a packaged executable."""

APP_ROOT = (
    Path(sys.executable).resolve().parent
    if IS_FROZEN
    else Path(__file__).resolve().parent.parent
)
"""The repository root, or the folder holding the executable."""

FIGURE_DIR = APP_ROOT / "figures"
