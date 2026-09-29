#!/usr/bin/env python3
"""Open the simulator's desktop window.

    python run_gui.py

Needs tkinter. On Debian and Ubuntu that is a separate package:

    sudo apt install -y python3-tk
"""

import sys

try:
    import tkinter  # noqa: F401
except ImportError:
    sys.exit(
        "tkinter is not installed.\n\n"
        "On Debian or Ubuntu:   sudo apt install -y python3-tk\n"
        "On Fedora:             sudo dnf install -y python3-tkinter\n"
        "On Arch:               sudo pacman -S tk\n\n"
        "Everything else in this project works without it; only this window "
        "needs it. Try `python run_demo.py` or `python -m lapsim recipe` "
        "meanwhile."
    )

from lapsim.gui import launch

raise SystemExit(launch())
