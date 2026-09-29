#!/usr/bin/env python3
"""Open the simulator's desktop window.

    python run_gui.py            open the window
    python run_gui.py --check    check everything the window needs, then exit

Needs tkinter. On Debian and Ubuntu that is a separate package:

    sudo apt install -y python3-tk

``--check`` opens no window.  It imports the window's code, simulates one
recipe and renders one figure into the figures folder, then exits with 0 if all
of that worked.  ``build_exe.cmd`` runs it against the finished ``LapSim.exe``
to prove the packaging picked up numpy, scipy, matplotlib and tkinter.  A
packaged window has no console, so the outcome is also written to
``lapsim-check.log`` beside the program.
"""

import sys


def check() -> int:
    """Exercise the engine, the plotting and the window's imports, headless."""
    from lapsim.paths import APP_ROOT

    try:
        import matplotlib

        matplotlib.use("Agg")
        import lapsim.gui  # noqa: F401  (pulls in tkinter without opening a window)
        from lapsim.plots import plot_vehicle
        from lapsim.simulate import simulate

        result = simulate(grit="600")
        figure = plot_vehicle()
        message = (f"ok: 600 grit cuts {result.removal_rate_um_min:.1f} um/min; "
                   f"figure written to {figure}")
        status = 0
    except Exception:                                   # noqa: BLE001
        import traceback

        message = "FAILED\n" + traceback.format_exc()
        status = 1

    (APP_ROOT / "lapsim-check.log").write_text(message + "\n", encoding="utf-8")
    print(message)   # silently skipped by a packaged window, which has no console
    return status


if "--check" in sys.argv[1:]:
    raise SystemExit(check())

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
