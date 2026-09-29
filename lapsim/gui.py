"""A desktop window for the simulator, built with tkinter.

Run it with ``python run_gui.py``.

The layout is the usual desktop form: settings down the left, results in tabs on
the right, a status line along the bottom.

Two speeds of work happen here, and they are handled differently:

*   **A single simulation takes about fifteen milliseconds**, so everything on
    the left re-runs as you move it and the readouts follow the sliders. A short
    debounce keeps a dragged slider from queueing dozens of runs.
*   **The searches evaluate a thousand mixes or more** and take seconds. Those
    sit behind buttons and run on a worker thread, because tkinter redraws
    nothing while a callback is running -- do that work inline and the window
    freezes, including the status message that is supposed to say it is busy.

Results come back through a queue that the UI thread polls, which is the only
safe way to touch widgets from another thread.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from lapsim.materials import ABRASIVES, LAPS, STONES, parse_micron_label
from lapsim.style import margin_hex
from lapsim.simulate import Result, Setup, simulate

DROP_ML = 0.05
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# The recipe window, drawn on a canvas
# --------------------------------------------------------------------------- #

class WindowMap(ttk.Frame):
    """Grit loading across, glycerine up, colour showing margin to the limits.

    Click anywhere to load that mix into the controls.
    """

    PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 56, 18, 16, 46

    def __init__(self, parent, on_pick):
        super().__init__(parent)
        self._on_pick = on_pick
        self._scan = None
        self._image: tk.PhotoImage | None = None
        self._marker = (float("nan"), float("nan"))

        self.canvas = tk.Canvas(self, background="#fcfcfb", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        self.canvas.bind("<Button-1>", self._clicked)
        self.canvas.bind("<Motion>", self._moved)
        self.canvas.bind("<Leave>", lambda _event: self._set_readout(None))
        self._hover = None

    def set_scan(self, scan) -> None:
        self._scan = scan
        self._image = None
        self.redraw()

    def set_marker(self, solids_pct: float, glycerine_pct: float) -> None:
        self._marker = (solids_pct, glycerine_pct)
        self.redraw()

    # ------------------------------------------------------------- geometry
    def _plot_box(self) -> tuple[int, int, int, int]:
        width = max(self.canvas.winfo_width(), 200)
        height = max(self.canvas.winfo_height(), 160)
        return (self.PAD_LEFT, self.PAD_TOP,
                width - self.PAD_RIGHT, height - self.PAD_BOTTOM)

    def _value_to_pixel(self, solids: float, glycerine: float) -> tuple[float, float]:
        left, top, right, bottom = self._plot_box()
        x_axis = self._scan.solids_mass_fractions * 100
        y_axis = self._scan.glycerine_mass_fractions * 100
        fx = (solids - x_axis[0]) / max(x_axis[-1] - x_axis[0], 1e-9)
        fy = (glycerine - y_axis[0]) / max(y_axis[-1] - y_axis[0], 1e-9)
        return left + fx * (right - left), bottom - fy * (bottom - top)

    def _pixel_to_cell(self, x: int, y: int) -> tuple[int, int] | None:
        left, top, right, bottom = self._plot_box()
        if not (left <= x <= right and top <= y <= bottom):
            return None
        columns = len(self._scan.solids_mass_fractions)
        rows = len(self._scan.glycerine_mass_fractions)
        column = round((x - left) / max(right - left, 1) * (columns - 1))
        row = round((bottom - y) / max(bottom - top, 1) * (rows - 1))
        return max(0, min(column, columns - 1)), max(0, min(row, rows - 1))

    # -------------------------------------------------------------- drawing
    def redraw(self) -> None:
        self.canvas.delete("all")
        if self._scan is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
                text='Press "Map the window" to scan every mix for this grade.',
                fill="#52514e", font=("TkDefaultFont", 10))
            return

        left, top, right, bottom = self._plot_box()
        self._draw_surface(left, top, right, bottom)
        self.canvas.create_rectangle(left, top, right, bottom, outline="#c3c2b7")
        self._draw_axes(left, top, right, bottom)
        self._draw_marker()
        self._draw_readout(left, top)

    def _draw_surface(self, left: int, top: int, right: int, bottom: int) -> None:
        """Paint the grid as a small image, then let Tk scale it up.

        Building the image once and zooming beats drawing a thousand rectangles,
        and it survives a window resize without recomputing any colours.
        """
        robustness = self._scan.robustness
        rows, columns = robustness.shape
        if self._image is None:
            image = tk.PhotoImage(width=columns, height=rows)
            # Rows top-down on screen, so the glycerine axis is reversed.
            pixels = " ".join(
                "{" + " ".join(margin_hex(value) for value in robustness[row]) + "}"
                for row in reversed(range(rows))
            )
            image.put(pixels)
            self._image = image

        scale_x = max(1, (right - left) // columns)
        scale_y = max(1, (bottom - top) // rows)
        scaled = self._image.zoom(scale_x, scale_y)
        # Keep a reference or Tk garbage-collects the image mid-draw.
        self._scaled = scaled
        self.canvas.create_image(left, top, anchor="nw", image=scaled)

    def _draw_axes(self, left: int, top: int, right: int, bottom: int) -> None:
        x_axis = self._scan.solids_mass_fractions * 100
        y_axis = self._scan.glycerine_mass_fractions * 100
        for step in range(6):
            fraction = step / 5

            x = left + fraction * (right - left)
            value = x_axis[0] + fraction * (x_axis[-1] - x_axis[0])
            self.canvas.create_line(x, bottom, x, bottom + 4, fill="#c3c2b7")
            self.canvas.create_text(x, bottom + 12, text=f"{value:.0f}",
                                    fill="#52514e", font=("TkDefaultFont", 8))

            y = bottom - fraction * (bottom - top)
            value = y_axis[0] + fraction * (y_axis[-1] - y_axis[0])
            self.canvas.create_line(left - 4, y, left, y, fill="#c3c2b7")
            self.canvas.create_text(left - 8, y, text=f"{value:.0f}", anchor="e",
                                    fill="#52514e", font=("TkDefaultFont", 8))

        self.canvas.create_text((left + right) // 2, bottom + 30,
                                text="grit loading (wt % of the slurry)",
                                fill="#52514e", font=("TkDefaultFont", 9))
        self.canvas.create_text(14, (top + bottom) // 2, angle=90,
                                text="glycerine in the liquid (wt %)",
                                fill="#52514e", font=("TkDefaultFont", 9))

    def _draw_marker(self) -> None:
        solids, glycerine = self._marker
        if solids != solids or glycerine != glycerine:   # NaN check
            return
        x, y = self._value_to_pixel(solids, glycerine)
        self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7,
                                outline="#ffffff", width=3)
        self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7,
                                outline="#0b0b0b", width=1.6)
        self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#0b0b0b",
                                outline="")

    def _draw_readout(self, left: int, top: int) -> None:
        text = self._hover
        if text is None:
            feasible = self._scan.feasible_fraction * 100
            text = (f"{self._scan.abrasive_name} {self._scan.grit}  -  "
                    f"{feasible:.0f} % of this map works. Click to try a mix.")
        self.canvas.create_rectangle(left + 4, top + 4, left + 10 + 7 * len(text),
                                     top + 22, fill="#fcfcfb", outline="")
        self.canvas.create_text(left + 8, top + 13, anchor="w", text=text,
                                fill="#0b0b0b", font=("TkDefaultFont", 8))

    # --------------------------------------------------------------- events
    def _set_readout(self, text: str | None) -> None:
        if text != self._hover:
            self._hover = text
            self.redraw()

    def _moved(self, event) -> None:
        if self._scan is None:
            return
        cell = self._pixel_to_cell(event.x, event.y)
        if cell is None:
            self._set_readout(None)
            return
        column, row = cell
        margin = self._scan.robustness[row][column]
        verdict = (f"works, margin {margin:+.2f}" if margin > 0
                   else f"fails on “{self._scan.binding_constraint[row][column]}”")
        self._set_readout(
            f"{self._scan.solids_mass_fractions[column] * 100:.0f} wt% grit, "
            f"{self._scan.glycerine_mass_fractions[row] * 100:.0f} % glycerine "
            f"- {verdict}, {self._scan.removal_rate_um_min[row][column]:.1f} um/min")

    def _clicked(self, event) -> None:
        if self._scan is None:
            return
        cell = self._pixel_to_cell(event.x, event.y)
        if cell is None:
            return
        column, row = cell
        self._on_pick(self._scan.solids_mass_fractions[column] * 100,
                      self._scan.glycerine_mass_fractions[row] * 100)


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #


class SimulatorWindow(ttk.Frame):
    """Settings on the left, results in tabs on the right, status along the bottom."""

    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=0)
        self.master.title("Lapping slurry simulator")
        self.master.geometry("1320x840")
        self.master.minsize(1060, 700)
        self.pack(fill="both", expand=True)

        self._results: queue.Queue = queue.Queue()
        self._pending_update: str | None = None
        self._busy = False
        self._loading = True
        self._figure_image: tk.PhotoImage | None = None

        self._use_native_theme()
        self._build_variables()
        self._build_layout()

        self._loading = False
        self._poll_results()
        self._queue_simulation()

    # --------------------------------------------------------------- styling
    def _use_native_theme(self) -> None:
        """Prefer the platform's own widget theme over Tk's 1990s default."""
        style = ttk.Style()
        for candidate in ("vista", "winnative", "clam", "default"):
            if candidate in style.theme_names():
                style.theme_use(candidate)
                break
        style.configure("Heading.TLabel", font=("TkDefaultFont", 9, "bold"))
        style.configure("Status.TLabel", foreground="#52514e")

    # ------------------------------------------------------------- variables
    def _build_variables(self) -> None:
        self.abrasive = tk.StringVar()
        self.grit = tk.StringVar()
        self.stone = tk.StringVar()
        self.lap = tk.StringVar()

        defaults = Setup()
        self.solids = tk.DoubleVar(value=defaults.solids_mass_fraction * 100)
        self.glycerine = tk.DoubleVar(value=defaults.glycerine_mass_fraction * 100)
        self.soap_drops = tk.DoubleVar(value=2.0)
        self.load = tk.DoubleVar(value=defaults.normal_load_N)
        self.area = tk.DoubleVar(value=defaults.contact_area_mm2)
        self.speed = tk.DoubleVar(value=defaults.sliding_speed_m_s)
        self.face_length = tk.DoubleVar(value=defaults.contact_length_mm)
        self.temperature = tk.DoubleVar(value=defaults.temperature_C)
        self.humidity = tk.DoubleVar(value=defaults.relative_humidity * 100)
        self.air_speed = tk.DoubleVar(value=defaults.air_speed_m_s)
        self.film_volume = tk.DoubleVar(value=defaults.film_volume_ml)
        self.status = tk.StringVar(value="Ready.")

    # ---------------------------------------------------------------- layout
    def _build_layout(self) -> None:
        self._build_menu()

        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=6, pady=(6, 0))

        left = ttk.Frame(panes, width=330)
        left.pack_propagate(False)
        panes.add(left, weight=0)

        right = ttk.Frame(panes)
        panes.add(right, weight=1)

        self._build_controls(left)
        self._build_tabs(right)

        status = ttk.Frame(self)
        status.pack(fill="x", side="bottom")
        ttk.Separator(status, orient="horizontal").pack(fill="x")
        ttk.Label(status, textvariable=self.status, style="Status.TLabel",
                  anchor="w").pack(fill="x", padx=8, pady=3)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.master)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save report…", command=self._save_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        tools = tk.Menu(menubar, tearoff=0)
        tools.add_command(label="Reset to defaults", command=self._reset_defaults)
        tools.add_command(label="Render figures", command=self._render_figures)
        menubar.add_cascade(label="Tools", menu=tools)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.master.config(menu=menubar)

    def _build_controls(self, parent: ttk.Frame) -> None:
        materials = ttk.LabelFrame(parent, text="Abrasive and materials", padding=8)
        materials.pack(fill="x", padx=4, pady=(0, 6))

        self._abrasive_box = self._combo(
            materials, "Abrasive", self.abrasive,
            [abrasive.name for abrasive in ABRASIVES.values()])
        self._abrasive_box.bind("<<ComboboxSelected>>",
                                lambda _e: self._abrasive_changed())
        # Editable: finer than F1000, type the supplier's micron size, e.g. 2.5um.
        self._grit_box = self._combo(materials, "Grade or µm", self.grit, [],
                                     editable=True)
        self._stone_box = self._combo(
            materials, "Stone", self.stone, [s.name for s in STONES.values()])
        self._lap_box = self._combo(
            materials, "Lap plate", self.lap, [l.name for l in LAPS.values()])

        self.abrasive.set(next(iter(ABRASIVES.values())).name)
        self.stone.set(STONES["agate"].name)
        self.lap.set(LAPS["glass"].name)
        self._refresh_grades()

        slurry = ttk.LabelFrame(parent, text="Slurry", padding=8)
        slurry.pack(fill="x", padx=4, pady=6)
        self._slider(slurry, "Grit loading (wt %)", self.solids, 2, 90, 1)
        self._slider(slurry, "Glycerine in liquid (wt %)", self.glycerine, 0, 95, 1)
        self._spin(slurry, "Dish soap (drops / 100 ml)", self.soap_drops, 0, 20, 1)

        process = ttk.LabelFrame(parent, text="How you work", padding=8)
        process.pack(fill="x", padx=4, pady=6)
        self._spin(process, "Pressing force (N)", self.load, 2, 400, 1)
        self._spin(process, "Face area (mm²)", self.area, 20, 5000, 10)
        self._spin(process, "Stroke speed (m/s)", self.speed, 0.02, 2.0, 0.01)
        self._spin(process, "Face length (mm)", self.face_length, 3, 200, 1)

        room = ttk.LabelFrame(parent, text="The room and the plate", padding=8)
        room.pack(fill="x", padx=4, pady=6)
        self._spin(room, "Temperature (°C)", self.temperature, 5, 40, 1)
        self._spin(room, "Humidity (% RH)", self.humidity, 5, 95, 1)
        self._spin(room, "Air movement (m/s)", self.air_speed, 0, 3, 0.1)
        self._spin(room, "Slurry on plate (ml)", self.film_volume, 1, 60, 1)

        actions = ttk.Frame(parent)
        actions.pack(fill="x", padx=4, pady=(8, 4))
        self._buttons = []
        for text, command in (
            ("Find the best recipe for this grade", self._find_recipe),
            ("Map the window of workable mixes", self._map_window),
            ("Plan the whole grit sequence", self._plan_sequence),
        ):
            button = ttk.Button(actions, text=text, command=command)
            button.pack(fill="x", pady=2)
            self._buttons.append(button)

    def _combo(self, parent, label, variable, values,
               editable: bool = False) -> ttk.Combobox:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=15).pack(side="left")
        box = ttk.Combobox(row, textvariable=variable, values=values,
                           state="normal" if editable else "readonly", width=18)
        box.pack(side="left", fill="x", expand=True)
        box.bind("<<ComboboxSelected>>", lambda _e: self._queue_simulation())
        if editable:
            box.bind("<Return>", lambda _e: self._queue_simulation())
            box.bind("<FocusOut>", lambda _e: self._queue_simulation())
        return box

    def _slider(self, parent, label, variable, low, high, step) -> None:
        """A slider with its own spin box, both bound to one variable."""
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text=label).pack(side="left")
        spin = ttk.Spinbox(row, from_=low, to=high, increment=step, width=6,
                           textvariable=variable, command=self._queue_simulation)
        spin.pack(side="right")
        spin.bind("<KeyRelease>", lambda _e: self._queue_simulation())
        scale = ttk.Scale(parent, from_=low, to=high, variable=variable,
                          orient="horizontal",
                          command=lambda _value: self._slider_moved(variable))
        scale.pack(fill="x", pady=(0, 4))

    def _slider_moved(self, variable: tk.DoubleVar) -> None:
        # ttk.Scale is continuous; round so the readouts show whole percents.
        variable.set(round(variable.get()))
        self._queue_simulation()

    def _spin(self, parent, label, variable, low, high, step) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=22).pack(side="left")
        spin = ttk.Spinbox(row, from_=low, to=high, increment=step, width=8,
                           textvariable=variable, command=self._queue_simulation)
        spin.pack(side="right")
        spin.bind("<KeyRelease>", lambda _e: self._queue_simulation())

    def _build_tabs(self, parent: ttk.Frame) -> None:
        self.tabs = ttk.Notebook(parent)
        self.tabs.pack(fill="both", expand=True)

        # At a glance
        glance = ttk.Frame(self.tabs, padding=4)
        self.glance = ttk.Treeview(glance, columns=("value",), show="tree headings",
                                   height=24)
        self.glance.heading("#0", text="Quantity")
        self.glance.heading("value", text="Value")
        self.glance.column("#0", width=300, stretch=False)
        self.glance.column("value", width=420)
        self._with_scrollbar(glance, self.glance)
        self.tabs.add(glance, text="At a glance")

        # Full report
        self.report = self._text_tab("Full report")

        # Best recipe
        recipe_tab = ttk.Frame(self.tabs, padding=4)
        self.recipe_text = tk.Text(recipe_tab, height=10, wrap="word",
                                   font=("monospace", 10), relief="flat",
                                   background="#ffffff")
        self.recipe_text.pack(fill="x", pady=(0, 6))
        self.margins = ttk.Treeview(
            recipe_tab, columns=("margin", "reading"), show="tree headings")
        self.margins.heading("#0", text="Limit")
        self.margins.heading("margin", text="Margin")
        self.margins.heading("reading", text="Reading")
        self.margins.column("#0", width=280, stretch=False)
        self.margins.column("margin", width=80, anchor="e", stretch=False)
        self.margins.column("reading", width=260)
        self._with_scrollbar(recipe_tab, self.margins)
        self.tabs.add(recipe_tab, text="Best recipe")

        # The recipe window
        map_tab = ttk.Frame(self.tabs, padding=4)
        self.window_map = WindowMap(map_tab, self._picked_from_map)
        self.window_map.pack(fill="both", expand=True)
        self.tabs.add(map_tab, text="Recipe window")

        # The grit sequence
        plan_tab = ttk.Frame(self.tabs, padding=4)
        columns = ("grade", "d50", "Ra", "damage", "cut", "clear", "minutes", "recipe")
        headings = ("Grade", "d50 µm", "Ra µm", "Damage µm",
                    "Cut µm/min", "Clear µm", "Minutes",
                    "Recipe per 100 ml")
        widths = (60, 65, 65, 90, 90, 80, 65, 380)
        self.plan = ttk.Treeview(plan_tab, columns=columns, show="tree headings",
                                 height=10)
        self.plan.heading("#0", text="Abrasive")
        self.plan.column("#0", width=160, stretch=False)
        for column, heading, width in zip(columns, headings, widths):
            self.plan.heading(column, text=heading)
            self.plan.column(column, width=width, anchor="e", stretch=False)
        self.plan.column("recipe", anchor="w")
        self.plan.pack(fill="both", expand=True)
        self.plan_notes = tk.Text(plan_tab, height=9, wrap="word", relief="flat",
                                  background="#ffffff")
        self.plan_notes.pack(fill="x", pady=(6, 0))
        self.tabs.add(plan_tab, text="Grit sequence")

        # Rendered figures
        figures = ttk.Frame(self.tabs, padding=4)
        bar = ttk.Frame(figures)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Button(bar, text="Render figures",
                   command=self._render_figures).pack(side="left")
        self.figure_choice = ttk.Combobox(bar, state="readonly", width=60)
        self.figure_choice.pack(side="left", padx=6, fill="x", expand=True)
        self.figure_choice.bind("<<ComboboxSelected>>",
                                lambda _e: self._show_figure())
        self.figure_canvas = tk.Canvas(figures, background="#f9f9f7",
                                       highlightthickness=0)
        self.figure_canvas.pack(fill="both", expand=True)
        self.tabs.add(figures, text="Figures")
        self._list_existing_figures()

    def _text_tab(self, title: str) -> tk.Text:
        frame = ttk.Frame(self.tabs, padding=4)
        text = tk.Text(frame, wrap="none", font=("monospace", 10), relief="flat",
                       background="#ffffff")
        vertical = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        vertical.pack(side="right", fill="y")
        horizontal.pack(side="bottom", fill="x")
        text.pack(side="left", fill="both", expand=True)
        self.tabs.add(frame, text=title)
        return text

    @staticmethod
    def _with_scrollbar(parent, widget) -> None:
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=widget.yview)
        widget.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        widget.pack(side="left", fill="both", expand=True)

    # ----------------------------------------------------------- the model
    def _lookup(self, registry, name: str):
        """Find a registry entry by its display name."""
        for entry in registry.values():
            if entry.name == name:
                return entry
        return next(iter(registry.values()))

    def _abrasive_changed(self) -> None:
        self._refresh_grades()
        self._queue_simulation()

    def _refresh_grades(self) -> None:
        """List the chosen abrasive's grades, preferring one that is in stock."""
        abrasive = self._lookup(ABRASIVES, self.abrasive.get())
        labels = []
        preferred = None
        for grade in abrasive.grades.values():
            label = (f"{grade.label}  -  d50 {grade.d50_um:g} µm"
                     + ("  (in stock)" if grade.in_stock else ""))
            labels.append(label)
            if grade.in_stock and preferred is None:
                preferred = label
        self._grit_box.configure(values=labels)
        typed_micron = parse_micron_label(self._grade_label()) is not None
        if self.grit.get() not in labels and not typed_micron:
            self.grit.set(preferred or (labels[0] if labels else ""))

    def _grade_label(self) -> str:
        """The grade as the engine wants it: a listed entry's label, or
        whatever micron size was typed into the box."""
        return self.grit.get().split("  -  ")[0].strip()

    def build_setup(self) -> Setup:
        """Read every control into a :class:`Setup`."""
        abrasive = self._lookup(ABRASIVES, self.abrasive.get())
        soap_fraction = max(self.soap_drops.get() * DROP_ML / 100.0, 1e-6)
        return Setup(
            abrasive=abrasive.key,
            # Passed through as typed: an unknown grade is reported on the
            # status line rather than silently swapped for another one.
            grit=self._grade_label() or next(iter(abrasive.grades)),
            stone=self._lookup(STONES, self.stone.get()).key,
            lap=self._lookup(LAPS, self.lap.get()).key,
            solids_mass_fraction=self.solids.get() / 100.0,
            glycerine_mass_fraction=self.glycerine.get() / 100.0,
            soap_volume_fraction=soap_fraction,
            normal_load_N=self.load.get(),
            contact_area_mm2=self.area.get(),
            sliding_speed_m_s=self.speed.get(),
            contact_length_mm=self.face_length.get(),
            temperature_C=self.temperature.get(),
            relative_humidity=self.humidity.get() / 100.0,
            air_speed_m_s=self.air_speed.get(),
            film_volume_ml=self.film_volume.get(),
        )

    # ------------------------------------------------- the live simulation
    def _queue_simulation(self) -> None:
        """Re-run shortly, collapsing a burst of slider events into one run."""
        if self._loading:
            return
        if self._pending_update is not None:
            self.after_cancel(self._pending_update)
        self._pending_update = self.after(120, self._run_simulation)

    def _run_simulation(self) -> None:
        self._pending_update = None
        try:
            result = simulate(self.build_setup())
        except Exception as error:                      # noqa: BLE001
            # KeyError quotes its message; show a bad grade's advice plainly.
            detail = error.args[0] if isinstance(error, KeyError) and error.args else error
            self.status.set(f"Could not simulate: {detail}")
            return

        self._show_glance(result)
        self.report.delete("1.0", "end")
        self.report.insert("1.0", result.report())
        self.window_map.set_marker(self.solids.get(), self.glycerine.get())

        if result.warnings:
            self.status.set(f"{len(result.warnings)} warning(s): {result.warnings[0]}")
        else:
            self.status.set("This mix sits inside every target band.")

    def _show_glance(self, result: Result) -> None:
        slurry, contact = result.slurry, result.contact
        removal, finish, drying = result.removal, result.finish, result.drying

        def duration(seconds: float) -> str:
            if seconds == float("inf"):
                return "indefinitely"
            return f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds:.0f} s"

        sections = {
            "Cutting": [
                ("Stone removal", f"{result.removal_rate_um_min:.1f} µm/min"),
                ("Plate wear", f"{result.lap_wear_um_min:.1f} µm/min "
                               f"({removal.plate_to_stone_wear_ratio:.2f}x the stone)"),
                ("Drag you feel", f"{removal.drag_force:.1f} N "
                                  f"({removal.drag_ratio:.2f}x the load)"),
                ("Contact warms by", f"{removal.temperature_rise:.0f} K"),
            ],
            "Finish": [
                ("Roughness Ra", f"{finish.Ra_um:.2f} µm"),
                ("Peak to valley Rz", f"{finish.Rz_um:.2f} µm"),
                ("Damage below the surface",
                 f"{finish.subsurface_damage_um:.1f} µm "
                 "- the next grade must remove this much"),
                ("Worst-case scratch", f"{finish.deepest_scratch_um:.2f} µm"),
                ("Cutting regime", finish.regime),
            ],
            "The slurry": [
                ("Consistency", slurry.consistency_label),
                ("Viscosity while cutting",
                 f"{slurry.plastic_viscosity * 1e3:.0f} mPa·s"),
                ("Yield stress", f"{slurry.yield_stress:.1f} Pa"),
                ("Blob height on the plate",
                 f"{slurry.puddle_height * 1e3:.2f} mm  - check this by eye"),
                ("Grit stays suspended", duration(slurry.settling_time())),
                ("Solids", f"{slurry.solids_mass_fraction * 100:.0f} wt% = "
                           f"{slurry.solids_volume_fraction * 100:.1f} vol% "
                           f"(jams at {slurry.max_packing_fraction * 100:.0f} vol%)"),
            ],
            "In the gap": [
                ("Regime", contact.regime),
                ("Plate gap", f"{contact.gap * 1e6:.1f} µm = "
                              f"{contact.gap / (result.setup.resolve()[1].d50_um * 1e-6):.1f}x d50"),
                ("Load carried by grains",
                 f"{contact.load_fraction_on_grains * 100:.0f} %"),
                ("Grains actually cutting",
                 f"{contact.active_grain_density / 1e6:.1f} per mm² "
                 f"({contact.active_fraction * 100:.3f} % of those present)"),
                ("Grains being crushed", f"{contact.fractured_fraction * 100:.1f} %"),
            ],
            "Working life": [
                ("Film on the plate",
                 f"{drying.film_thickness_um:.0f} µm thick, losing "
                 f"{drying.evaporation_rate_g_per_min:.2f} g water/min"),
                ("Stays usable for", duration(drying.workable_time_s)),
                ("Dries down to",
                 f"{drying.equilibrium_solids_fraction * 100:.0f} vol% solids "
                 f"(unworkable above {drying.jamming_solids_fraction * 100:.0f} vol%)"),
                ("Can it dry to a crust?",
                 "Yes - it will" if drying.seizes else "No, never"),
            ],
        }

        self.glance.delete(*self.glance.get_children())
        for title, rows in sections.items():
            parent = self.glance.insert("", "end", text=title, open=True)
            for label, value in rows:
                self.glance.insert(parent, "end", text="   " + label, values=(value,))

    # ---------------------------------------------------- background work
    def _run_in_background(self, message: str, work, on_done) -> None:
        """Run ``work()`` off the UI thread and hand the result to ``on_done``.

        tkinter redraws nothing while a callback runs, so doing a multi-second
        search inline would freeze the window -- including the status line meant
        to say it is busy.
        """
        if self._busy:
            return
        self._busy = True
        for button in self._buttons:
            button.state(["disabled"])
        self.status.set(message)
        self.master.configure(cursor="watch")

        def worker():
            try:
                self._results.put((on_done, work(), None))
            except Exception as error:                  # noqa: BLE001
                self._results.put((on_done, None, error))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_results(self) -> None:
        """Drain finished background work on the UI thread."""
        try:
            while True:
                on_done, value, error = self._results.get_nowait()
                self._busy = False
                for button in self._buttons:
                    button.state(["!disabled"])
                self.master.configure(cursor="")
                if error is not None:
                    self.status.set(f"Failed: {error}")
                    messagebox.showwarning("The model reported a problem", str(error))
                else:
                    on_done(value)
        except queue.Empty:
            pass
        self.after(80, self._poll_results)

    # -------------------------------------------------------- the searches
    def _find_recipe(self) -> None:
        from lapsim.recipes import diagnose_infeasible, explain_infeasible, find_recipe

        setup = self.build_setup()

        def work():
            best = find_recipe(setup, n_solids=18, n_glycerine=18)
            if best is not None:
                return best, best.tolerances(), None
            remedies = diagnose_infeasible(setup, n_solids=12, n_glycerine=12)
            return None, None, explain_infeasible(setup, remedies)

        self._run_in_background("Searching every mix for this grade…",
                                work, self._recipe_found)

    def _recipe_found(self, payload) -> None:
        from lapsim.recipes import constraint_margins

        best, spans, explanation = payload
        self.recipe_text.delete("1.0", "end")
        self.margins.delete(*self.margins.get_children())
        self.tabs.select(2)

        if best is None:
            self.recipe_text.insert("1.0", explanation)
            self.status.set("No mix works here - see the suggestions.")
            return

        lines = [best.recipe.describe(), ""]
        if spans:
            solids, glycerine = spans["solids"], spans["glycerine"]
            lines.append(
                f"Still works between {solids[0] * 100:.0f} and "
                f"{solids[1] * 100:.0f} wt% grit, and between "
                f"{glycerine[0] * 100:.0f} and {glycerine[1] * 100:.0f} % "
                "glycerine (each varied on its own).")
        lines.append("")
        lines.append(f"Tightest limit: {best.binding_constraint} "
                     f"(margin {best.robustness:+.2f})")
        self.recipe_text.insert("1.0", "\n".join(lines))

        for name, margin in constraint_margins(best.result).items():
            if margin < 0:
                reading = "breaks this limit"
            elif margin < 0.15:
                reading = "only just inside - fragile"
            elif margin < 0.5:
                reading = "comfortable"
            else:
                reading = "plenty of room"
            self.margins.insert("", "end", text=name,
                                values=(f"{margin:+.2f}", reading))

        # Load the winning mix into the controls so the readouts follow it.
        self._loading = True
        self.solids.set(round(best.setup.solids_mass_fraction * 100))
        self.glycerine.set(round(best.setup.glycerine_mass_fraction * 100))
        self.soap_drops.set(max(round(best.recipe.soap_drops), 1))
        self._loading = False
        self._run_simulation()
        self.status.set("Best recipe loaded into the controls.")

    def _map_window(self) -> None:
        from lapsim.recipes import scan_window

        setup = self.build_setup()
        self._run_in_background(
            "Scanning the recipe window…",
            lambda: scan_window(setup, n_solids=30, n_glycerine=30),
            self._window_scanned)

    def _window_scanned(self, scan) -> None:
        self.window_map.set_scan(scan)
        self.window_map.set_marker(self.solids.get(), self.glycerine.get())
        self.tabs.select(3)
        self.status.set("Hover to read any mix; click to load it.")

    def _picked_from_map(self, solids_pct: float, glycerine_pct: float) -> None:
        self.solids.set(round(solids_pct))
        self.glycerine.set(round(glycerine_pct))
        self._queue_simulation()

    def _plan_sequence(self) -> None:
        from lapsim.recipes import plan_sequence

        setup = self.build_setup()
        self._run_in_background(
            "Planning the sequence - this searches every grade…",
            lambda: plan_sequence(setup, n_solids=14, n_glycerine=14),
            self._plan_ready)

    def _plan_ready(self, plan) -> None:
        self.plan.delete(*self.plan.get_children())
        for stage in plan.stages:
            recipe = stage.candidate.recipe
            summary = (f"{recipe.grit_grams:.0f} g grit + "
                       f"{recipe.water_ml:.0f} ml water + "
                       f"{recipe.glycerine_ml:.0f} ml glycerine + "
                       f"{max(round(recipe.soap_drops), 1)} drops")
            name = stage.abrasive_name + ("" if stage.in_stock else "  (to buy)")
            self.plan.insert("", "end", text=name, values=(
                stage.grit, f"{stage.d50_um:g}", f"{stage.Ra_um:.3f}",
                f"{stage.subsurface_damage_um:.1f}",
                f"{stage.candidate.removal_rate_um_min:.1f}",
                f"{stage.damage_to_remove_um:.0f}",
                f"{stage.minutes_required:.1f}", summary))

        notes = [f"Total {plan.total_minutes:.0f} minutes of continuous grinding at "
                 "the stated pressure and speed.",
                 "Real bench time is longer: strokes lift, pressure varies, and the "
                 "plate needs re-charging."]
        for skip in plan.skipped:
            notes += ["", f"SKIPPED  {skip.abrasive_name} {skip.grit}",
                      skip.explanation]
        self.plan_notes.delete("1.0", "end")
        self.plan_notes.insert("1.0", "\n".join(notes))
        self.tabs.select(4)
        self.status.set("Sequence planned.")

    # ------------------------------------------------------------ figures
    def _render_figures(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from lapsim.plots import plot_all

        setup = self.build_setup()
        self._run_in_background(
            "Rendering figures - this takes a minute…",
            lambda: plot_all(setup), self._figures_rendered)

    def _figures_rendered(self, paths) -> None:
        self.figure_choice.configure(values=[str(path) for path in paths])
        if paths:
            self.figure_choice.set(str(paths[0]))
            self._show_figure()
        self.tabs.select(5)
        self.status.set(f"{len(paths)} figures rendered.")

    def _list_existing_figures(self) -> None:
        """Offer whatever has already been rendered, so the tab is useful at once."""
        folder = REPOSITORY_ROOT / "figures"
        existing = sorted(str(path) for path in folder.glob("*.png"))
        self.figure_choice.configure(values=existing)
        if existing:
            self.figure_choice.set(existing[0])
            self.after(200, self._show_figure)

    def _show_figure(self) -> None:
        path = self.figure_choice.get()
        if not path or not Path(path).exists():
            return
        image = tk.PhotoImage(file=path)
        # Tk cannot scale smoothly, so shrink by an integer factor to fit.
        width = max(self.figure_canvas.winfo_width(), 100)
        factor = max(1, -(-image.width() // width))
        if factor > 1:
            image = image.subsample(factor)
        self._figure_image = image          # keep a reference or Tk drops it
        self.figure_canvas.delete("all")
        self.figure_canvas.create_image(4, 4, anchor="nw", image=image)

    # -------------------------------------------------------------- menus
    def _save_report(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", initialfile="slurry-report.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            Path(path).write_text(self.report.get("1.0", "end"), encoding="utf-8")
            self.status.set(f"Saved {path}")

    def _reset_defaults(self) -> None:
        self._loading = True
        defaults = Setup()
        self.solids.set(defaults.solids_mass_fraction * 100)
        self.glycerine.set(defaults.glycerine_mass_fraction * 100)
        self.soap_drops.set(2)
        self.load.set(defaults.normal_load_N)
        self.area.set(defaults.contact_area_mm2)
        self.speed.set(defaults.sliding_speed_m_s)
        self.face_length.set(defaults.contact_length_mm)
        self.temperature.set(defaults.temperature_C)
        self.humidity.set(defaults.relative_humidity * 100)
        self.air_speed.set(defaults.air_speed_m_s)
        self.film_volume.set(defaults.film_volume_ml)
        self._loading = False
        self._run_simulation()

    def _about(self) -> None:
        messagebox.showinfo(
            "About",
            "Lapping slurry simulator\n\n"
            "A physics model of loose-abrasive lapping between two flat plates.\n\n"
            "Relative comparisons follow from the physics. Absolute numbers rest "
            "on fitted constants in lapsim/calibration.py - retune them against "
            "your own bench measurements.")


def launch() -> int:
    """Open the window."""
    root = tk.Tk()
    SimulatorWindow(root)
    root.mainloop()
    return 0
