# Grinding slurry simulator

A physics model of loose-abrasive lapping between two flat plates: silicon
carbide grains suspended in water, glycerine and dish soap, ground between a
stone and a glass plate. It exists to answer the question that is genuinely hard
to settle by trial and error — **how much grit, and how much glycerine?** — and
to say why.

You do not need the physics to use it. The short version of what it found:

> **Only the water evaporates.** Glycerine stays behind and is hygroscopic, so a
> slurry stops losing water once its liquid matches the humidity of your room. That
> fixes the solids fraction it ends up at, regardless of how long you leave it. If
> that endpoint is below the jamming point, the slurry *cannot* dry to a crust — it
> thickens, stops, and a spray of water brings it back. Start with less grit and
> the endpoint stays safe. That is exactly the effect you noticed at the bench, and
> it is the single most useful design rule here.

---

## Getting started

```bash
pip install -r requirements.txt

python run_gui.py             # the desktop window
python run_demo.py            # everything: recipes, a plan, and all six figures
```

### The desktop window

`python run_gui.py` opens a normal desktop form: settings down the left, results
in tabs on the right, a status line along the bottom.

It needs tkinter, which Debian and Ubuntu ship separately:

```bash
sudo apt install -y python3-tk        # Debian / Ubuntu, about 2.5 MB
sudo dnf install -y python3-tkinter   # Fedora
sudo pacman -S tk                     # Arch
```

Nothing else in the project needs it. Under WSL the window appears on the
Windows desktop through WSLg, with no extra setup.

What the tabs do:

- **At a glance** — the numbers that matter, grouped. Everything on the left
  re-runs as you move it, so the readouts follow the sliders.
- **Full report** — the same text `python -m lapsim report` prints.
- **Best recipe** — searches every mix for the chosen grade, then loads the
  winner into the controls. The table underneath shows how much room each limit
  has left, so you can see which one you are about to break.
- **Recipe window** — the feasibility map, drawn on a canvas. Hover to read any
  mix, **click to load it**.
- **Grit sequence** — the whole progression with a recipe and a time per stage,
  and an explanation for any grade it had to skip.
- **Figures** — renders the six charts, or shows ones already on disk.

The searches take seconds rather than milliseconds, so they run on a worker
thread and the window stays responsive while they work.

### On the command line

Or drive it piecewise:

```bash
python -m lapsim list                        # what materials are defined
python -m lapsim recipe --grit 600           # best recipe for one grade
python -m lapsim report --grit 600 --solids 0.45 --glycerine 0.6
python -m lapsim plan --humidity 55          # the whole grit sequence
python -m lapsim figures --theme dark
python -m lapsim calibrate                   # check the fitted constants
python tests/test_lapsim.py                  # 27 physics checks
python tests/test_gui_contract.py            # window/engine contract
```

From Python:

```python
from lapsim import Setup, simulate, find_recipe

print(simulate(grit="600", solids_mass_fraction=0.45).report())

best = find_recipe(Setup(grit="600", relative_humidity=0.55))
print(best.recipe.describe())
print(best.describe_tolerances())
```

---

## What it actually models

Six layers, each feeding the next. Every one is a separate module you can read
or replace.

**1. The liquid** (`vehicle.py`). Glycerol–water viscosity from Cheng's (2008)
correlation, accurate to about 1 % — and steeply non-linear, so going from 50 %
to 80 % glycerine multiplies viscosity roughly fifteenfold. Water activity is
fitted to measured values for glycerol–water (within 0.02 up to 92 % glycerol);
glycerol holds water more tightly than Raoult's law says, and this is what sets
the drying endpoint. At 45 % humidity the liquid ends at about 82 % glycerine. Surface tension
from the Szyszkowski equation, which falls steeply and then stops dead at the
critical micelle concentration — the reason one drop of detergent per 100 ml
does the whole job and the tenth drop only foams.

**2. The suspension** (`slurry.py`). Krieger–Dougherty for viscosity, which
diverges as the grains jam. A yield stress that scales as surface tension over
grain size, so **fine grit thickens a slurry far more than coarse grit at the
same loading** — usually the thing that surprises people moving down the grit
sequence. Settling from hindered Stokes, plus the yield-number criterion that
says when grains never settle at all.

The gel point depends on grain size, and that dependence matters: interparticle
attraction grows with grain diameter while grain weight grows with its cube, so
fine powders cohere into a network at modest loading while coarse ones stay
pourable right up to jamming. Practically, **your 320 and 400 grit will never
develop a yield stress at a usable loading, so their consistency has to come
from glycerine.** Fine grades, and especially the micron alumina you are
planning to buy, gel on their own — there you control consistency with the
amount of grit instead.

**3. The gap** (`contact.py`). The mechanical heart. The gap between the plates
is found by force balance: bridging grains plus hydrodynamic film pressure
together carry the applied load. Only grains coarser than the gap touch both
surfaces — **typically under one grain in a thousand**, a few per square
millimetre, carrying everything. Each indents *both* surfaces, split by
hardness, so on a glass lap against agate about 58 % of every grain's
penetration goes into the plate rather than the stone. Grains that are
overloaded fracture rather than digging deeper, which is why lapping gives a
repeatable finish over a wide range of hand pressure.

Two results fall out of this that are worth knowing:

- **The gap settles just under the top cut** — about 0.92–0.95 of the largest
  grain size the grading allows, which is 1.6 to 2.0 × d50 across F320–F1000.
  So the coarse tail of the grading, not its median, sets the depth of cut,
  and the finish and damage predictions move a lot with the assumed top cut.
- **Hydrodynamic lift scales as 1/gap².** Dropping from F320 to F1000 shrinks
  the gap about fivefold and so multiplies the lift roughly 25 times. A vehicle
  that behaves perfectly at 320 will make 1000 grit skate. This is a real
  constraint, and the simulator will tell you when you have hit it.

**4. Cutting and finish** (`removal.py`). Each bridging grain is a tiny indenter
dragged across the stone. Below a critical load it plows a groove and most of
the material is pushed aside rather than removed. Above it the stone cracks:
lateral cracks run sideways from the plastic zone and spall out a chip about ten
times the groove. The critical load goes as toughness to the *fourth* power, so
agate cracks above about 11 mN while nephrite jade needs a hundred times more —
which is why jade will not cut and has to be ground down the slow way.

This layer also reports **subsurface damage**: cracked stone left *below* the
surface you just made. It is invisible, and it is the thing that costs the most
time, because the surface looks finished until the next stage polishes down into
it and scratches appear from nowhere. `plan_sequence` uses it to work out how
long each stage has to run.

**5. Drying** (`drying.py`). The water-activity endpoint described at the top,
plus the time course: water loss driven by the difference between the slurry's
water activity and the room's humidity, over the area the film is spread across.
That area dominates — the same 8 ml lasts hours in a puddle and minutes smeared
over a 200 mm plate.

**6. Finding recipes** (`recipes.py`). There is no single best slurry, there is a
*window* of workable ones, and a recipe in the middle of the window tolerates a
sloppy measurement and a humid day where one on the edge does not. `find_recipe`
maximises the distance to the nearest constraint by default. `scan_window` maps
the whole region. When nothing works, `diagnose_infeasible` searches for the
change of technique that would fix it rather than just reporting failure.

---

## The figures

`python run_demo.py` writes six charts to `figures/`. Add `--theme dark` for a
dark-surface version, `--animate` for a view of grains moving through the gap.

| | |
|---|---|
| `fig1_vehicle` | what glycerine and soap do to the liquid |
| `fig2_recipe_windows` | the workable region for each grade, with the best mix marked |
| `fig3_drying` | why a lighter loading stays wet longer |
| `fig4_contact` | the gap drawn to scale, and which grains carry load |
| `fig5_grit_comparison` | cut rate, finish, damage and gap across grades |
| `fig6_sequence` | the progression, and how long each stage takes |

---

## Grit sizes, and adding the abrasives you are planning to buy

Grit numbers only mean something next to the standard they come from, and the
standards disagree badly at the fine end: "1000 grit" is 4.5 µm under FEPA F,
11.5–15.5 µm under JIS R6001 and about 18 µm on FEPA P sandpaper. So the
simulator names grades two ways.

**Numbered grades, 240 to 1000, are FEPA F** (FEPA 42-2:2006, the grading for
bonded and loose fused alumina and silicon carbide). Grit sold by number with no
standard stated is taken to be FEPA F. Each grade uses the standard's median
(ds50) and, as its top cut, the ds3 maximum, the size only 3 % of a compliant
powder may exceed:

| grit | 240 | 280 | 320 | 360 | 400 | 500 | 600 | 800 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| d50, µm | 44.5 | 36.5 | 29.2 | 22.8 | 17.3 | 12.8 | 9.3 | 6.5 | 4.5 |
| top cut, µm | 70 | 59 | 49 | 40 | 32 | 25 | 19 | 14 | 10 |

Black SiC also has sieved 120 and 220 grit. `"F600"` works as well as `"600"`.

**Anything finer than F1000 is named by its micron size**, taken from the
supplier's specification. Grit numbers above 1000 are refused rather than
guessed. Any size works straight away, in the window's grade box, on the command
line or from Python:

```bash
python -m lapsim report --abrasive sic_green --grit 2.5um
```

```python
simulate(abrasive="alumina_white", grit="3um")
```

To record a powder properly, with its top size and whether it is on the shelf,
register it. Give the top size if the spec lists one (often as d3, ds3 or
"max"); without it the top is taken as 2.2 × d50, a powder only just inside what
FEPA allows at F1000–F1200. The top size matters, because the coarsest grains
are the ones that carry the load:

```python
from lapsim import add_micron_grade, set_stock

add_micron_grade("sic_green", 2.5, d_max_um=6.0, in_stock=True)
add_micron_grade("alumina_white", 1.0)
set_stock("alumina_white", ("600", "1um"))
```

Green silicon carbide and white fused alumina are **already defined** in
`materials.py`, with the FEPA F grades and 3, 2 and 1.2 µm examples (the medians
FEPA calls F1200, F1500 and F2000). There is a marked place at the end of
`materials.py` for your own `add_micron_grade` lines, so they load every time.

A new abrasive is one dataclass. Nothing outside `materials.py` needs to change:

```python
from lapsim import Abrasive, GritGrade, register_abrasive

register_abrasive(Abrasive(
    key="boron_carbide",
    name="Boron carbide",
    density=2520.0,
    hardness=30e9,              # Knoop, Pa
    youngs_modulus=460e9,
    grain_strength_ref=500e6,   # single-grain crushing strength at 50 um
    tip_semi_angle_deg=62.0,    # sharper = cuts deeper per unit load
    friability=0.4,
    grades={"800": GritGrade("800", d50_um=6.5, d_max_um=14.0, sigma_g=1.28)},
))
```

Stones and lap plates work the same way, via `register_stone` and
`register_lap`. Six stones and four plate materials are defined already.

---

## What to trust, and what to check

The numbers here split into two kinds, and they deserve different amounts of
faith.

**Trust the comparisons of consistency and drying.** Does more glycerine help?
Will this mix crust? Which grade will hydroplane first? These follow from
published material properties and mechanics that are not in dispute, and they
are what the simulator is for.

**Do not trust how finish changes with grit loading.** The model has fewer
grains each carrying more load, so a lean slurry digs deeper and finishes
rougher. At the bench a lean charge of friable SiC breaks down as you work it
and finishes *brighter*, and the model does not follow that breakdown. For that
reason the recipe search no longer holds recipes to a finish ceiling unless you
set `Targets(enforce_finish=True)`; Ra is still reported.

**Check the absolute numbers against your own bench.** "35 µm/min" and "Ra
0.9 µm" rest on about a dozen fitted coefficients, all collected in
`calibration.py` with the anchor values they were tuned to. `python -m lapsim
calibrate` shows how the current constants score. If you ever time a removal
rate or judge a finish, retune them there and everything downstream sharpens up.

Other limits worth stating plainly:

- Times assume **continuous contact at the stated pressure and speed**. Real
  bench time is longer — strokes lift, pressure varies, the plate needs
  re-charging.
- The contact is treated as flat and uniformly loaded. Real cabochons rock, and
  the edges see more pressure than the middle.
- Grain breakdown is detected but not tracked through time; the model reports
  the fraction of grains fracturing rather than evolving the grading as you work.
  This is the missing physics behind the finish-versus-loading problem above.
- Subsurface damage is a rule of thumb (2.5 × Rz), not a crack calculation, and
  the stage times in the grit sequence rest on it.
- Diamond pastes are not modelled; they come ready-made, so there is no recipe
  to search for.
- Curved surfaces, wheels and vibratory tumbling are all out of scope. This is
  flat-on-flat only, which is what you asked for.
- The `Targets` you are judged against are a *choice*, not physics. The most
  consequential one is `min_low_shear_viscosity_Pas`, which decides how thick
  "not too runny" means. `run_demo.py` prints a sensitivity table for it.

---

## File map

```
lapsim/
  calibration.py   fitted coefficients, in one place, with their anchors
  materials.py     abrasives, gradings, stones, lap plates  <- the registry
  vehicle.py       water / glycerine / soap
  slurry.py        solids loading, rheology, settling, consistency
  contact.py       the plate gap: load sharing, penetration, grain fracture
  removal.py       cutting rate, roughness, subsurface damage, drag, heat
  drying.py        evaporation and the working life of the film
  simulate.py      Setup -> Result, and the warnings
  recipes.py       the recipe window, the search, the grit sequence
  plots.py         six figures plus an animation
  style.py         one palette, light and dark, shared by figures and window
  gui.py           the desktop window
  cli.py           command line
  api.py           JSON bridge, for driving the model from another language
tests/             physics checks, plus the engine/window contract
run_gui.py         open the desktop window
run_demo.py        everything, end to end
LapSim.UI/         optional C# WinForms front end (see below)
```

### The optional Windows Forms front end

`LapSim.UI/` holds a C# WinForms version of the same window, which talks to the
engine over the JSON pipe in `lapsim/api.py` rather than importing it. It builds
with `dotnet build LapSim.UI/LapSim.UI.csproj` and launches with `run_ui.cmd`,
and it needs an interpreter with numpy and scipy (matplotlib is optional, and
only enables rendering new figures).

It is a genuine alternative rather than the main path: the Python window above
runs in the same environment as the rest of the project and needs no second
build system. Reach for this one only if you specifically want a .NET binary.

---

## Sources

- Cheng, N.-S. (2008). Formula for the viscosity of a glycerol–water mixture.
  *Ind. Eng. Chem. Res.* **47**, 3285–3288.
- Krieger, I. M. & Dougherty, T. J. (1959). A mechanism for non-Newtonian flow
  in suspensions of rigid spheres. *Trans. Soc. Rheol.* **3**, 137–152.
- Marshall, D. B., Lawn, B. R. & Evans, A. G. (1982). Elastic/plastic indentation
  damage in ceramics: the lateral crack system. *J. Am. Ceram. Soc.* **65**,
  561–566.
- Lawn, B. R. & Evans, A. G. (1977). A model for crack initiation in
  elastic/plastic indentation fields. *J. Mater. Sci.* **12**, 2195–2199.
- Richardson, J. F. & Zaki, W. N. (1954). Sedimentation and fluidisation.
  *Trans. Inst. Chem. Eng.* **32**, 35–53.
- Reynolds, O. (1886), for the inclined-slider film pressure; Stefan (1874), for
  the squeeze film.
- FEPA 42-2:2006, grain size distribution of microgrits F230–F2000, Table 1.
- *Molecular Basis of Water Activity in Glycerol–Water Mixtures*, PubMed Central
  PMC6839025, Table 1 (measured water activity at 25 °C).
