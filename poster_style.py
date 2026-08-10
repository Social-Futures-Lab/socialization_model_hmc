"""
Poster styling for matplotlib/seaborn figures.

Usage:
    from poster_style import use_poster_style, DATA_COLORS, save
    use_poster_style()
    ...
    save(fig, "figure.pdf")
"""

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ---------------------------------------------------------------------------
# Custom font
# ---------------------------------------------------------------------------
# Path is resolved relative to THIS file, not the working directory, so the
# style module keeps working no matter where you run the script from.
FONT_DIR = Path(__file__).resolve().parent / "fonts"


def register_fonts(font_dir=FONT_DIR):
    """Register every static .ttf/.otf in `font_dir`; return the family names."""
    families = set()
    if not Path(font_dir).is_dir():
        warnings.warn(f"No font directory at {font_dir}; using system fonts.")
        return []
    for path in sorted(Path(font_dir).iterdir()):
        if path.suffix.lower() not in {".ttf", ".otf"}:
            continue
        if "[" in path.name:          # variable font (e.g. PublicSans[wght].ttf)
            continue                  # matplotlib only reads its default instance
        fm.fontManager.addfont(str(path))
        families.add(fm.FontProperties(fname=str(path)).get_name())
    return sorted(families)


def check_font(family):
    """Print which file matplotlib actually resolves for each weight.

    If any line says DejaVu (or anything that isn't your font), matplotlib
    silently fell back and the font is NOT being used.
    """
    print(f"Requested family: {family}")
    registered = [(e.weight, Path(e.fname).name)
                  for e in fm.fontManager.ttflist if e.name == family]
    print(f"  registered faces: {sorted(registered) or 'NONE -- not registered'}")
    for weight in ("normal", "medium", "semibold", "bold"):
        resolved = fm.findfont(fm.FontProperties(family=family, weight=weight))
        flag = "" if family.replace(" ", "").lower() in Path(resolved).name.lower() \
            else "   <-- FALLBACK"
        print(f"  {weight:<9} -> {Path(resolved).name}{flag}")
    print(f"  rcParams font.family    = {mpl.rcParams['font.family']}")
    print(f"  rcParams font.sans-serif[:3] = {mpl.rcParams['font.sans-serif'][:3]}")


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Your poster colors. Great for panel backgrounds, title bars, section rules --
# anything that covers a large area. Too light for lines/markers on white.
ACCENT_COLORS = {
    "rose":   "#e19d9e",
    "violet": "#dec7e3",
    "amber":  "#ffcc8d",
}

# Same hues, pushed darker/more saturated so they hold up as thin lines and
# small markers. These are what the data should be drawn in.
DATA_COLORS = {
    "rose":   "#9E3B47",
    "violet": "#6A4A86",
    "amber":  "#C9820F",
    "slate":  "#3F5661",   # spare 4th category / annotations
}

# Ordered list for seaborn `palette=`. Lightness is deliberately staggered
# (violet darkest -> amber lightest) so the series stay distinguishable in
# grayscale and under red-green color blindness.
DATA_PALETTE = [DATA_COLORS["violet"], DATA_COLORS["rose"], DATA_COLORS["amber"],
                DATA_COLORS["slate"]]

INK = "#22242A"      # axis lines, ticks, text -- softer than pure black
MUTED = "#6B7280"    # secondary text, grid


def use_poster_style(base_size=15, font="Public Sans"):
    """Apply poster-scale typography and clean axes.

    base_size: point size of tick labels *as printed*. Everything else scales
    from it. 15-16 works when the figure is placed at roughly its saved size
    on an A0 poster; bump to 18-20 if the figure will sit small on the board.

    font: family name to prefer. Pass None to use the system sans stack.
    """
    families = register_fonts()
    stack = ["Helvetica Neue", "Helvetica", "Arial",
             "Source Sans Pro", "Inter", "DejaVu Sans"]
    if font:
        if font not in families and font not in {f.name for f in fm.fontManager.ttflist}:
            warnings.warn(
                f"'{font}' not found (registered from {FONT_DIR}: {families}). "
                "Falling back to the system sans stack; run check_font() for detail.")
        stack.insert(0, font)

    mpl.rcParams.update({
        # --- fonts -------------------------------------------------------
        # Note: set font.family to the generic "sans-serif" and put the real
        # family at the head of font.sans-serif. Matplotlib then falls through
        # the list if a glyph or the font itself is missing, instead of
        # erroring out or silently landing on DejaVu.
        "font.family": "sans-serif",
        "font.sans-serif": stack,
        "font.size": base_size,
        "axes.labelsize": base_size + 3,
        "axes.titlesize": base_size + 4,
        "xtick.labelsize": base_size,
        "ytick.labelsize": base_size,
        "legend.fontsize": base_size,
        "axes.labelweight": "medium",
        "axes.titleweight": "semibold",

        # --- color / ink -------------------------------------------------
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.prop_cycle": mpl.cycler(color=DATA_PALETTE),

        # --- line and marker weight (thin strokes vanish at 2 m) ---------
        "lines.linewidth": 3.0,
        "lines.markersize": 10,
        "lines.markeredgewidth": 0,
        "axes.linewidth": 1.6,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 6,
        "ytick.major.size": 6,

        # --- chrome ------------------------------------------------------
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#D9DCE1",
        "grid.linewidth": 1.0,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        # --- export ------------------------------------------------------
        "figure.dpi": 110,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",     # stops the clipped y-labels you had
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,          # embed real text, not outlines
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save(fig, path_no_ext, also_png=True):
    """Save a vector PDF (for the poster) and a PNG (for quick checks)."""
    fig.savefig(f"{path_no_ext}.pdf")
    if also_png:
        fig.savefig(f"{path_no_ext}.png")
    plt.close(fig)
