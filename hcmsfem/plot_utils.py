import subprocess
from enum import Enum
from os import remove
from typing import Iterable

import matplotlib as mpl
import matplotlib.colors as mpl_colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig

from hcmsfem.root import get_venv_root

FIG_FOLDER = get_venv_root() / "figures"

MPL_LINE_STYLES = [
    "-",  # solid
    "--",  # dashed
    "-.",  # dashdot
    ":",  # dotted
    "-",  # solid line style
    "--",  # dashed line style
    "-.",  # dash-dot line style
    ":",  # dotted line style
    ((0, (1, 10))),  # loosely dotted
    ((0, (1, 5))),  # dotted
    ((0, (5, 10))),  # densely dotted
    ((0, (5, 1))),  # densely dashed
    ((0, (3, 10, 1, 10))),  # loosely dashdotted
    ((0, (3, 5, 1, 5))),  # dashdotted
    ((0, (3, 1, 1, 1))),  # densely dashdotted
    ((0, (3, 5, 1, 5, 1, 5))),  # dashdotdotted
    ((0, (3, 1, 1, 1, 1, 1))),  # densely dashdotdotted
]

MPL_MARKER_STYLES = [
    ".",  # point marker
    ",",  # pixel marker
    "o",  # circle marker
    "v",  # triangle_down marker
    "^",  # triangle_up marker
    "<",  # triangle_left marker
    ">",  # triangle_right marker
    "1",  # tri_down marker
    "2",  # tri_up marker
    "3",  # tri_left marker
    "4",  # tri_right marker
    "s",  # square marker
    "p",  # pentagon marker
    "*",  # star marker
    "h",  # hexagon1 marker
    "H",  # hexagon2 marker
    "+",  # plus marker
    "x",  # x marker
    "D",  # diamond marker
    "d",  # thin_diamond marker
    "|",  # vline marker
    "_",  # hline marker
]

# default matplotlib tableau colors
MPL_COLORS = list(mpl_colors.TABLEAU_COLORS)

# custom palette
CUSTOM_COLOURS_P1 = [  # sand colour
    "#D1BEAF",
    "#E1CEC7",
    "#E9DEDA",
    "#D7CBBD",
    "#F4EFE9",
    "#ACBBC2",
    "#FBFAF6",
    "#DDE5E7",
    "#E8F0F3",
    "#F5F9F8",
]
CUSTOM_COLOURS_P2 = [  # blueish grey
    "#869099",
    "#C0C7CF",
    "#9FA7AA",
    "#697784",
    "#CED6D9",
    "#131A24",
    "#565C68",
    "#7EAFF1",
    "#5274B4",
    "#253242",
]
CUSTOM_COLORS_P3 = [  # reds
    "#945357",
    "#4C2631",
    "#792026",
    "#651319",
    "#86343A",
    "#7D0C10",
    "#7F1217",
    "#AA7678",
    "#720B0E",
    "#761115",
    "#B38B6D",
]

CUSTOM_COLORS_FULL = CUSTOM_COLOURS_P1 + CUSTOM_COLOURS_P2 + CUSTOM_COLORS_P3

CUSTOM_COLORS_SIMPLE = [
    "#945357",  # Deep Reddish Brown
    "#7A8F99",  # Blueish Muted Blue-Gray
    # "#869099",  # Muted Blue-Gray
    "#253242",  # Dark Navy
    "#B38B6D",  # Golden Brown
    "#7EAFF1",  # Soft Sky Blue
    "#B79A89",  # Dark Warm Beige
]


class CustomColors(Enum):
    RED = "#945357"
    BLUE = "#7A8F99"
    NAVY = "#253242"
    GOLD = "#B38B6D"
    SKY = "#7EAFF1"
    BEIGE = "#B79A89"
    SOFTSKY = "#9CC3F5"


# define pre and post strings
LATEX_STANDALONE_PGF_PRE = r"""\documentclass{standalone} 
\def\mathdefault#1{#1}
\everymath=\expandafter{\the\everymath\displaystyle}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{siunitx}
\usepackage{pgf}
\usepackage{lmodern}
\usepackage[no-math]{fontspec}
\setmainfont{Tex Gyre Heros}
\setmathsf{Tex Gyre Heros}
\setmathtt{Tex Gyre Heros}
\setmathrm{Tex Gyre Heros}
\makeatletter\@ifpackageloaded{underscore}{}{\usepackage[strings]{underscore}}\makeatother
\begin{document} 
"""

LATEX_STANDALONE_PGF_POST = r""" 
\end{document}
"""


class CartesianCoordinate(Enum):
    X = 0
    Y = 1
    Z = 2


def save_latex_figure(fn: str, fig: Figure | None = None) -> None:
    # ensure figure directory exists
    FIG_FOLDER.mkdir(parents=True, exist_ok=True)

    # save currently active or provided figure as pgf using matplotlib
    pgf_p = FIG_FOLDER / (fn + ".pgf")
    if fig is None:
        savefig(pgf_p, backend="pgf")
    else:
        fig.savefig(pgf_p, backend="pgf")

    # create tex file
    tex_p = pgf_p.with_suffix(".tex")
    with open(tex_p, "w") as tex_f:
        tex_f.write(LATEX_STANDALONE_PGF_PRE)
        tex_f.write(r"    \input{" + pgf_p.as_posix() + r"}")
        tex_f.write(LATEX_STANDALONE_PGF_POST)

    # run lualatex on tex file
    subprocess.run(
        ["lualatex", tex_p.name], check=True, cwd=FIG_FOLDER, capture_output=True
    )
    print(f"Saved figure to {tex_p.with_suffix('.pdf')}")

    # clean up .aux and .log files
    for file in FIG_FOLDER.glob("*.aux"):
        remove(file)
    for file in FIG_FOLDER.glob("*.log"):
        remove(file)

    # delete pgf & tex files
    remove(tex_p)
    remove(pgf_p)


# set standard matploylib style
def set_mpl_style(fontsize: int = 10):
    mpl.rcParams.update(
        {
            "pgf.texsystem": "lualatex",
            "pgf.preamble": "\n".join(
                [
                    r"\usepackage{amsmath}",
                    r"\usepackage{amssymb}",
                    r"\usepackage{siunitx}",
                    r"\usepackage[no-math]{fontspec}",
                    r"\setmainfont{Arial}",
                    r"\setmathsf{Arial}",
                    r"\setmathtt{Arial}",
                    r"\setmathrm{Arial}",
                ]
            ),
            "axes.labelweight": "bold",
            "lines.linewidth": 1.5,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "legend.frameon": True,
            "legend.framealpha": 1,
            "legend.fancybox": True,
            "legend.shadow": True,
            "legend.borderpad": 1,
            "legend.borderaxespad": 1,
            "legend.handletextpad": 1,
            "legend.handlelength": 1.5,
            "legend.labelspacing": 1,
            "legend.columnspacing": 2,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "savefig.transparent": False,
            "savefig.orientation": "landscape",
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "legend.title_fontsize": fontsize,
            "figure.titlesize": fontsize,
        }
    )


def set_mpl_cycler(lines: bool = False, colors: bool = False, markers: bool = False):
    # default style
    d_line = ["-"]
    d_color = ["black"]
    d_marker = ["None"]

    n_props = []
    custom_line = []
    custom_colors = []
    custom_markers = []
    if lines:
        custom_line = MPL_LINE_STYLES
        n_lines = len(custom_line)
        n_props.append(n_lines)
    if colors:
        custom_colors = CUSTOM_COLORS_SIMPLE
        n_colors = len(custom_colors)
        n_props.append(n_colors)
    if markers:
        custom_markers = MPL_MARKER_STYLES
        n_markers = len(custom_markers)
        n_props.append(n_markers)

    min_props = min(n_props)

    if lines:
        line = custom_line[:min_props]
    else:
        line = d_line * min_props

    if colors:
        color = custom_colors[:min_props]
    else:
        color = d_color * min_props

    if markers:
        marker = custom_markers[:min_props]
    else:
        marker = d_marker * min_props

    line_cycler = mpl.cycler("linestyle", line)  # type: ignore
    color_cycler = mpl.cycler("color", color)  # type: ignore
    marker_cycler = mpl.cycler("marker", marker)  # type: ignore

    custom_cycler = line_cycler + color_cycler + marker_cycler
    mpl.rcParams["axes.prop_cycle"] = custom_cycler


def mpl_graph_plot_style(
    ax: Axes,
    domain: tuple,
    codomain: tuple,
    xtick_locs: Iterable[float] = [],
    ytick_locs: Iterable[float] = [],
    origin: bool = False,
) -> Axes:
    # check input
    if len(domain) != 2 or len(codomain) != 2:
        raise ValueError("Domain and codomain must be tuples of length 2")

    # turn off mpl axes
    ax.set_axis_off()

    # axis ranges
    domain_range = domain[1] - domain[0]
    codomain_range = codomain[1] - codomain[0]

    # axes
    v_axis_x = domain[0]
    h_axis_y = codomain[0]
    if domain[0] < 0 < domain[1]:
        v_axis_x = 0
    if codomain[0] < 0 < codomain[1]:
        h_axis_y = 0

    # horizontal axis
    ax.plot(
        domain, [h_axis_y, h_axis_y], color="black", lw=1, linestyle="-", marker="None"
    )

    # vertical axis
    ax.plot(
        [v_axis_x, v_axis_x],
        codomain,
        color="black",
        lw=1,
        linestyle="-",
        marker="None",
    )

    # origin
    if v_axis_x == 0 and h_axis_y == 0 and origin:
        ax.text(
            -0.02 * domain_range,
            -0.02 * codomain_range,
            r"$O$",
            verticalalignment="top",
            horizontalalignment="right",
        )
    elif origin and (v_axis_x != 0 or h_axis_y != 0):
        raise ValueError(
            "Origin is requested, but not present in specified domain and codomain."
        )

    # place ticks
    for x in xtick_locs:
        ax = mpl_add_custom_tick(
            ax, (x, h_axis_y), CartesianCoordinate.Y, 0.01 * codomain_range
        )

    for y in ytick_locs:
        ax = mpl_add_custom_tick(
            ax, (v_axis_x, y), CartesianCoordinate.X, 0.01 * domain_range
        )

    return ax


def mpl_add_custom_tick(
    ax: Axes,
    loc: tuple,
    orientation: CartesianCoordinate,
    ticksize: float,
    label_text: str | None = None,
):
    x_tick = []
    y_tick = []
    label = 0
    x_label = 0
    y_label = 0
    horizontalalignment = ""
    verticalalignment = ""
    if orientation == CartesianCoordinate.X:
        x_tick = [loc[0] - 0.5 * ticksize, loc[0] + 0.5 * ticksize]
        y_tick = [loc[1], loc[1]]

        label = loc[1]
        x_label = loc[0] - 2 * ticksize
        y_label = loc[1]

        horizontalalignment = "right"
        verticalalignment = "bottom"
    elif orientation == CartesianCoordinate.Y:
        x_tick = [loc[0], loc[0]]
        y_tick = [loc[1] - 0.5 * ticksize, loc[1] + 0.5 * ticksize]

        label = loc[0]
        x_label = loc[0]
        y_label = loc[1] - 2 * ticksize

        horizontalalignment = "left"
        verticalalignment = "top"

    # plot the tick
    ax.plot(x_tick, y_tick, color="black", lw=1, linestyle="-", marker="None")

    # label the tick
    label_text = f"${label:.1f}$" if label_text is None else label_text
    ax.text(
        x_label,
        y_label,
        label_text,
        verticalalignment=verticalalignment,
        horizontalalignment=horizontalalignment,
    )

    return ax
