from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def plot_synaptic_event_raster(
    ev_df: pl.DataFrame,
    xname: str = "time",
    yname: str = "pos",
    color: str = "#2ec700",
    alpha: float = 0.9,
    s: float = 8,
    width: float = 8,
    inches_per_pos: float = 0.05,
    margin_inches: float = 0.05,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a raster of synaptic events with height proportional to unique positions.

    The figure height scales with the y-range so that the physical spacing
    between adjacent positions and the dot size are identical across plots.
    A dendrite with 60 positions produces a 3-inch-tall plot by default;
    one with 30 positions produces 1.5 inches — visually, the shorter plot
    looks like the taller one cropped in half.

    Parameters
    ----------
    ev_df : pl.DataFrame
        Event data with columns for x (time) and y (position).
    xname : str
        Column name for x-axis.
    yname : str
        Column name for y-axis.
    color : str
        Dot color.
    alpha : float
        Dot transparency.
    s : float
        Dot size in points² — kept constant across all plots.
    width : float
        Figure width in inches.
    inches_per_pos : float
        Vertical inches per position row. Default 0.05 gives 3 inches for
        60 rows.
    margin_inches : float
        Top and bottom margin in inches (keeps edge dots from clipping).
    ax : matplotlib Axes, optional
        If provided, plot onto this axes instead of creating a new figure.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
    """
    y_vals = ev_df[yname]
    y_min: float = float(y_vals.min())  # type: ignore[arg-type]
    y_max: float = float(y_vals.max())  # type: ignore[arg-type]
    y_span = y_max - y_min + 1  # integer rows including both endpoints

    data_height = y_span * inches_per_pos
    fig_height = data_height + 2 * margin_inches

    if ax is None:
        f, ax = plt.subplots(figsize=(width, fig_height))
    else:
        f = ax.get_figure()

    # Set subplot margins so the data area is exactly data_height inches
    bottom_frac = margin_inches / fig_height
    top_frac = 1 - margin_inches / fig_height
    f.subplots_adjust(left=0.005, right=0.995, bottom=bottom_frac, top=top_frac)

    ax.scatter(
        ev_df[xname].to_numpy(),
        y_vals.to_numpy(),
        s=s,
        color=color,
        alpha=alpha,
        linewidths=0,
    )

    # Half-unit padding so edge rows aren't clipped
    ax.set_ylim(y_min - 0.5, y_max + 0.5)

    # Clean/minimal: no ticks, labels, or spines
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("none")
    f.patch.set_facecolor("white")

    return f, ax


def atomic_raster(
    mua_df,
    xname="time",
    yname="pos",
    color="blue",
    alpha=0.7,
    s=40,
    figsize=(8, 4),
):
    """Plot a raster plot of MUA data

    Parameters
    ----------
    mua_df : pl.DataFrame or pd.DataFrame
        the MUA data to plot, should have columns according to xname and yname
    xname : str, optional
        Column name for the x-axis, by default 'datetime'
    yname : str, optional
        Column name for the y-axis, by default 'negchan'
    color : str, optional
        Color of the raster plot, by default 'blue'
    figsize : tuple, optional
        Size of the figure, by default (24, 8)

    Returns
    -------
    f, ax : tuple
        Matplotlib Figure and axes objects
    """

    plt.rcParams["axes.spines.bottom"] = False
    plt.rcParams["axes.spines.left"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.grid"] = False
    plt.rcParams["xtick.major.size"] = 0
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "None"

    assert xname in mua_df.columns, f"xname {xname} not in mua_df"
    assert yname in mua_df.columns, f"yname {yname} not in mua_df"

    f, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(
        mua_df, x=xname, y=yname, linewidth=0, alpha=alpha, s=s, ax=ax, color=color
    )

    ax.set_xlim()
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    return f, ax
