import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def plot_synaptic_traces(
    traces: np.ndarray,
    fs: float,
    *,
    tlim: tuple[float, float] | None = None,
    # preprocessing
    normalize: str = "dff",  # "none" | "dff" | "zscore"
    f0_percentile: float = 10.0,  # for dF/F0, per-trace F0 = percentile
    smooth_win_s: float | None = 0.0,  # moving-average window (seconds); 0/None = off
    clip_quantiles: tuple[float, float] | None = (0.5, 99.5),  # robust clipping
    max_points: int = 5000,  # decimate for speed if extremely long
    # stacking/appearance
    vertical_spacing: float | None = None,  # if None, auto from data
    trace_height_frac: float = 0.8,  # fraction of spacing a trace occupies
    per_trace_scaling: bool = True,  # scale each trace to fill trace_height
    sort_by_activity: bool = False,  # sort by robust std (descending)
    line_color: str = "#22bf1f",  # green
    colors: list | np.ndarray | None = None,  # RGBA per trace; overrides line_color
    linewidth: float | None = None,  # None = auto-scale based on figure size
    alpha: float = 1.0,
    antialiased: bool = False,  # False for crisp lines on dark backgrounds
    # optional overlay of "active" segments (thicker/darker)
    active_mask: np.ndarray | None = None,  # bool (n_synapses, n_timepoints)
    active_linewidth: float = 1.0,
    active_color: str | None = None,  # None -> same as line_color
    # figure/axes
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int = 300,
    background: str = "black",
):
    """
    Return (fig, ax).

    Notes
    -----
    - traces are stacked with equal vertical offsets; by default each trace is
      normalized (dF/F0) and scaled so its 2.5–97.5% range occupies
      `trace_height_frac` of the vertical spacing.
    - `vertical_spacing=None` chooses spacing based on robust amplitude across traces.
    - `active_mask` (optional) can highlight segments (e.g., detected events)
      with a thicker overlay line.
    - `colors` (optional) list/array of RGBA values with length matching
      traces.shape[0]. Each trace will be drawn with its corresponding color.
      If provided, overrides `line_color`.
    """
    if traces.ndim != 2:
        raise ValueError("`traces` must be 2D: (n_synapses, n_timepoints).")

    n_traces, n_t = traces.shape
    if n_traces == 0 or n_t == 0:
        raise ValueError("Empty input.")

    # Time window selection ----------------------------------------------------
    if tlim is None:
        i0, i1 = 0, n_t
    else:
        start, end = max(0.0, tlim[0]), min(n_t / fs, tlim[1])
        i0, i1 = int(np.floor(start * fs)), int(np.ceil(end * fs))
        i0, i1 = max(0, i0), min(n_t, i1)
        if i1 <= i0:
            raise ValueError("Invalid tlim; no samples selected.")

    Y = traces[:, i0:i1].astype(np.float64, copy=False)
    T = np.arange(i0, i1) / fs

    # Optional decimation for speed -------------------------------------------
    if max_points and T.size > max_points:
        step = int(np.ceil(T.size / max_points))
        T = T[::step]
        Y = Y[:, ::step]
        if active_mask is not None:
            active_mask = active_mask[:, i0:i1][:, ::step]

    # Preprocessing: normalize -------------------------------------------------
    def moving_average(x, win):
        if win <= 1:
            return x
        # NaN-safe simple moving average
        kern = np.ones(win, dtype=float)
        valid = ~np.isnan(x)
        num = np.convolve(np.nan_to_num(x), kern, mode="same")
        den = np.convolve(valid.astype(float), kern, mode="same")
        out = np.divide(num, den, where=den > 0)
        out[den == 0] = np.nan
        return out

    Yp = Y.copy()

    if normalize.lower() == "dff":
        # per-trace F0 from percentile; handle NaNs
        f0 = np.nanpercentile(Yp, f0_percentile, axis=1, keepdims=True)
        # avoid division by ~0
        f0 = np.where(np.isfinite(f0) & (np.abs(f0) > 1e-12), f0, np.nanmedian(f0))
        Yp = (Yp - f0) / f0
    elif normalize.lower() == "zscore":
        mu = np.nanmean(Yp, axis=1, keepdims=True)
        sd = np.nanstd(Yp, axis=1, keepdims=True)
        sd = np.where(sd > 0, sd, 1.0)
        Yp = (Yp - mu) / sd
    elif normalize.lower() == "none":
        pass
    else:
        raise ValueError("normalize must be one of: 'none', 'dff', 'zscore'.")

    # Smoothing (light, optional) ---------------------------------------------
    if smooth_win_s and smooth_win_s > 0:
        win = int(round(smooth_win_s * fs))
        win = max(1, win)
        if win % 2 == 0:  # odd often looks nicer; not required though
            win += 1
        for i in range(n_traces):
            Yp[i] = moving_average(Yp[i], win)

    # Robust clipping to suppress outliers (optional) -------------------------
    if clip_quantiles is not None:
        lo_q, hi_q = clip_quantiles
        lo = np.nanpercentile(Yp, lo_q, axis=1, keepdims=True)
        hi = np.nanpercentile(Yp, hi_q, axis=1, keepdims=True)
        Yp = np.clip(Yp, lo, hi)

    # Sorting by "activity" (robust std) --------------------------------------
    if sort_by_activity:
        rob_std = np.nanpercentile(Yp, 84, axis=1) - np.nanpercentile(Yp, 16, axis=1)
        order = np.argsort(rob_std)[::-1]  # most active on top
        Yp = Yp[order]
        if active_mask is not None:
            active_mask = active_mask[order]
    else:
        order = np.arange(n_traces)

    # Compute scaling & offsets -----------------------------------------------
    # How tall should a trace be relative to spacing?
    if per_trace_scaling:
        robust_amp = np.nanpercentile(Yp, 97.5, axis=1) - np.nanpercentile(
            Yp, 2.5, axis=1
        )
        robust_amp[~np.isfinite(robust_amp) | (robust_amp <= 0)] = 1.0
        # We will scale each trace so this robust range == trace_height.
        # trace_height determined below when spacing is known.
    else:
        # single global amplitude for all traces
        robust_amp_all = np.nanpercentile(Yp, 97.5) - np.nanpercentile(Yp, 2.5)
        if not np.isfinite(robust_amp_all) or robust_amp_all <= 0:
            robust_amp_all = 1.0
        robust_amp = np.full(n_traces, robust_amp_all)

    # Determine spacing automatically if not provided
    if vertical_spacing is None:
        # Choose spacing so that the *median* robust range fits nicely
        med_amp = np.nanmedian(robust_amp)
        trace_height = med_amp if med_amp > 0 else 1.0
        vertical_spacing = trace_height / trace_height_frac
    else:
        trace_height = vertical_spacing * trace_height_frac

    # Scaling factors
    scales = (trace_height / robust_amp)[:, None]  # (n_traces, 1)
    if not per_trace_scaling:
        scales = np.full_like(scales, trace_height / robust_amp[0])

    Ys = Yp * scales

    # Apply vertical offsets (top trace is index 0)
    offsets = np.arange(n_traces)[::-1] * vertical_spacing
    Yplot = Ys + offsets[:, None]

    # Prepare per-trace colors ------------------------------------------------
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape[0] != n_traces:
            raise ValueError(
                f"`colors` length ({colors.shape[0]}) must match number of traces ({n_traces})."
            )
        # Reorder colors if traces were sorted by activity
        trace_colors = colors[order]
    else:
        trace_colors = line_color  # single color for all traces

    # Prepare figure/axes ------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.set_facecolor(background)
    ax.set_facecolor(background)

    # Auto-scale linewidth based on figure size for consistent visual weight
    if linewidth is None:
        # Scale linewidth so traces appear equally bold regardless of figure size
        # Base: 0.8pt for a 6x6 figure. Scale proportionally with figure height.
        base_lw = 0.8
        height_ratio = figsize[1] / 6.0
        linewidth = base_lw * height_ratio

    # Base traces via LineCollection (fast & crisp)
    segs = [np.column_stack((T, Yplot[i])) for i in range(n_traces)]
    lc = LineCollection(
        segs,
        colors=trace_colors,
        linewidths=linewidth,
        antialiased=antialiased,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection(lc)

    # Optional "active" overlay -----------------------------------------------
    if active_mask is not None:
        if active_color is None:
            active_color = "#0b1c2f"  # deep navy/black-ish overlay
        act_segs = []
        for i in range(n_traces):
            m = active_mask[i]
            if m is None:
                continue
            m = np.asarray(m, dtype=bool)
            if m.shape[0] != T.size:
                raise ValueError("active_mask must match selected time window length.")
            if not np.any(m):
                continue
            # find contiguous True runs
            idx = np.flatnonzero(m)
            # boundaries where consecutive indices break
            cuts = np.where(np.diff(idx) > 1)[0]
            starts = np.r_[idx[0], idx[cuts + 1]]
            ends = np.r_[idx[cuts], idx[-1]]
            for s, e in zip(starts, ends, strict=False):
                # include end+1 so segment reaches the last True sample
                sl = slice(s, e + 1)
                act_segs.append(np.column_stack((T[sl], Yplot[i, sl])))

        if act_segs:
            lc_act = LineCollection(
                act_segs,
                colors=active_color,
                linewidths=active_linewidth,
                antialiased=True,
                alpha=1.0,
                capstyle="round",
                joinstyle="round",
            )
            ax.add_collection(lc_act)

    # Clean look: no spines/ticks/labels; tight margins
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=0)
    ax.set_xlim(T[0], T[-1])
    ymin = -vertical_spacing * 0.1
    ymax = offsets.max() + vertical_spacing * 1.1
    ax.set_ylim(ymin, ymax)
    ax.set_clip_on(True)
    fig.tight_layout(pad=0.1)

    return fig, ax
