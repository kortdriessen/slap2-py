# ===============================================
# This module contains functions for plotting activity data,
# so far, this deals with glutamate data from localized sources,
# and calcium data from manually identified ROIs.
# ===============================================


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from typing import Sequence, Optional, Tuple, Dict, Literal, Union


def plot_timeseries_raster(
    series_list: Sequence[np.ndarray],
    timestamps: np.ndarray,
    *,
    # visualization
    cmap: str = "bwr",
    vcenter: float = 1.0,
    robust: bool = True,
    q: Tuple[float, float] = (0.01, 0.99),  # robust percentiles
    gap: float = 0.05,  # vertical gap between rows (in “row-height” units)
    row_height: float = 1.0,  # actual bar height (controls overall plot height)
    gap_indices: Optional[Union[Sequence[int], Dict[int, int]]] = None,
    show_colorbar: bool = False,
    labels: Optional[Sequence[str]] = None,
    hide_yticks: bool = True,
    # axes management
    ax: Optional[plt.Axes] = None,  # bars axis (optional)
    tight: bool = True,
    # NEW: optional line panel above
    line_series: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    line_height_ratio: float = 0.28,  # relative height of line panel vs bars panel
    line_kwargs: Optional[dict] = None,  # passed to ax_line.plot(...)
    # NEW: optional per-chunk line panels (above each chunk)
    chunked_line_arrays: Optional[Sequence[np.ndarray]] = None,
    chunk_line_height_ratio: float = 0.1,  # min height of each chunk line axis vs bars axis
    line_labels: Optional[Sequence[str]] = None,  # labels for each chunk line plot
) -> Dict[str, object]:
    """
    Plot a list of time series as a compact raster (one strip per series) using pcolormesh,
    with color centered at vcenter (default=1.0) in the 'bwr' colormap.
    If `line_series` is provided, draw one or more line plots in separate axis panels
    stacked ABOVE the raster, sharing the same x-axis (timestamps) for perfect alignment.
    If `chunked_line_arrays` is provided, draw one line per chunk (chunks are defined by
    `gap_indices` separators). The first chunk’s line is placed in a top panel above the raster;
    each subsequent chunk’s line is placed inside the corresponding inserted blank spacer region.

    Parameters
    ----------
    series_list : list of 1D np.ndarray
        All must have equal length T.
    timestamps : 1D np.ndarray, shape (T,)
        Sample times corresponding to each series value (can be non-uniform).
    cmap : str
        Colormap name (default 'bwr').
    vcenter : float
        Value mapped to the colormap's center (white for 'bwr'). Default 1.0.
    robust : bool
        If True, use robust percentiles to choose vmin/vmax and then make them
        symmetric around vcenter.
    q : (low, high)
        Percentiles for robust scaling if robust=True.
    gap : float
        Vertical spacing between series strips (in “row-height” units). This is added
        between every adjacent strip.
    row_height : float
        Height of each colored bar strip. Reducing this makes the raster more compact
        vertically. Defaults to 1.0 to preserve historical appearance.
    gap_indices : sequence of int or dict[int, int], optional
        Insert blank (NaN) spacer rows AFTER the specified 0-based indices.
        If a sequence is given, insert 1 spacer after each index.
        If a dict is given, values indicate how many spacers to insert per index.
    show_colorbar : bool
        If True, add a colorbar attached to the raster axis.
    labels : list of str, optional
        Per-series labels for y-axis. If None, rows are unlabeled.
    hide_yticks : bool
        If True, hide y-axis ticks for compactness.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw the raster on. If provided together with line_series,
        the function will *repack* the provided axis to make vertical room for the
        line panel above (preserving figure and position).
    tight : bool
        If True, call tight_layout at the end (may be ignored when repacking axes).
    line_series : 1D np.ndarray or list of 1D np.ndarray, optional
        If provided (each with shape (T,)), one or multiple line plots will be drawn
        above the raster aligned to the same timestamps.
    line_height_ratio : float
        Total height of the stacked line panels relative to the bars panel height.
        For example, 0.28 means all line panels together will be ~28% as tall as the
        bars panel. If multiple line series are given, each gets an equal share of
        this total line height.
    line_kwargs : dict
        Extra kwargs passed to ax_line.plot (e.g., {'lw':1.2}).
    chunked_line_arrays : list of 1D np.ndarray, optional
        If provided, must have length equal to the number of chunks defined by
        `gap_indices` (i.e., number of separators + 1). The first array is drawn in a
        top panel above the raster; remaining arrays are drawn into the inserted blank
        spacer regions that separate the chunks. Each array must have shape (T,).
    chunk_line_height_ratio : float
        Minimum height (as a fraction of the bars axis height) for each per‑chunk line
        axis drawn inside the raster area. If a gap’s physical height is smaller than
        this minimum, the axis will expand beyond the gap, centered at the gap, so that
        the line is still readable.
    line_labels : list of str, optional
        Labels for the per-chunk line plots. Must have the same length as
        `chunked_line_arrays`. Each label is rendered in bold at the top-left of the
        corresponding chunk line axis.

    Returns
    -------
    dict with keys: fig, ax (raster axis), ax_line (or None), ax_lines (or None),
                    chunk_line_axes (or None),
                    quadmesh, norm, vmin, vmax, time_edges, y_edges
    """
    # ---------- validate & stack ----------
    if len(series_list) == 0:
        raise ValueError("series_list is empty.")
    T = len(timestamps)
    for i, arr in enumerate(series_list):
        if arr.shape != (T,):
            raise ValueError(
                f"series_list[{i}] has shape {arr.shape}, expected ({T},)."
            )
    data = np.vstack([np.asarray(a, dtype=float) for a in series_list])  # (N, T)
    orig_N = data.shape[0]

    # ---------- expand rows by inserting blank spacers after requested indices ----------
    counts_map: Dict[int, int] = {}
    if gap_indices is not None:
        if isinstance(gap_indices, dict):
            counts_map = {int(k): int(v) for k, v in gap_indices.items()}
        else:
            counts_map = {int(i): 6 for i in gap_indices}
        if any(idx < 0 or idx >= orig_N for idx in counts_map.keys()):
            raise ValueError("gap_indices contain out-of-range index.")
        if any(v <= 0 for v in counts_map.values()):
            raise ValueError("gap_indices counts must be positive.")

        expanded_rows = []
        for i in range(orig_N):
            expanded_rows.append(data[i])
            num_blanks = counts_map.get(i, 0)
            if num_blanks:
                expanded_rows.extend(
                    [np.full((T,), np.nan, dtype=float) for _ in range(num_blanks)]
                )
        data = np.vstack(expanded_rows)

    N = data.shape[0]

    # ---------- timestamps -> edges (handles non-uniform dt) ----------
    t = np.asarray(timestamps, dtype=float)
    if np.any(~np.isfinite(t)):
        raise ValueError("timestamps contain non-finite values.")
    diffs = np.diff(t)
    if np.any(diffs < 0):
        raise ValueError("timestamps must be monotonically non-decreasing.")
    if np.any(diffs == 0):
        # jitter exact duplicates to make strictly increasing for pcolormesh
        eps = np.finfo(float).eps
        for k in np.where(diffs == 0)[0]:
            t[k + 1] = t[k + 1] + eps
        diffs = np.diff(t)

    mids = t[:-1] + 0.5 * diffs
    left_edge = t[0] - 0.5 * diffs[0]
    right_edge = t[-1] + 0.5 * diffs[-1]
    time_edges = np.concatenate(([left_edge], mids, [right_edge]))  # (T+1,)

    # ---------- vertical edges for raster ----------
    if not np.isfinite(row_height) or row_height <= 0:
        raise ValueError("row_height must be a positive finite number.")
    y_step = float(row_height) + float(gap)
    y_edges = np.arange(N + 1, dtype=float) * y_step  # (N+1,)

    # ---------- colormap normalization centered at vcenter ----------
    flat = data[np.isfinite(data)]
    if flat.size == 0:
        raise ValueError("All data are non-finite.")
    if robust:
        lo = np.quantile(flat, q[0])
        hi = np.quantile(flat, q[1])
    else:
        lo = float(np.min(flat))
        hi = float(np.max(flat))

    # symmetric around vcenter so white=1 and both sides use full dynamic range
    dist_lo = max(1e-9, vcenter - lo)
    dist_hi = max(1e-9, hi - vcenter)
    radius = max(dist_lo, dist_hi)
    vmin = vcenter - radius
    vmax = vcenter + radius
    if not (vmin < vcenter < vmax):
        vmin = vcenter - 1.0
        vmax = vcenter + 1.0

    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # ---------- determine chunks and validate chunked_line_arrays ----------
    # Identify blank rows (inserted spacers) to split chunks
    is_blank_row = np.all(~np.isfinite(data), axis=1)
    chunks: list[Tuple[int, int]] = (
        []
    )  # (start_row_idx, end_row_idx) inclusive, in expanded data
    gaps_after: list[Tuple[int, int]] = (
        []
    )  # for each chunk except last: (gap_start_row, gap_len)
    i_row = 0
    while i_row < N:
        # skip any leading blanks (robustness)
        while i_row < N and is_blank_row[i_row]:
            i_row += 1
        if i_row >= N:
            break
        start = i_row
        while i_row < N and not is_blank_row[i_row]:
            i_row += 1
        end = i_row - 1
        chunks.append((start, end))
        # now collect gap after this chunk
        gap_start = i_row
        gap_len = 0
        while i_row < N and is_blank_row[i_row]:
            gap_len += 1
            i_row += 1
        if gap_len > 0:
            gaps_after.append((gap_start, gap_len))
    num_chunks = len(chunks)

    chunk_line_axes: Optional[list[plt.Axes]] = None

    # Prepare optional global line arrays (top panels) and optional per-chunk arrays
    ax_lines = None
    line_arrays: Optional[Sequence[np.ndarray]] = None
    if line_series is not None:
        if isinstance(line_series, np.ndarray):
            arr = np.asarray(line_series, dtype=float)
            if arr.shape != (T,):
                raise ValueError(f"line_series has shape {arr.shape}, expected ({T},).")
            line_arrays = [arr]
        elif isinstance(line_series, (list, tuple)):
            line_arrays = []
            for idx_arr, arr_like in enumerate(line_series):
                arr = np.asarray(arr_like, dtype=float)
                if arr.shape != (T,):
                    raise ValueError(
                        f"line_series[{idx_arr}] has shape {arr.shape}, expected ({T},)."
                    )
                line_arrays.append(arr)
        else:
            raise ValueError(
                "line_series must be a 1D array or a list/tuple of 1D arrays."
            )

    chunk_arrays: Optional[list[np.ndarray]] = None
    if chunked_line_arrays is not None:
        if not isinstance(chunked_line_arrays, (list, tuple)):
            raise ValueError("chunked_line_arrays must be a list/tuple of 1D arrays.")
        # Convert and validate shapes
        chunk_arrays = []
        for idx_arr, arr_like in enumerate(chunked_line_arrays):
            arr = np.asarray(arr_like, dtype=float)
            if arr.shape != (T,):
                raise ValueError(
                    f"chunked_line_arrays[{idx_arr}] has shape {arr.shape}, expected ({T},)."
                )
            chunk_arrays.append(arr)
        if num_chunks == 0 and len(chunk_arrays) > 0:
            raise ValueError(
                "chunked_line_arrays provided but no non-blank chunks were found."
            )
        if num_chunks > 0 and len(chunk_arrays) != num_chunks:
            raise ValueError(
                f"chunked_line_arrays length ({len(chunk_arrays)}) must equal number of chunks ({num_chunks})."
            )
        if line_labels is not None and len(line_labels) != len(chunk_arrays):
            raise ValueError(
                f"line_labels length ({len(line_labels)}) must equal length of chunked_line_arrays ({len(chunk_arrays)})."
            )
    else:
        if line_labels is not None:
            raise ValueError("line_labels provided but chunked_line_arrays is None.")

    # ---------- axes creation (handle optional top panels) ----------
    # Top panels include: global line_series axes (if any) PLUS the first chunk's line (if any)
    num_top_axes = (len(line_arrays) if line_arrays is not None else 0) + (
        1 if (chunk_arrays is not None and num_chunks > 0) else 0
    )
    if num_top_axes > 0:
        if ax is None:
            # Figure height scales with row_height
            fig = plt.figure(figsize=(16, max(2.4, 0.25 * (N * row_height) + 0.8)))
            import matplotlib.gridspec as gridspec

            # Distribute total line height 'line_height_ratio' equally among top axes
            line_weights = [line_height_ratio / max(1, num_top_axes)] * num_top_axes
            height_ratios = line_weights + [1.0]
            gs = gridspec.GridSpec(
                nrows=num_top_axes + 1,
                ncols=1,
                height_ratios=height_ratios,
                hspace=0.05,
                figure=fig,
            )
            top_axes: list[plt.Axes] = []
            # Create global line_series axes first (top to bottom order of gs)
            if line_arrays is not None and len(line_arrays) > 0:
                for j in range(len(line_arrays)):
                    if j == 0:
                        ax_j = fig.add_subplot(gs[j])
                    else:
                        ax_j = fig.add_subplot(gs[j], sharex=top_axes[0])
                    top_axes.append(ax_j)
            # Then add the first chunk line axis (closest to bars)
            if chunk_arrays is not None and num_chunks > 0:
                j = len(top_axes)
                if j == 0:
                    ax_chunk0 = fig.add_subplot(gs[j])
                else:
                    ax_chunk0 = fig.add_subplot(gs[j], sharex=top_axes[0])
                top_axes.append(ax_chunk0)
            # Bars axis shares x with the first top axis (if present)
            if len(top_axes) > 0:
                ax = fig.add_subplot(gs[num_top_axes], sharex=top_axes[0])
            else:
                ax = fig.add_subplot(gs[num_top_axes])
            # Assign to return variables
            ax_lines = (
                top_axes[: len(line_arrays) if line_arrays is not None else 0] or None
            )
            # chunk_line_axes list will be filled later; include top chunk axis if present
            if chunk_arrays is not None and num_chunks > 0:
                if ax_lines is None:
                    chunk_line_axes = [top_axes[0]]
                else:
                    chunk_line_axes = [top_axes[-1]]
        else:
            # Repack existing bars axis and add stacked top axes above it
            fig = ax.figure
            pos = ax.get_position()
            total_h = pos.height
            top_h_total = total_h * (line_height_ratio / (1.0 + line_height_ratio))
            bars_h = total_h - top_h_total
            # Update bars axis (keep same bottom)
            ax.set_position([pos.x0, pos.y0, pos.width, bars_h])
            # Create top axes stacked directly above bars axis
            top_axes: list[plt.Axes] = []
            if num_top_axes > 0:
                each_h = top_h_total / num_top_axes
                for j in range(num_top_axes):
                    y0 = pos.y0 + bars_h + j * each_h
                    ax_j = fig.add_axes([pos.x0, y0, pos.width, each_h], sharex=ax)
                    top_axes.append(ax_j)
            # Split top_axes into global line axes and first chunk axis
            n_line_axes = len(line_arrays) if line_arrays is not None else 0
            ax_lines = top_axes[:n_line_axes] or None
            if chunk_arrays is not None and num_chunks > 0:
                # First chunk line is the last of the top axes (closest to bars)
                if chunk_line_axes is None:
                    chunk_line_axes = [top_axes[-1]]
                else:
                    chunk_line_axes.append(top_axes[-1])
    else:
        # No top panels: keep single axis behavior
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, max(2, 0.25 * (N * row_height) + 0.5)))
        else:
            fig = ax.figure

    # ---------- draw raster ----------
    quad = ax.pcolormesh(
        time_edges, y_edges, data, cmap=cmap, norm=norm, shading="flat"
    )
    ax.invert_yaxis()
    ax.set_xlim(time_edges[0], time_edges[-1])  # explicit for shared x alignment

    labels_expanded: Optional[Sequence[str]] = None
    if labels is not None:
        if len(labels) == orig_N:
            # Expand labels to account for inserted blank rows
            labels_tmp: list[str] = []
            for i in range(orig_N):
                labels_tmp.append(labels[i])
                num_blanks = 0
                if gap_indices is not None:
                    num_blanks = counts_map.get(i, 0)
                if num_blanks:
                    labels_tmp.extend([""] * num_blanks)
            labels_expanded = labels_tmp
        elif len(labels) == N:
            labels_expanded = list(labels)
        else:
            raise ValueError(
                "labels length must match original series count or expanded count after gaps."
            )

    if labels_expanded is not None:
        centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        ax.set_yticks(centers)
        ax.set_yticklabels(labels_expanded if not hide_yticks else [""] * N)
    elif hide_yticks:
        ax.set_yticks([])

    ax.set_xlabel("Time")
    # ax.set_ylabel("Series")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if show_colorbar:
        cbar = fig.colorbar(quad, ax=ax, pad=0.02)
        cbar.set_label("Value")

    # ---------- draw global line(s) (if requested) ----------
    if ax_lines is not None and line_arrays is not None:
        kw = dict(lw=1.2)
        if line_kwargs:
            kw.update(line_kwargs)
        for ax_l, arr in zip(ax_lines, line_arrays):
            ax_l.plot(t, arr, **kw)
            ax_l.spines["right"].set_visible(False)
            ax_l.spines["top"].set_visible(False)
            ax_l.set_xlim(time_edges[0], time_edges[-1])  # identical to raster
            # Hide x tick labels on the line panels to avoid duplication with bars
            plt.setp(ax_l.get_xticklabels(), visible=False)
            ax_l.set_ylabel("")  # minimalist
            ax_l.grid(False)

    # ---------- draw per-chunk line(s) ----------
    if chunk_arrays is not None and num_chunks > 0:
        if chunk_line_axes is None:
            chunk_line_axes = []
        kw = dict(lw=1.2)
        if line_kwargs:
            kw.update(line_kwargs)

        # We already handled chunk 0 in the top panels (added to chunk_line_axes).
        # Now add axes for subsequent chunks inside their blank spacer regions.
        # Map spacer rows (in data coordinates) to figure coordinates.
        pos = ax.get_position()
        y_total = y_edges[-1] if y_edges.size > 0 else 1.0

        # Helper to convert data-y to figure y in [0,1]
        def data_y_to_fig_y(y_data: float) -> float:
            # Inverted y-axis: y=0 maps to top of bars axis
            frac = 1.0 - (y_data / y_total)
            return pos.y0 + frac * pos.height

        # chunk 0 already has an axis in top panels (if present)
        start_chunk_idx = 1 if len(chunk_line_axes) > 0 else 0
        for j in range(start_chunk_idx, num_chunks):
            # For chunk j>0, place axis in the spacer below it (i.e., gap after chunk j-1)
            if j == 0:
                # If there was no top panel created (unlikely), skip gracefully
                continue
            gap_group_index = j - 1
            if gap_group_index < 0 or gap_group_index >= len(gaps_after):
                # No spacer info; skip safely
                continue
            gap_start_row, gap_len = gaps_after[gap_group_index]
            # Compute the exact vertical interval of this spacer group
            y_top = y_edges[gap_start_row]  # upper edge (smaller data y)
            y_bot = y_edges[gap_start_row + gap_len]  # lower edge (larger data y)
            # Convert to figure coords
            fig_y_top = data_y_to_fig_y(y_top)
            fig_y_bot = data_y_to_fig_y(y_bot)
            # Center the axis on the gap but ensure a minimum height so it remains legible
            gap_h_fig = abs(fig_y_top - fig_y_bot)
            min_h_fig = max(1e-6, float(chunk_line_height_ratio)) * pos.height
            h_fig = max(gap_h_fig, min_h_fig)
            if h_fig > pos.height:
                h_fig = pos.height
            center_fig = 0.5 * (fig_y_top + fig_y_bot)
            y0_fig = center_fig - 0.5 * h_fig
            # Clamp to stay inside the bars axis vertical bounds
            y0_fig = max(pos.y0, min(y0_fig, pos.y0 + pos.height - h_fig))
            if h_fig <= 1e-6:
                continue
            ax_chunk = fig.add_axes([pos.x0, y0_fig, pos.width, h_fig], sharex=ax)
            ax_chunk.set_facecolor("none")  # overlay lines without obscuring raster
            ax_chunk.set_zorder(3)
            chunk_line_axes.append(ax_chunk)

        # Now plot each chunk array onto its corresponding axis in order
        # chunk_line_axes length should match len(chunk_arrays)
        if len(chunk_line_axes) != len(chunk_arrays):
            # Length mismatch can happen if some spacers were too small; fall back:
            # ensure we only plot for the available axes.
            usable = min(len(chunk_line_axes), len(chunk_arrays))
            axes_to_use = chunk_line_axes[:usable]
            arrays_to_use = chunk_arrays[:usable]
            labels_to_use = (
                list(line_labels[:usable])
                if line_labels is not None
                else [None] * usable
            )
        else:
            axes_to_use = chunk_line_axes
            arrays_to_use = chunk_arrays
            labels_to_use = (
                list(line_labels)
                if line_labels is not None
                else [None] * len(arrays_to_use)
            )

        for idx_label, (ax_c, arr) in enumerate(zip(axes_to_use, arrays_to_use)):
            ax_c.plot(t, arr, **kw)
            ax_c.spines["right"].set_visible(False)
            ax_c.spines["top"].set_visible(False)
            ax_c.set_xlim(time_edges[0], time_edges[-1])
            plt.setp(ax_c.get_xticklabels(), visible=False)
            ax_c.set_ylabel("")
            ax_c.grid(False)
            label_text = labels_to_use[idx_label] if labels_to_use else None
            if label_text:
                ax_c.text(
                    0.01,
                    0.99,
                    str(label_text),
                    transform=ax_c.transAxes,
                    ha="left",
                    va="top",
                    fontweight="bold",
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
                    zorder=4,
                )

    if tight and ax_lines is None:
        fig.tight_layout()

    return dict(
        fig=fig,
        ax=ax,
        ax_line=(ax_lines[0] if ax_lines else None),
        ax_lines=ax_lines,
        chunk_line_axes=chunk_line_axes,
        quadmesh=quad,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        time_edges=time_edges,
        y_edges=y_edges,
    )
