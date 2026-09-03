import os as os
import shutil
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from numcodecs import Blosc

import slap2_py as spy

if TYPE_CHECKING:
    import matplotlib.figure


def _build_time_coord(
    per_trial_sample_counts: list[int],
    fs: float,
    trial_epochs: np.ndarray | None,
    epoch_offsets_s: dict[int, float] | None,
) -> np.ndarray:
    """Build the concatenated-trial time coordinate.

    - Single-epoch (both kwargs None): returns ``np.arange(total_n) / fs``.
    - Multi-epoch: for each trial, samples start at
      ``epoch_offsets_s[trial_epochs[t]] + (within_epoch_sample_idx / fs)``
      and continue contiguously through the trial. Within-epoch index
      resets at the first trial of each new epoch.

    Parameters
    ----------
    per_trial_sample_counts : list[int]
        Length of the time axis for each trial, in trial-index order.
    fs : float
        Sampling rate (Hz).
    trial_epochs : np.ndarray | None
        1-based epoch index per trial (length = len(per_trial_sample_counts)).
    epoch_offsets_s : dict[int, float] | None
        Seconds-since-epoch-1 anchor per epoch.
    """
    if trial_epochs is None and epoch_offsets_s is None:
        total = int(sum(per_trial_sample_counts))
        return np.arange(total, dtype=np.float64) / float(fs)
    if trial_epochs is None or epoch_offsets_s is None:
        raise ValueError(
            "trial_epochs and epoch_offsets_s must both be provided or both be None"
        )
    if len(trial_epochs) != len(per_trial_sample_counts):
        raise ValueError(
            f"trial_epochs length {len(trial_epochs)} does not match number "
            f"of trials {len(per_trial_sample_counts)}"
        )

    pieces: list[np.ndarray] = []
    within_epoch_counter: dict[int, int] = {}
    for n_samp, epoch in zip(per_trial_sample_counts, trial_epochs, strict=True):
        e = int(epoch)
        if e not in epoch_offsets_s:
            raise KeyError(
                f"epoch_offsets_s missing entry for epoch {e}; "
                f"have {sorted(epoch_offsets_s.keys())}"
            )
        start_idx = within_epoch_counter.get(e, 0)
        pieces.append(
            epoch_offsets_s[e]
            + (np.arange(start_idx, start_idx + n_samp, dtype=np.float64) / float(fs))
        )
        within_epoch_counter[e] = start_idx + n_samp
    return np.concatenate(pieces)


def get_dFx(
    expt_summary_path: str,
    trace_type: str,
    *,
    epoch_offsets_s: dict[int, float] | None = None,
    acq_dir: str | None = None,
):
    """High-level convenience: ExSum path → dict[str, DataArray] for dF.

    For multi-epoch ExSums, either pass ``epoch_offsets_s`` explicitly, or
    pass ``acq_dir`` to have offsets computed from the raw ``.dat`` files.
    Single-epoch ExSums need neither.
    """
    trial_data, refdata, fs, ntrials, trial_epochs = spy.xsum.read_full_trial_data_dict(
        expt_summary_path
    )
    multi_epoch = int(trial_epochs.max()) > 1
    clean_trials = spy.xsum.get_clean_trial_dict(
        trial_data, trial_epochs=trial_epochs if multi_epoch else None
    )
    trial_data = spy.xsum.replace_bad_trials_with_null_data(
        trial_data, clean_trials, trial_epochs=trial_epochs if multi_epoch else None
    )
    spy.xsum.check_all_trial_shapes_match(
        trial_data, clean_trials, trial_epochs=trial_epochs if multi_epoch else None
    )
    if multi_epoch and epoch_offsets_s is None:
        if acq_dir is None:
            raise ValueError(
                f"ExSum {expt_summary_path} is multi-epoch "
                f"(max epoch = {int(trial_epochs.max())}); pass either "
                f"epoch_offsets_s= or acq_dir= so per-epoch offsets can be "
                f"computed."
            )
        from slap2_py.utils.datfile import compute_epoch_offsets_from_dats

        epoch_offsets_s = compute_epoch_offsets_from_dats(acq_dir)
    return dF_data_to_xr(
        trial_data,
        trace_type,
        fs,
        trial_epochs=trial_epochs if multi_epoch else None,
        epoch_offsets_s=epoch_offsets_s if multi_epoch else None,
    )


def dF_data_to_xr(
    trial_data: dict,
    trace_type: str,
    fs: float,
    *,
    trial_epochs: np.ndarray | None = None,
    epoch_offsets_s: dict[int, float] | None = None,
) -> dict[str, xr.DataArray]:
    """Convert trial_data dict to a dict of DataArrays, one per DMD.

    Single-epoch (both ``trial_epochs`` and ``epoch_offsets_s`` None):
    trials are concatenated along the time axis and the ``time`` coord is
    ``np.arange(n_total) / fs``.

    Multi-epoch (both provided): trials are still concatenated along time,
    but the ``time`` coord is built per-trial so that each trial's samples
    start at ``epoch_offsets_s[trial_epochs[t]] + within_epoch_sample_idx/fs``.
    The resulting coord is strictly monotonic with jumps at epoch boundaries
    (no NaN padding of the gaps).

    Data is cast to float32 and stored C-contiguous.
    """
    result = {}
    for dmd in sorted(trial_data.keys()):
        trials = trial_data[dmd]
        ordered_keys = sorted(trials.keys())
        # Raw per-trial shape: (n_channels, n_timepoints, n_synapses)
        arrays = [trials[t]["dF"][trace_type] for t in ordered_keys]
        per_trial_samples = [a.shape[1] for a in arrays]
        concat = np.concatenate(arrays, axis=1)

        # Transpose to (channel, syn_id, time) and make C-contiguous float32
        data = np.ascontiguousarray(concat.transpose(0, 2, 1), dtype=np.float32)

        n_channels, n_synapses, _n_timepoints = data.shape
        time_coord = _build_time_coord(
            per_trial_samples, fs, trial_epochs, epoch_offsets_s
        )
        result[f"dmd_{dmd}"] = xr.DataArray(
            data,
            dims=["channel", "syn_id", "time"],
            coords={
                "channel": np.arange(n_channels),
                "syn_id": np.arange(n_synapses),
                "time": time_coord,
            },
        )

    return result


def F0_data_to_xr(
    trial_data: dict,
    fs: float,
    *,
    trial_epochs: np.ndarray | None = None,
    epoch_offsets_s: dict[int, float] | None = None,
) -> dict[str, xr.DataArray]:
    """Same contract as :func:`dF_data_to_xr` but for the per-trial F0
    baseline trace."""
    result = {}
    for dmd in sorted(trial_data.keys()):
        trials = trial_data[dmd]
        ordered_keys = sorted(trials.keys())
        # Raw per-trial shape: (n_channels, n_timepoints, n_synapses)
        arrays = [trials[t]["F0"] for t in ordered_keys]
        per_trial_samples = [a.shape[1] for a in arrays]
        concat = np.concatenate(arrays, axis=1)

        # Transpose to (channel, syn_id, time) and make C-contiguous float32
        data = np.ascontiguousarray(concat.transpose(0, 2, 1), dtype=np.float32)

        n_channels, n_synapses, _n_timepoints = data.shape
        time_coord = _build_time_coord(
            per_trial_samples, fs, trial_epochs, epoch_offsets_s
        )
        result[f"dmd_{dmd}"] = xr.DataArray(
            data,
            dims=["channel", "syn_id", "time"],
            coords={
                "channel": np.arange(n_channels),
                "syn_id": np.arange(n_synapses),
                "time": time_coord,
            },
        )

    return result


def ROI_data_to_xr(
    trial_data: dict,
    roi_type: str,
    fs: float,
    roi_info: list[dict],
    *,
    trial_epochs: np.ndarray | None = None,
    epoch_offsets_s: dict[int, float] | None = None,
) -> dict[str, xr.DataArray]:
    """Same contract as :func:`dF_data_to_xr` but for ROI (soma) traces.

    Raw per-trial shape is ``(n_timepoints, n_channels, n_somas)``; trials
    are concatenated along axis 0 (time).
    """
    result = {}
    for dmd in sorted(trial_data.keys()):
        rinf = roi_info[dmd - 1]
        if len(rinf) == 0:
            continue

        trials = trial_data[dmd]
        ordered_keys = sorted(trials.keys())
        roi_names = []
        for roi_meta in rinf:
            roi_names.append(roi_meta["Label"])
        roi_ids = np.array(roi_names)
        # Raw per-trial shape: (n_timepoints, n_channels, n_somas)
        arrays = [trials[t]["ROIs"][roi_type] for t in ordered_keys]
        per_trial_samples = [a.shape[0] for a in arrays]
        concat = np.concatenate(arrays, axis=0)

        # Transpose to (channel, soma_id, time) and make C-contiguous float32
        data = np.ascontiguousarray(concat.transpose(1, 2, 0), dtype=np.float32)

        n_channels, n_somas, _n_timepoints = data.shape
        time_coord = _build_time_coord(
            per_trial_samples, fs, trial_epochs, epoch_offsets_s
        )
        result[f"dmd_{dmd}"] = xr.DataArray(
            data,
            dims=["channel", "soma_id", "time"],
            coords={
                "channel": np.arange(n_channels),
                "soma_id": roi_ids,
                "time": time_coord,
            },
        )

    return result


def save_xr_to_zarr(das: dict[str, xr.DataArray], path: str):
    """Save dict of DataArrays to a Zarr store, one group per DMD.

    The store is written to a sibling temporary directory first and only swapped
    into place once fully written, so an interrupted or concurrent write can
    never leave a half-written (corrupt) store at ``path``. The slow part — the
    actual chunk writes — happens entirely in the temp location and never
    touches the live store; the swap itself is two fast directory renames. If
    the process dies mid-swap, both the previous store and the new one remain on
    disk (as ``{path}.bak`` / ``{path}.tmp``) and are recovered on the next call
    rather than being lost.

    This atomicity is why DMD groups being written second (``dmd_2``) used to end
    up as the corrupt one after an interrupted generation — the writer clobbered
    the live store in place. It no longer can.
    """
    if not das:
        return

    compressor = Blosc(cname="zstd", clevel=3)
    tmp_path = f"{path}.tmp"
    bak_path = f"{path}.bak"

    # Self-heal from a prior crash mid-swap: if the live store is gone but a
    # backup survived, restore it before doing anything else.
    if not os.path.exists(path) and os.path.exists(bak_path):
        os.rename(bak_path, path)

    # Clear leftovers from any previously interrupted run.
    for stale in (tmp_path, bak_path):
        if os.path.exists(stale):
            shutil.rmtree(stale)

    # Write the complete store into the temp location (never touches `path`).
    for key, da in das.items():
        n_ch, n_syn, n_time = da.shape
        chunks = (1, 1, n_time)  # one channel + one synapse per chunk
        ds = da.to_dataset(name="data")
        ds.to_zarr(
            tmp_path,
            group=key,
            mode="w" if key == sorted(das.keys())[0] else "a",
            encoding={"data": {"chunks": chunks, "compressor": compressor}},
        )

    # Swap the finished store into place. The only destructive steps are two
    # fast renames; an interruption between them leaves both stores recoverable
    # (see the self-heal above) rather than corrupt.
    if os.path.exists(path):
        os.rename(path, bak_path)
    try:
        os.rename(tmp_path, path)
    except BaseException:
        # Final swap failed — roll back to the previous store if we moved it.
        if os.path.exists(bak_path) and not os.path.exists(path):
            os.rename(bak_path, path)
        raise
    finally:
        if os.path.exists(bak_path):
            shutil.rmtree(bak_path)


def load_xr_from_zarr(
    path: str,
    dmd: str | None = None,
    sel: dict | None = None,
    isel: dict | None = None,
) -> dict[str, xr.DataArray]:
    """Load DataArrays from a Zarr store, with optional subsetting.

    Data is opened lazily via ``xr.open_zarr``, subsetted (if requested),
    and only then loaded into memory.  Because zarr stores written by
    ``save_xr_to_zarr`` are chunked as ``(1, 1, n_time)``, selecting a
    subset of synapses/channels reads only the necessary chunks from disk.

    Parameters
    ----------
    path : str
        Path to the ``.zarr`` store.
    dmd : str or None
        If given (e.g. ``"dmd_1"``), load only that group.  Otherwise load
        all groups.
    sel : dict or None
        Label-based selection passed to ``DataArray.sel()``.  For example
        ``{"syn_id": [0, 3, 7], "channel": 1}`` loads only those synapses
        and that channel.
    isel : dict or None
        Integer-index-based selection passed to ``DataArray.isel()``.

    Returns
    -------
    dict[str, xr.DataArray]
        Mapping from group name (e.g. ``"dmd_1"``) to loaded DataArray.
    """
    import zarr

    keys = [dmd] if dmd else sorted(zarr.open(path, mode="r").group_keys())
    result = {}
    for key in keys:
        ds = xr.open_zarr(path, group=key)
        da = ds["data"]
        if sel is not None:
            da = da.sel(**sel)
        if isel is not None:
            da = da.isel(**isel)
        result[key] = da.load()
    return result


def plot_actIM(
    exsum_path: str,
    *,
    dmds: tuple[int, ...] = (1, 2),
    vmin_pct: float = 0.0,
    vmax_pct: float = 98.0,
    cmap: str = "viridis",
    figsize_scale: float = 1.0 / 30.0,
    dark_bg: bool = True,
    title_prefix: str | None = None,
    hide_ticks: bool = True,
) -> dict[int, "matplotlib.figure.Figure"]:
    """Plot the activity image (``actIM``) for each DMD of an ExSum.

    Parameters
    ----------
    exsum_path : str
        Path to the ExperimentSummary ``.mat`` file.
    dmds : tuple[int, ...], optional
        DMDs to plot (1-indexed). Defaults to both.
    vmin_pct, vmax_pct : float, optional
        Percentiles (0-100) of the activity image used to set the colormap
        limits. Defaults match the typical exploratory choice (0, 98).
    cmap : str, optional
        Matplotlib colormap.
    figsize_scale : float, optional
        Figure dimensions = image pixels * figsize_scale (inches). The default
        of 1/30 matches ``plot_mean_im_with_footprints``.
    dark_bg : bool, optional
        Render under matplotlib's ``dark_background`` style (scoped locally,
        does not mutate global style state).
    title_prefix : str or None, optional
        Optional text prepended to each figure's title.
    hide_ticks : bool, optional
        If True, remove pixel tick marks.

    Returns
    -------
    dict[int, matplotlib.figure.Figure]
        Mapping from DMD number to its matplotlib Figure. Figures are not
        closed or saved — the caller owns them.
    """
    import matplotlib.pyplot as plt

    act_im = spy.xsum.get_actIM(exsum_path)

    style = "dark_background" if dark_bg else "default"
    figs: dict[int, "matplotlib.figure.Figure"] = {}
    with plt.style.context(style):
        for dmd in dmds:
            if dmd not in act_im:
                continue
            img = act_im[dmd]

            fh = img.shape[0] * figsize_scale
            fw = img.shape[1] * figsize_scale
            v_min = np.nanpercentile(img, vmin_pct)
            v_max = np.nanpercentile(img, vmax_pct)

            fig, ax = plt.subplots(1, 1, figsize=(fw, fh))
            ax.imshow(img, vmin=v_min, vmax=v_max, cmap=cmap)

            if hide_ticks:
                ax.set_xticks([])
                ax.set_yticks([])

            title = f"DMD-{dmd} actIM"
            if title_prefix:
                title = f"{title_prefix} | {title}"
            ax.set_title(title)

            figs[dmd] = fig

    return figs


def plot_mean_im_with_footprints(
    exsum_path: str,
    *,
    dmds: tuple[int, ...] = (1, 2),
    channel: int = 1,
    vmin_pct: float = 5.0,
    vmax_pct: float = 75.0,
    mean_cmap: str = "viridis",
    footprint_threshold: float | None = None,
    footprint_cmap: str = "gist_rainbow",
    footprint_alpha: float = 0.85,
    show_synapse_ids: bool = False,
    id_fontsize: float = 6.0,
    id_color: str = "white",
    figsize_scale: float = 1.0 / 30.0,
    dark_bg: bool = True,
    title_prefix: str | None = None,
    hide_ticks: bool = True,
) -> dict[int, "matplotlib.figure.Figure"]:
    """Plot the mean image overlaid with synapse footprints, one figure per DMD.

    Parameters
    ----------
    exsum_path : str
        Path to the ExperimentSummary ``.mat`` file.
    dmds : tuple[int, ...], optional
        DMDs to plot (1-indexed). Missing DMDs are silently skipped.
    channel : int, optional
        Mean-image channel (0-indexed) to display. In the standard two-indicator
        setup, channel 0 is iGluSnFR4f (green) and channel 1 is jRGECO1a (red).
    vmin_pct, vmax_pct : float, optional
        Percentiles (0-100) of the mean image used to set the colormap limits.
    mean_cmap : str, optional
        Matplotlib colormap for the mean image.
    footprint_threshold : float or None, optional
        Extra threshold applied to the underlying footprint-value map on top
        of the ExSum-level cut (0.02, baked into ``spy.xsum.get_fp_info``).
        Pixels whose footprint value is < this are hidden. None = no extra
        cut.
    footprint_cmap : str, optional
        Matplotlib colormap used to color each source in the overlay. The
        overlay is rendered from the integer source-label map (not the
        footprint value map), so every pixel of a given synapse gets the
        same bright cmap slot — no fade-to-dark at weak footprint pixels.
        Use ``"gist_rainbow"`` / ``"hsv"`` / ``"turbo"`` for distinct
        per-source hues, or e.g. ``"Reds"`` for a uniform accent.
    footprint_alpha : float, optional
        Opacity (0-1) of the footprint overlay.
    show_synapse_ids : bool, optional
        If True, draw each synapse's integer ID at its footprint centroid.
    id_fontsize : float, optional
        Font size for synapse-ID labels.
    id_color : str, optional
        Color of the synapse-ID labels.
    figsize_scale : float, optional
        Figure dimensions = image pixels * figsize_scale (inches). The default
        of 1/30 matches the canvas-image sizing used for annotation materials.
    dark_bg : bool, optional
        Render under matplotlib's ``dark_background`` style (scoped locally,
        does not mutate global style state).
    title_prefix : str or None, optional
        Optional text prepended to each figure's title.
    hide_ticks : bool, optional
        If True, remove pixel tick marks.

    Returns
    -------
    dict[int, matplotlib.figure.Figure]
        Mapping from DMD number to its matplotlib Figure. Figures are not
        closed or saved — the caller owns them.
    """
    import matplotlib.pyplot as plt

    mean_im = spy.xsum.get_meanIM(exsum_path)
    synmaps, fp_vals = spy.xsum.get_fp_info(exsum_path)

    style = "dark_background" if dark_bg else "default"
    figs = {}
    with plt.style.context(style):
        for dmd in dmds:
            if dmd not in mean_im:
                continue
            img = mean_im[dmd][channel, :, :]
            synmap = synmaps[dmd]
            fpv = fp_vals[dmd].astype(float, copy=True)

            # Overlay on synmap (integer source label, 1..N) rather than fp_vals
            # so every footprint pixel renders at the cmap's full-saturation
            # slot for its source, independent of the underlying footprint
            # strength. Gives a much brighter, uniform-contrast overlay.
            overlay = synmap.astype(float)
            overlay[synmap <= 0] = np.nan
            if footprint_threshold is not None:
                overlay[fpv < footprint_threshold] = np.nan

            source_labels = np.unique(synmap[synmap > 0])
            n_sources = int(source_labels.size)

            fh = img.shape[0] * figsize_scale
            fw = img.shape[1] * figsize_scale
            v_min = np.nanpercentile(img, vmin_pct)
            v_max = np.nanpercentile(img, vmax_pct)

            fig, ax = plt.subplots(1, 1, figsize=(fw, fh))
            ax.imshow(img, vmin=v_min, vmax=v_max, cmap=mean_cmap)
            ax.imshow(overlay, cmap=footprint_cmap, alpha=footprint_alpha)

            if hide_ticks:
                ax.set_xticks([])
                ax.set_yticks([])

            title = f"DMD-{dmd}, channel {channel}, {n_sources} sources"
            if title_prefix:
                title = f"{title_prefix} | {title}"
            ax.set_title(title)

            if show_synapse_ids:
                # synmap pixels hold (source_idx + 1); user-facing synapse ID
                # is (label - 1), matching syn_id in scopex zarrs and the
                # synapse_ids/dmd-#/{0..N-1}.png files.
                for lbl in source_labels:
                    ys, xs = np.where(synmap == lbl)
                    ax.text(
                        float(xs.mean()),
                        float(ys.mean()),
                        str(int(lbl) - 1),
                        ha="center",
                        va="center",
                        color=id_color,
                        fontsize=id_fontsize,
                    )
            figs[dmd] = fig

    return figs
