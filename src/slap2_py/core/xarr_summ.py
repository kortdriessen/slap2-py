import os as os

import numpy as np
import xarray as xr
from numcodecs import Blosc


def dF_data_to_xr(
    trial_data: dict, trace_type: str, fs: float
) -> dict[str, xr.DataArray]:
    """Convert trial_data dict to a dict of DataArrays, one per DMD.

    Each DataArray has dimensions (channel, syn_id, time) with time coordinates
    computed from the sampling rate. Trials are concatenated along the time axis.
    Data is cast to float32 and stored C-contiguous.
    Args:
        trial_data: dict of trial data, structured as trial_data[dmd][trial][trace_type]
        trace_type: which trace type to extract (options are 'ls', 'events', 'denoised')
        fs: sampling rate in Hz
    """
    result = {}
    for dmd in sorted(trial_data.keys()):
        trials = trial_data[dmd]
        # Raw shape: (n_channels, n_timepoints, n_synapses)
        arrays = [trials[t]["dF"][trace_type] for t in sorted(trials.keys())]
        concat = np.concatenate(arrays, axis=1)

        # Transpose to (channel, syn_id, time) and make C-contiguous float32
        data = np.ascontiguousarray(concat.transpose(0, 2, 1), dtype=np.float32)

        n_channels, n_synapses, n_timepoints = data.shape
        result[f"dmd_{dmd}"] = xr.DataArray(
            data,
            dims=["channel", "syn_id", "time"],
            coords={
                "channel": np.arange(n_channels),
                "syn_id": np.arange(n_synapses),
                "time": np.arange(n_timepoints) / fs,
            },
        )

    return result


def F0_data_to_xr(trial_data: dict, fs: float) -> dict[str, xr.DataArray]:
    """Convert trial_data dict to a dict of DataArrays, one per DMD.

    Each DataArray has dimensions (channel, syn_id, time) with time coordinates
    computed from the sampling rate. Trials are concatenated along the time axis.
    Data is cast to float32 and stored C-contiguous.
    Args:
        trial_data: dict of trial data, structured as trial_data[dmd][trial][trace_type]
        fs: sampling rate in Hz
    """
    result = {}
    for dmd in sorted(trial_data.keys()):
        trials = trial_data[dmd]
        # Raw shape: (n_channels, n_timepoints, n_synapses)
        arrays = [trials[t]["F0"] for t in sorted(trials.keys())]
        concat = np.concatenate(arrays, axis=1)

        # Transpose to (channel, syn_id, time) and make C-contiguous float32
        data = np.ascontiguousarray(concat.transpose(0, 2, 1), dtype=np.float32)

        n_channels, n_synapses, n_timepoints = data.shape
        result[f"dmd_{dmd}"] = xr.DataArray(
            data,
            dims=["channel", "syn_id", "time"],
            coords={
                "channel": np.arange(n_channels),
                "syn_id": np.arange(n_synapses),
                "time": np.arange(n_timepoints) / fs,
            },
        )

    return result


def ROI_data_to_xr(
    trial_data: dict, roi_type: str, fs: float, roi_info: list[dict]
) -> dict[str, xr.DataArray]:
    """Convert trial_data dict to a dict of DataArrays, one per DMD.

    Each DataArray has dimensions (channel, syn_id, time) with time coordinates
    computed from the sampling rate. Trials are concatenated along the time axis.
    Data is cast to float32 and stored C-contiguous.
    Args:
        trial_data: dict of trial data, structured as trial_data[dmd][trial][trace_type]
        fs: sampling rate in Hz
    """
    result = {}
    for dmd in sorted(trial_data.keys()):
        rinf = roi_info[dmd - 1]
        if len(rinf) == 0:
            continue

        trials = trial_data[dmd]
        roi_names = []
        for roi_meta in rinf:
            roi_names.append(roi_meta["Label"])
        roi_ids = np.array(roi_names)
        # Raw shape: (n_timepoints, n_channels, n_somas)
        arrays = [trials[t]["ROIs"][roi_type] for t in sorted(trials.keys())]
        concat = np.concatenate(arrays, axis=0)

        # Transpose to (channel, soma_id, time) and make C-contiguous float32
        data = np.ascontiguousarray(concat.transpose(1, 2, 0), dtype=np.float32)

        n_channels, n_somas, n_timepoints = data.shape
        result[f"dmd_{dmd}"] = xr.DataArray(
            data,
            dims=["channel", "soma_id", "time"],
            coords={
                "channel": np.arange(n_channels),
                "soma_id": roi_ids,
                "time": np.arange(n_timepoints) / fs,
            },
        )

    return result


def save_xr_to_zarr(das: dict[str, xr.DataArray], path: str):
    """Save dict of DataArrays to a Zarr store, one group per DMD."""
    compressor = Blosc(cname="zstd", clevel=3)
    for key, da in das.items():
        n_ch, n_syn, n_time = da.shape
        chunks = (1, 1, n_time)  # one channel + one synapse per chunk
        ds = da.to_dataset(name="data")
        ds.to_zarr(
            path,
            group=key,
            mode="w" if key == sorted(das.keys())[0] else "a",
            encoding={"data": {"chunks": chunks, "compressor": compressor}},
        )


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
