import os as os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _decode_matlab_string(data: np.ndarray) -> str | np.ndarray:
    """
    Try to convert typical MATLAB char arrays to Python str.
    Falls back to original array if it doesn't look like text.
    """
    # MATLAB chars often come as 1D/2D uint16/uint8 arrays.
    if not isinstance(data, np.ndarray):
        return data

    if data.dtype.kind in ("u", "i") and data.ndim >= 1:
        # Flatten and drop zeros, then try UTF-8 / UTF-16
        flat = data.ravel()
        # Many MATLAB strings are stored as uint16 code units:
        try:
            # Heuristic: if max value < 256, treat as UTF-8 bytes
            if flat.max(initial=0) < 256:
                return (
                    bytes(flat.astype("uint8"))
                    .decode("utf-8", errors="ignore")
                    .strip("\x00")
                )
            else:
                return (
                    flat
                    .astype("uint16")
                    .tobytes()
                    .decode("utf-16le", errors="ignore")
                    .strip("\x00")
                )
        except Exception:
            return data

    return data


def _h5_to_py(obj: h5py.Group | h5py.Dataset, f: h5py.File | None = None) -> Any:
    """
    Recursively convert an h5py Group/Dataset into Python structures.

    - Groups → dict of {name: value}
    - Datasets → numpy arrays / scalars / strings
    If *f* is provided, any h5py.Reference values found inside arrays
    are recursively dereferenced.
    """
    if isinstance(obj, h5py.Group):
        out = {}
        for key, item in obj.items():
            out[key] = _h5_to_py(item, f)
        return out

    elif isinstance(obj, h5py.Dataset):
        data = obj[()]  # read the whole dataset

        # MATLAB cell arrays / objects often show up as object arrays or references;
        # here we only handle straightforward numeric / char data cleanly.
        if isinstance(data, np.ndarray):
            # Recursively dereference HDF5 object references
            if f is not None and data.dtype == object:
                flat = data.flat
                if len(flat) > 0 and isinstance(flat[0], h5py.Reference):
                    results = [_h5_to_py(f[r], f) for r in flat]
                    out = np.empty(len(results), dtype=object)
                    for i, r in enumerate(results):
                        out[i] = r
                    return out.reshape(data.shape)

            # Convert 0-d array to Python scalar
            if data.shape == ():
                data = data.item()

            # Try to turn MATLAB char arrays into strings
            matlab_class = obj.attrs.get("MATLAB_class", None)
            if matlab_class is not None:
                try:
                    matlab_class = (
                        matlab_class.decode()
                        if isinstance(matlab_class, (bytes, bytearray))
                        else matlab_class
                    )
                except Exception:
                    pass

            if matlab_class == "char":
                data = _decode_matlab_string(data)

        # If it's a bytes object, decode
        if isinstance(data, (bytes, bytearray)):
            try:
                data = data.decode("utf-8", errors="ignore")
            except Exception:
                pass

        return data

    else:
        # Shouldn't really happen with h5py, but just in case.
        return obj


def load_mat73_to_dict(path: Path | str, root: str | None = None) -> Any:
    """
    Load a MATLAB 7.3 (HDF5) .mat file into nested Python structures.

    Parameters
    ----------
    path : Path | str
        Path to the .mat file.
    root : str or None
        If None, convert the whole file.
        If a group name (e.g. 'exptSummary'), only that group is converted.

    Returns
    -------
    Any
        Usually a dict, but can be an array or scalar depending on root.
    """
    with h5py.File(path, "r") as f:
        if root is None:
            return _h5_to_py(f)
        else:
            if root not in f:
                raise KeyError(
                    f"Root group '{root}' not found in file. Available keys: {list(f.keys())}"
                )
            return _h5_to_py(f[root])


def deref_h5(ref, file_path: str | Path):
    """Given an h5py.Reference (e.g. data['meanIM'][0][0]), return the actual array."""
    with h5py.File(file_path, "r") as f:
        return f[ref][()]  # follow the reference and read the dataset


def _h5_to_py_simple(obj: Any, f: h5py.File | None = None) -> Any:
    """Minimal recursive converter: Dataset -> array, Group -> dict.
    If *f* is provided, any h5py.Reference values found inside arrays
    are recursively dereferenced."""
    if isinstance(obj, h5py.Dataset):
        data = obj[()]  # full array
        if f is not None and isinstance(data, np.ndarray) and data.dtype == object:
            # Check if elements are HDF5 references and dereference them
            flat = data.flat
            if len(flat) > 0 and isinstance(flat[0], h5py.Reference):
                results = [_h5_to_py_simple(f[r], f) for r in flat]
                out = np.empty(len(results), dtype=object)
                for i, r in enumerate(results):
                    out[i] = r
                return out.reshape(data.shape)
        return data
    if isinstance(obj, h5py.Group):
        return {k: _h5_to_py_simple(v, f) for k, v in obj.items()}
    return obj


def _parse_h5_path(path_str: str) -> tuple[str, list[int]]:
    """
    Parse a string like "/exptSummary/userROIs[0][0]" into
    ("/exptSummary/userROIs", [0, 0]).
    """
    indices: list[int] = []
    # Pull off trailing [N] groups
    while path_str.endswith("]"):
        bracket_open = path_str.rindex("[")
        indices.append(int(path_str[bracket_open + 1 : -1]))
        path_str = path_str[:bracket_open]
    indices.reverse()
    return path_str, indices


def deref_h5_any(ref: Any, file_path: str | Path) -> Any:
    """
    Given something like data['E'][0][0] (which may be a tuple/ndarray
    containing an HDF5 object reference), open the file and return the
    actual data (array or nested dict).

    *ref* can also be a plain string path with optional bracket indices,
    e.g. ``"/exptSummary/userROIs[0][0]"``.  The path portion is used to
    look up the HDF5 dataset, the bracket indices are applied to the
    resulting array, and if that yields an h5py.Reference it is followed
    automatically.
    """
    file_path = str(file_path)

    # 0. Handle a plain string path like "/exptSummary/userROIs[0][0]"
    if isinstance(ref, str) and "/" in ref:
        h5_path, indices = _parse_h5_path(ref)
        with h5py.File(file_path, "r") as f:
            obj = f[h5_path]
            if indices:
                data = obj[()]  # read full dataset
                for idx in indices:
                    data = data[idx]
                # If indexing landed on a reference, follow it
                if isinstance(data, h5py.Reference):
                    return _h5_to_py(f[data], f)
                # Unwrap numpy scalar
                if isinstance(data, np.ndarray) and data.shape == ():
                    data = data.item()
                if isinstance(data, h5py.Reference):
                    return _h5_to_py(f[data], f)
                return data
            return _h5_to_py(obj, f)

    # 1. Unwrap containers (np.ndarray, list, tuple) until we get a reference or path
    while isinstance(ref, (np.ndarray, list, tuple)):
        if isinstance(ref, np.ndarray):
            if ref.shape == ():
                ref = ref.item()
            else:
                ref = ref.flat[0]
        else:  # list or tuple
            if len(ref) == 0:
                raise ValueError("Empty reference container")
            ref = ref[0]

    with h5py.File(file_path, "r") as f:
        # 2. Follow reference or path
        if isinstance(ref, h5py.Reference):
            obj = f[ref]
        elif isinstance(ref, (str, bytes)):
            obj = f[ref]
        else:
            raise TypeError(f"Unsupported reference type {type(ref)}: {ref!r}")

        # 3. Convert to Python structures (pass f so nested refs get resolved)
        return _h5_to_py(obj, f)


def get_exsum_basics(
    exsum_path: Path | str, root: str = "exptSummary"
) -> tuple[dict, int, int]:
    data = load_mat73_to_dict(exsum_path, root=root)
    ntrials = int(data["E"].shape[1])
    fs = int(data["params"]["analyzeHz"][0][0])
    return data, ntrials, fs


def _load_trial_epochs(exsum_path: Path | str) -> np.ndarray:
    """Load ``trialTable.epoch`` from an ExSum as a 1-D int array.

    Every ExSum (single- or multi-epoch) has this column. Single-epoch
    acqs always have all-ones. Multi-epoch acqs have values 1..n_epochs.
    """
    from slap2_py.hf import load_any

    ep = load_any(exsum_path, "/exptSummary/trialTable['epoch']")
    ep = np.asarray(ep).squeeze().astype(int, copy=False)
    if ep.ndim == 0:
        ep = ep.reshape(1)
    assert ep.ndim == 1, f"trialTable.epoch expected 1-D, got shape {ep.shape}"
    assert ep.size > 0 and int(ep.min()) >= 1, (
        f"trialTable.epoch must be non-empty and 1-indexed; got min={ep.min()}"
    )
    return ep


def read_full_trial_data_dict(exsum_path):
    """Read all per-trial data from an ExSum.

    Returns
    -------
    trial_data : dict
        ``{dmd: {trial_idx: trial_dict | "BAD_TRIAL"}}`` for DMDs 1 and 2.
    data : dict
        The full parsed ExSum top-level struct (``exptSummary``).
    fs : int
        Sampling rate in Hz.
    ntrials : int
        Number of trials.
    trial_epochs : np.ndarray
        1-D int array, length ``ntrials``, mapping each trial to its epoch
        index (1..n_epochs). For single-epoch acqs this is all ones.
    """
    data, ntrials, fs = get_exsum_basics(exsum_path)
    ntrials = data["E"].shape[1]

    trial_data = {}
    for dmd in [1, 2]:
        dmd_idx = dmd - 1
        trial_data[dmd] = {}
        for trl in range(ntrials):
            try:
                tdat = deref_h5_any(data["E"][dmd_idx][trl], exsum_path)
                if type(tdat) is dict:
                    trial_data[dmd][trl] = tdat
                elif type(tdat) is np.ndarray:
                    if tdat.shape == (2,):
                        print(f"bad trial, zero-array, {dmd}-{trl}")
                        trial_data[dmd][trl] = "BAD_TRIAL"
                else:
                    print(f"unknown trial type, INVESTIGATE: {dmd}-{trl}")
                    trial_data[dmd][trl] = "BAD_TRIAL"
            except Exception as e:
                print(f"ERROR: {e} ---- trial {dmd}-{trl}")
                print("filling in missing_data")
                trial_data[dmd][trl] = "BAD_TRIAL"
                continue
    fs = int(data["params"]["analyzeHz"][0][0])
    trial_epochs = _load_trial_epochs(exsum_path)
    assert len(trial_epochs) == ntrials, (
        f"trialTable.epoch has {len(trial_epochs)} entries but E has {ntrials} trials"
    )
    return trial_data, data, fs, ntrials, trial_epochs


def create_null_trial_data(clean):
    null_data = {}
    for key in clean.keys():
        if type(clean[key]) is np.ndarray:
            null_data[key] = np.zeros_like(clean[key]) * np.nan
        elif type(clean[key]) is dict:
            null_data[key] = {}
            for subkey in clean[key].keys():
                if type(clean[key][subkey]) is np.ndarray:
                    null_data[key][subkey] = np.zeros_like(clean[key][subkey]) * np.nan
                elif type(clean[key][subkey]) is dict:
                    null_data[key][subkey] = {}
                    for subsubkey in clean[key][subkey].keys():
                        if type(clean[key][subkey][subsubkey]) is np.ndarray:
                            null_data[key][subkey][subsubkey] = (
                                np.zeros_like(clean[key][subkey][subsubkey]) * np.nan
                            )
                        else:
                            print(
                                f"unknown subsubkey type, INVESTIGATE: {key}-{subkey}-{subsubkey}"
                            )
    return null_data


# Time axis (within each per-trial array) by top-level key in the SLAP2
# ExSum format. Multi-epoch acqs legitimately vary along this axis — both
# between epochs and (rarely, in the final trial of an epoch) within an
# epoch when the microscope ended a frame early. When
# ``ignore_time_axis_variation=True`` is passed to shape-check helpers,
# comparison drops this axis and catches only genuine corruption
# (wrong channels / synapses / ROIs).
_TIME_AXIS_BY_TOP_KEY: dict[str, int] = {
    "dF": 1,  # shape (channels, time, synapses) for sub-keys ls/denoised/events
    "F0": 1,  # shape (channels, time, synapses)
    "ROIs": 0,  # shape (time, channels, somas); currently skipped anyway
    "noiseEst": 1,  # nested under dF/F0 sub-keys, same layout
}


def _shape_for_compare(arr_shape, top_key, ignore_time_axis):
    if not ignore_time_axis:
        return tuple(arr_shape)
    ax = _TIME_AXIS_BY_TOP_KEY.get(top_key)
    if ax is None:
        return tuple(arr_shape)
    return tuple(s for i, s in enumerate(arr_shape) if i != ax)


def assert_shape_match(clean, comparison_data, ignore_time_axis_variation=False):
    """Assert two trial dicts have matching shapes.

    - ``ignore_time_axis_variation=False`` (single-epoch default): strict
      full-shape equality across every np.ndarray in the trial dict.
    - ``ignore_time_axis_variation=True`` (multi-epoch): only validates
      the non-time dims of keys in ``_TIME_AXIS_BY_TOP_KEY``. Other keys
      (e.g. ``discardFrames``, ``global``, ``footprints``) are skipped
      because their time-axis convention isn't standardized and we don't
      consume them in the scopex builders. This catches genuine
      corruption (wrong channel/synapse/soma counts on the saved traces)
      without tripping on benign time-length variation.
    """
    for key in clean.keys():
        if "ROI" in key:  # TODO: Fix this immedidately!!! Hack!
            continue
        if ignore_time_axis_variation and key not in _TIME_AXIS_BY_TOP_KEY:
            continue
        if type(clean[key]) is np.ndarray:
            cs = _shape_for_compare(
                comparison_data[key].shape, key, ignore_time_axis_variation
            )
            clean_s = _shape_for_compare(
                clean[key].shape, key, ignore_time_axis_variation
            )
            assert cs == clean_s, (
                f"Shape mismatch for {key}: {comparison_data[key].shape} vs {clean[key].shape}"
            )
        elif type(clean[key]) is dict:
            for subkey in clean[key].keys():
                if type(clean[key][subkey]) is np.ndarray:
                    cs = _shape_for_compare(
                        comparison_data[key][subkey].shape,
                        key,
                        ignore_time_axis_variation,
                    )
                    clean_s = _shape_for_compare(
                        clean[key][subkey].shape, key, ignore_time_axis_variation
                    )
                    assert cs == clean_s, (
                        f"Shape mismatch for {key}-{subkey}: {comparison_data[key][subkey].shape} vs {clean[key][subkey].shape}"
                    )
                elif type(clean[key][subkey]) is dict:
                    for subsubkey in clean[key][subkey].keys():
                        if type(clean[key][subkey][subsubkey]) is np.ndarray:
                            cs = _shape_for_compare(
                                comparison_data[key][subkey][subsubkey].shape,
                                key,
                                ignore_time_axis_variation,
                            )
                            clean_s = _shape_for_compare(
                                clean[key][subkey][subsubkey].shape,
                                key,
                                ignore_time_axis_variation,
                            )
                            assert cs == clean_s, (
                                f"Shape mismatch for {key}-{subkey}-{subsubkey}: {comparison_data[key][subkey][subsubkey].shape} vs {clean[key][subkey][subsubkey].shape}"
                            )


def get_clean_trial_dict(trial_data, trial_epochs=None):
    """Pick the first clean trial per DMD (or per (DMD, epoch) if
    ``trial_epochs`` is provided).

    Parameters
    ----------
    trial_data : dict
        ``{dmd: {trial_idx: trial_dict | "BAD_TRIAL"}}``.
    trial_epochs : np.ndarray | None
        If given (1-D int array, length ntrials), return a nested dict
        ``{dmd: {epoch: template}}`` picking the first clean trial per
        ``(dmd, epoch)``. Needed for multi-epoch acqs where per-trial
        shape can differ between epochs.

    Returns
    -------
    dict
        ``{dmd: template}`` when ``trial_epochs is None``;
        ``{dmd: {epoch: template}}`` otherwise.
    """
    if trial_epochs is None:
        clean_trials = {}
        for dmd in trial_data.keys():
            for trl in trial_data[dmd].keys():
                if trial_data[dmd][trl] != "BAD_TRIAL":
                    clean_trials[dmd] = trial_data[dmd][trl]
                    break
        return clean_trials

    clean_trials = {}
    for dmd in trial_data.keys():
        clean_trials[dmd] = {}
        for trl, epoch in zip(
            sorted(trial_data[dmd].keys()), trial_epochs, strict=True
        ):
            epoch_key = int(epoch)
            if epoch_key in clean_trials[dmd]:
                continue
            if trial_data[dmd][trl] != "BAD_TRIAL":
                clean_trials[dmd][epoch_key] = trial_data[dmd][trl]
    return clean_trials


def _lookup_template(clean_trials, dmd, trl, trial_epochs):
    """Resolve the clean-trial template for a given (dmd, trl), supporting
    both flat and per-epoch nested shapes."""
    if trial_epochs is None:
        return clean_trials[dmd]
    epoch_key = int(trial_epochs[trl])
    return clean_trials[dmd][epoch_key]


def replace_bad_trials_with_null_data(trial_data, clean_trials, trial_epochs=None):
    """Replace every ``"BAD_TRIAL"`` entry with a NaN-filled template.

    When ``trial_epochs`` is given, the template is the clean trial from
    the same epoch as the bad trial (since per-trial shape can differ
    between epochs in multi-epoch acqs).
    """
    for dmd in trial_data.keys():
        for trl in trial_data[dmd].keys():
            template = _lookup_template(clean_trials, dmd, trl, trial_epochs)
            if trial_data[dmd][trl] == "BAD_TRIAL":
                trial_data[dmd][trl] = create_null_trial_data(template)
            elif trial_data[dmd][trl].keys() != template.keys():
                print(
                    f"trial {dmd}-{trl} has different keys than clean trial, INVESTIGATE THIS!!"
                )
                trial_data[dmd][trl] = create_null_trial_data(template)
    return trial_data


def check_all_trial_shapes_match(trial_data, clean_trials, trial_epochs=None):
    """Assert every trial's arrays match its epoch's clean template.

    For single-epoch acqs (``trial_epochs is None``): the classic strict
    full-shape check across the whole DMD.

    For multi-epoch acqs: time-axis length variation is allowed (both
    between epochs and occasionally within an epoch when the microscope
    ends a frame early). Only the non-time dims are compared. Genuine
    corruption (wrong channels, synapses, ROIs) still fails.
    """
    ignore_time = trial_epochs is not None
    for dmd in trial_data.keys():
        for trl in trial_data[dmd].keys():
            if trial_data[dmd][trl] == "BAD_TRIAL":
                raise ValueError(
                    f"NO TRIALS SHOULD BE BAD HERE, DMD {dmd}, TRIAL {trl}"
                )
            template = _lookup_template(clean_trials, dmd, trl, trial_epochs)
            assert_shape_match(
                template, trial_data[dmd][trl], ignore_time_axis_variation=ignore_time
            )
    print("all trial shapes match (ignore_time_axis=%s)" % ignore_time)
    return


def get_roi_list(esum_p, dmd):
    rp = f"/exptSummary/userROIs[{dmd - 1}][0]"
    rr = deref_h5_any(rp, esum_p)
    rois = []
    if len(rr.shape) == 1:
        return rois
    for i in range(rr.shape[0]):
        rois.append(rr[i, 0])
    return rois


def get_meanIM(esum_p: str) -> dict:
    # if it is, load the object from the path
    mim = {}
    ref, nt, fs = get_exsum_basics(esum_p)
    for dmd in [1, 2]:
        meanim = deref_h5_any(ref["meanIM"][dmd - 1], esum_p)
        meanim = meanim.swapaxes(1, 2)
        mim[dmd] = meanim
    return mim


def get_actIM(esum_p: str) -> dict[int, np.ndarray]:
    """Load the per-DMD activity image from an ExSum.

    Returns a dict keyed by 1-indexed DMD. Each value is a 2-D array already
    transposed into display orientation (i.e. ready to pass straight to
    ``imshow`` without a further ``.T``).
    """
    aim = {}
    ref, nt, fs = get_exsum_basics(esum_p)
    for dmd in [1, 2]:
        actim = deref_h5_any(ref["actIM"][dmd - 1], esum_p)
        aim[dmd] = np.asarray(actim).T
    return aim


def footprint_to_image(footprint_1d, sel_pix):
    """Map a single source's compressed footprint back to full image coordinates."""
    img = np.full(sel_pix.shape, np.nan)
    img[sel_pix.astype(bool)] = footprint_1d
    return img


def _load_fp_all_with_trial_fallback(
    refdata: dict, dmd: int, trial: int, nt: int, esum_p: str
) -> np.ndarray:
    """Fetch ``E[dmd-1][trial]["footprints"]`` with fallback across trials.

    Some trials are missing/invalid for a given DMD and raise ``IndexError``
    when dereferenced. Try the requested trial first, then wrap through all
    ``nt`` trials; only re-raise once every trial has failed.
    """
    last_err: IndexError | None = None
    for offset in range(nt):
        t = (trial + offset) % nt
        try:
            return deref_h5_any(refdata["E"][dmd - 1][t], esum_p)["footprints"]
        except IndexError as e:
            last_err = e
            if t != trial:
                continue
            print(
                f"[xsum] DMD {dmd}: trial {t} invalid for footprints, "
                f"trying subsequent trials..."
            )
    assert last_err is not None
    raise last_err


def get_fp_info(
    esum_p: str, trial: int = 0, threshold: float = 0.02
) -> tuple[dict, dict]:

    refdata, nt, fs = get_exsum_basics(esum_p)
    synmaps = {}
    fp_vals = {}

    for dmd in [1, 2]:
        sel_pix = deref_h5_any(refdata["selPix"][dmd - 1], esum_p)
        fp_all = _load_fp_all_with_trial_fallback(refdata, dmd, trial, nt, esum_p)
        fpims = []
        for src in range(fp_all.shape[0]):
            fpim = footprint_to_image(fp_all[src], sel_pix)
            fpim[fpim < threshold] = np.nan
            fpim[~np.isnan(fpim)] = int(src + 1)
            fpims.append(fpim.T)
        fp = np.stack(fpims)
        synmap = np.nanmax(fp, axis=0)
        synmap[np.isnan(synmap)] = -1
        synmap = synmap.astype(int)
        fpims = []
        for src in range(fp_all.shape[0]):
            fpim = footprint_to_image(fp_all[src], sel_pix)
            fpim[fpim < threshold] = np.nan
            fpims.append(fpim.T)
        fp = np.stack(fpims)
        fp_values = np.nanmax(fp, axis=0)
        synmaps[dmd] = synmap
        fp_vals[dmd] = fp_values
    return synmaps, fp_vals


def get_raw_fp(esum_p: str, trial: int = 0) -> tuple[dict, dict]:

    refdata, nt, fs = get_exsum_basics(esum_p)
    synmaps = {}
    fp_vals = {}

    for dmd in [1, 2]:
        sel_pix = deref_h5_any(refdata["selPix"][dmd - 1], esum_p)
        fp_all = _load_fp_all_with_trial_fallback(refdata, dmd, trial, nt, esum_p)
        fpims = []
        for src in range(fp_all.shape[0]):
            fpim = footprint_to_image(fp_all[src], sel_pix)
            # fpim[fpim < 0.02] = np.nan
            fpim[~np.isnan(fpim)] = int(src + 1)
            fpims.append(fpim.T)
        fp = np.stack(fpims)
        synmap = np.nanmax(fp, axis=0)
        synmap[np.isnan(synmap)] = -1
        synmap = synmap.astype(int)
        fpims = []
        for src in range(fp_all.shape[0]):
            fpim = footprint_to_image(fp_all[src], sel_pix)
            # fpim[fpim < 0.02] = np.nan
            fpims.append(fpim.T)
        fp = np.stack(fpims)
        fp_values = np.nanmax(fp, axis=0)
        synmaps[dmd] = synmap
        fp_vals[dmd] = fp_values
    return synmaps, fp_vals
