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


def _h5_to_py(obj: h5py.Group | h5py.Dataset) -> Any:
    """
    Recursively convert an h5py Group/Dataset into Python structures.

    - Groups → dict of {name: value}
    - Datasets → numpy arrays / scalars / strings
    """
    if isinstance(obj, h5py.Group):
        out = {}
        for key, item in obj.items():
            out[key] = _h5_to_py(item)
        return out

    elif isinstance(obj, h5py.Dataset):
        data = obj[()]  # read the whole dataset

        # MATLAB cell arrays / objects often show up as object arrays or references;
        # here we only handle straightforward numeric / char data cleanly.
        if isinstance(data, np.ndarray):
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


def _h5_to_py_simple(obj: Any) -> Any:
    """Minimal recursive converter: Dataset -> array, Group -> dict."""
    if isinstance(obj, h5py.Dataset):
        return obj[()]  # full array
    if isinstance(obj, h5py.Group):
        return {k: _h5_to_py_simple(v) for k, v in obj.items()}
    return obj


def deref_h5_any(ref: Any, file_path: str | Path) -> Any:
    """
    Given something like data['E'][0][0] (which may be a tuple/ndarray
    containing an HDF5 object reference), open the file and return the
    actual data (array or nested dict).
    """
    file_path = str(file_path)

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

        # 3. Convert to Python structures
        return _h5_to_py_simple(obj)


def get_exsum_basics(exsum_path: Path | str) -> tuple[dict, int, int]:
    data = load_mat73_to_dict(exsum_path, root="exptSummary")
    ntrials = int(data["E"].shape[1])
    fs = int(data["params"]["analyzeHz"][0][0])
    return data, ntrials, fs


def read_full_trial_data_dict(exsum_path):
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
    return trial_data, data, fs, ntrials


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


def assert_shape_match(clean, comparison_data):
    for key in clean.keys():
        if "ROI" in key:  # TODO: Fix this immedidately!!! Hack!
            continue
        if type(clean[key]) is np.ndarray:
            assert comparison_data[key].shape == clean[key].shape, (
                f"Shape mismatch for {key}: {comparison_data[key].shape} vs {clean[key].shape}"
            )
        elif type(clean[key]) is dict:
            for subkey in clean[key].keys():
                if type(clean[key][subkey]) is np.ndarray:
                    assert (
                        comparison_data[key][subkey].shape == clean[key][subkey].shape
                    ), (
                        f"Shape mismatch for {key}-{subkey}: {comparison_data[key][subkey].shape} vs {clean[key][subkey].shape}"
                    )
                elif type(clean[key][subkey]) is dict:
                    for subsubkey in clean[key][subkey].keys():
                        if type(clean[key][subkey][subsubkey]) is np.ndarray:
                            assert (
                                comparison_data[key][subkey][subsubkey].shape
                                == clean[key][subkey][subsubkey].shape
                            ), (
                                f"Shape mismatch for {key}-{subkey}-{subsubkey}: {comparison_data[key][subkey][subsubkey].shape} vs {clean[key][subkey][subsubkey].shape}"
                            )


def get_clean_trial_dict(trial_data):
    clean_trials = {}
    for dmd in trial_data.keys():
        for trl in trial_data[dmd].keys():
            if trial_data[dmd][trl] != "BAD_TRIAL":
                clean_trials[dmd] = trial_data[dmd][trl]
                break
    return clean_trials


def replace_bad_trials_with_null_data(trial_data, clean_trials):
    for dmd in trial_data.keys():
        for trl in trial_data[dmd].keys():
            if trial_data[dmd][trl] == "BAD_TRIAL":
                trial_data[dmd][trl] = create_null_trial_data(clean_trials[dmd])
    return trial_data


def check_all_trial_shapes_match(trial_data, clean_trials):
    for dmd in trial_data.keys():
        for trl in trial_data[dmd].keys():
            if trial_data[dmd][trl] == "BAD_TRIAL":
                raise ValueError(
                    f"NO TRIALS SHOULD BE BAD HERE, DMD {dmd}, TRIAL {trl}"
                )
            else:
                assert_shape_match(clean_trials[dmd], trial_data[dmd][trl])
    print("all trial shapes match")
    return
