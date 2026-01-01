import os
from typing import Any, Dict, List

import slap2_py as spy
from slap2_py.hf.core import MatV73Reader


def get_n_trials(esum_path):
    ep = spy.hf.load_any(esum_path, f"/exptSummary/trialTable['epoch']")
    return len(ep)


def load_f0_dmd_trial(esum_path, dmd=1, trial=0):
    fz = spy.hf.load_any(esum_path, f"/exptSummary/E[{dmd - 1}][{trial}]['F0']")
    return fz


def load_f0_full_dmd(esum_path, dmd=1):
    n_trials = get_n_trials(esum_path)
    f0_list = []
    for trial in range(n_trials):
        print(trial)
        fz = load_f0_dmd_trial(esum_path, dmd=dmd, trial=trial)
        f0_list.append(fz)
    return f0_list


def get_trial_shape(esum_path, dmd=1, key="F0"):
    i = 1
    try:
        fz = spy.hf.load_any(esum_path, f"/exptSummary/E[{dmd - 1}][{i}]['{key}']")
    except Exception as e:
        print(f"Error loading trial shape: {e}")
        # increment i and try again
        while True:
            i += 1
            try:
                fz = spy.hf.load_any(
                    esum_path, f"/exptSummary/E[{dmd - 1}][{i}]['{key}']"
                )
                break
            except Exception as e:
                print(f"Error loading trial shape at index {i}: {e}")
                if i > 20:  # arbitrary limit to prevent infinite loop
                    raise ValueError("Could not determine trial shape.")
    return fz.shape


def get_trial_shape_noise(esum_path, dmd=1, group="dF", ttype="matchFilt"):
    i = 1
    try:
        fz = spy.hf.load_any(
            esum_path,
            f"/exptSummary/E[{dmd - 1}][{i}]['noiseEst']['{group}']['{ttype}']",
        )
    except Exception as e:
        print(f"Error loading trial shape: {e}")
        # increment i and try again
        while True:
            i += 1
            try:
                fz = spy.hf.load_any(
                    esum_path,
                    f"/exptSummary/E[{dmd - 1}][{i}]['noiseEst']['{group}']['{ttype}']",
                )
                break
            except Exception as e:
                print(f"Error loading trial shape at index {i}: {e}")
                if i > 20:  # arbitrary limit to prevent infinite loop
                    raise ValueError("Could not determine trial shape.")
    return fz.shape


def load_f0_full_dmd_fast(esum_path: str) -> Dict[int, List[Any]]:
    """
    Faster loader that opens the HDF5 file once and pulls only the 'F0' field
    for each trial for BOTH DMDs in a single pass. This avoids re-opening the
    file per trial or per DMD and minimizes overhead by dereferencing each cell
    and reading only the 'F0' dataset within that struct.

    Returns a dict mapping dmd index (1-based) to a list of F0 arrays ordered
    by trial index, e.g. {1: [...], 2: [...]}.
    """

    f0s: Dict[int, List[Any]] = {}
    with MatV73Reader(esum_path) as r:
        # Access the cell array once; shape is typically (n_dmd, n_trials)
        E = r._f["/exptSummary/E"]
        shape = getattr(E, "shape", None)
        if not shape or len(shape) < 2:
            # Fallback: retain previous slower-but-safe approach
            n_trials = get_n_trials(esum_path)
            for dmd in [1, 2]:
                f0s[dmd] = []
                for trial in range(n_trials):
                    f0 = spy.hf.load_any(
                        esum_path, f"/exptSummary/E[{dmd - 1}][{trial}]['F0']"
                    )
                    f0s[dmd].append(f0)
            return f0s

        n_dmd, n_trials = shape[0], shape[1]
        chosen_dmds = [d for d in (1, 2) if 0 <= (d - 1) < n_dmd]
        for dmd in chosen_dmds:
            f0s[dmd] = [None] * n_trials  # type: ignore[list-item]

        # Directly dereference each cell and read only the 'F0' dataset
        for dmd in chosen_dmds:
            di = dmd - 1
            trial_shape = get_trial_shape(esum_path, dmd=dmd, key="F0")
            for trial in range(n_trials):
                try:
                    ref = E[di, trial]  # type: ignore
                    grp = r._f[ref]
                    # Access only the 'F0' dataset within this struct
                    if "F0" in getattr(grp, "keys", lambda: [])():
                        ds = grp["F0"]  # type: ignore
                        f0 = ds[...]  # type: ignore
                    else:
                        # Rare case: fallback to generic loader
                        f0 = spy.hf.load_any(
                            esum_path, f"/exptSummary/E[{di}][{trial}]['F0']"
                        )
                except Exception:
                    # Conservative fallback on any unexpected shape/structure
                    # here the fallback should just be to append a nan array with the
                    # appropriate shape as a placeholder
                    print(trial, "fallback to NaN array")
                    f0 = np.full(trial_shape, np.nan)
                f0s[dmd][trial] = f0
    return f0s


import numpy as np


def load_noise_estimates(
    esum_path: str, group="dF", ttype="matchFilt"
) -> Dict[int, List[Any]]:
    """ """

    f0s: Dict[int, List[Any]] = {}
    with MatV73Reader(esum_path) as r:
        # Access the cell array once; shape is typically (n_dmd, n_trials)
        E = r._f["/exptSummary/E"]
        shape = getattr(E, "shape", None)
        if not shape or len(shape) < 2:
            return None  # type: ignore

        n_dmd, n_trials = shape[0], shape[1]
        chosen_dmds = [d for d in (1, 2) if 0 <= (d - 1) < n_dmd]
        for dmd in chosen_dmds:
            f0s[dmd] = [None] * n_trials  # type: ignore[list-item]

        # Directly dereference each cell and read only the 'F0' dataset
        for dmd in chosen_dmds:
            di = dmd - 1
            trial_shape = get_trial_shape_noise(
                esum_path, dmd=dmd, group=group, ttype=ttype
            )
            for trial in range(n_trials):
                try:
                    ref = E[di, trial]  # type: ignore
                    grp = r._f[ref]
                    # Access only the 'noiseEst' dataset within this struct
                    if "noiseEst" in getattr(grp, "keys", lambda: [])():
                        ds = grp["noiseEst"][f"{group}"][f"{ttype}"]  # type: ignore
                        f0 = ds[...]  # type: ignore
                    else:
                        # Rare case: fallback to generic loader
                        f0 = spy.hf.load_any(
                            esum_path,
                            f"/exptSummary/E[{di}][{trial}]['noiseEst']['{group}']['{ttype}']",
                        )
                except Exception:
                    # Conservative fallback on any unexpected shape/structure
                    # here the fallback should just be to append a nan array with the
                    # appropriate shape as a placeholder
                    print(trial, "fallback to NaN array")
                    f0 = np.full(trial_shape, np.nan)
                f0s[dmd][trial] = f0
    return f0s


import numpy as np


def load_full_Eset_all_dmds(
    esum_path: str,
    *,
    exclude_fields: List[str] | None = None,
    verbose: bool = False,
):
    """
    Interestingly, loading a single set via something like:
    spy.hf.load_any(esp, f"/exptSummary/E[{dmd - 1}][{trial}]['F0']")

    Is not really faster than just loading the entire set of data for that DMD-trial combination, via:
    spy.hf.load_any(esp, f"/exptSummary/E[{dmd - 1}][{trial}]")

    Therefore, in some cases, it will make sense to have a function where we just return the full dataset at each trial, rather than just pulling F0.
    That is this function.
    """

    result: Dict[int, List[Any]] = {}
    with MatV73Reader(esum_path) as r:
        # Access the cell array once; shape is typically (n_dmd, n_trials)
        try:
            E = r._f["/exptSummary/E"]
        except Exception:
            # Fallback: brute force using load_any if E doesn't exist
            if verbose:
                print("Fallback on E loading")
            n_trials = get_n_trials(esum_path)
            for dmd in [1, 2]:
                result[dmd] = []
                for trial in range(n_trials):
                    try:
                        full = spy.hf.load_any(
                            esum_path, f"/exptSummary/E[{dmd - 1}][{trial}]"
                        )
                    except Exception:
                        full = None
                    result[dmd].append(full)
            return result

        shape = getattr(E, "shape", None)
        if not shape or len(shape) < 2:
            # Fallback to safer approach
            if verbose:
                print("Fallback on shape")
            n_trials = get_n_trials(esum_path)
            for dmd in [1, 2]:
                result[dmd] = []
                for trial in range(n_trials):
                    try:
                        full = spy.hf.load_any(
                            esum_path, f"/exptSummary/E[{dmd - 1}][{trial}]"
                        )
                    except Exception:
                        full = None
                    result[dmd].append(full)
            return result

        n_dmd, n_trials = shape[0], shape[1]
        chosen_dmds = [d for d in (1, 2) if 0 <= (d - 1) < n_dmd]
        for dmd in chosen_dmds:
            result[dmd] = [None] * n_trials  # type: ignore[list-item]

        # Read the entire struct for each (dmd, trial) by dereferencing the cell
        for dmd in chosen_dmds:
            di = dmd - 1
            for trial in range(n_trials):
                try:
                    ref = E[di, trial]  # type: ignore
                    grp = r._f[ref]
                    # Load the full struct into native Python objects, with optional field filtering
                    if hasattr(grp, "keys"):
                        if exclude_fields:
                            full = {}
                            for k in grp.keys():  # type: ignore
                                if k in exclude_fields:
                                    continue
                                full[k] = r._load_obj(grp[k])  # type: ignore[attr-defined]
                        else:
                            full = r._load_obj(grp)  # type: ignore[attr-defined]
                    else:
                        # Unexpected: fallback to meta-loader
                        full = spy.hf.load_any(
                            esum_path, f"/exptSummary/E[{di}][{trial}]"
                        )
                except Exception:
                    # As a conservative fallback, try the meta-loader; else None
                    try:
                        full = spy.hf.load_any(
                            esum_path, f"/exptSummary/E[{di}][{trial}]"
                        )
                    except Exception:
                        full = None
                result[dmd][trial] = full

    return result


def load_f0_full_helper(esum_path: str) -> Dict[int, np.ndarray]:
    """
    Load full F0 data for all DMDs from the given ExSum HDF5 file.

    Returns a dict mapping dmd index (1-based) to a list of F0 arrays ordered
    by trial index, e.g. {1: [...], 2: [...]}.
    """
    fzero = {}
    f0 = load_f0_full_dmd_fast(esum_path)
    fzero[1] = np.transpose(
        np.concatenate(f0[1], axis=1), (0, 2, 1)
    )  # gets us (channel, source, timepoint)
    fzero[2] = np.transpose(
        np.concatenate(f0[2], axis=1), (0, 2, 1)
    )  # gets us (channel, source, timepoint)
    return fzero


def load_single_trace_type(
    esum_path: str, group="dF", ttype="matchFilt"
) -> Dict[int, List[Any]]:
    """ """

    traces: Dict[int, List[Any]] = {}
    with MatV73Reader(esum_path) as r:
        # Access the cell array once; shape is typically (n_dmd, n_trials)
        E = r._f["/exptSummary/E"]
        shape = getattr(E, "shape", None)
        if not shape or len(shape) < 2:
            return None  # type: ignore

        n_dmd, n_trials = shape[0], shape[1]
        chosen_dmds = [d for d in (1, 2) if 0 <= (d - 1) < n_dmd]
        for dmd in chosen_dmds:
            traces[dmd] = [None] * n_trials  # type: ignore[list-item]

        # Directly dereference each cell and read only the 'F0' dataset
        for dmd in chosen_dmds:
            di = dmd - 1
            trial_shape = get_trial_shape(
                esum_path,
                dmd=dmd,
            )
            for trial in range(n_trials):
                try:
                    ref = E[di, trial]  # type: ignore
                    grp = r._f[ref]
                    # Access only the trace group dataset within this struct
                    if group in getattr(grp, "keys", lambda: [])():
                        ds = grp[f"{group}"][f"{ttype}"]  # type: ignore
                        trace = ds[...]  # type: ignore
                    else:
                        # Rare case: fallback to generic loader
                        trace = spy.hf.load_any(
                            esum_path,
                            f"/exptSummary/E[{di}][{trial}]['{group}']['{ttype}']",
                        )
                except Exception:
                    # Conservative fallback on any unexpected shape/structure
                    # here the fallback should just be to append a nan array with the
                    # appropriate shape as a placeholder
                    print(trial, "fallback to NaN array")
                    trace = np.full(trial_shape, np.nan)
                traces[dmd][trial] = trace
    return traces
