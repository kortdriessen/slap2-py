from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mat73
import numpy as np
import polars as pl


def extract_trace_group(
    dat: ExSum, group: str = "dF", trace: str = "matchFilt", extract_chan: int = 2
):
    n_chunks = len(dat.data["E"])
    full_data = {}
    full_data["DMD1"] = []
    full_data["DMD2"] = []
    for dmd in [1, 2]:
        for chunk in range(n_chunks):
            if dat.data["E"][chunk][dmd - 1][group][trace].ndim == 3:
                full_data[f"DMD{dmd}"].append(
                    dat.data["E"][chunk][dmd - 1][group][trace][:, :, extract_chan - 1]
                )
            elif dat.data["E"][chunk][dmd - 1][group][trace].ndim == 2:
                full_data[f"DMD{dmd}"].append(
                    dat.data["E"][chunk][dmd - 1][group][trace]
                )
            else:
                raise ValueError("Unexpected number of dimensions")
        full_data[f"DMD{dmd}"] = np.concatenate(full_data[f"DMD{dmd}"], axis=1)
    return full_data


@dataclass(slots=True)
class ExSum:
    """
    Minimal container for MATLAB->Python analysis results.

    Attributes
    ----------
    data : dict[str, Any]
        The loaded content (either the entire MAT-file mapping or a selected variable).
    source_path : str | None
        Path to the source .mat file, if loaded from disk.
    selected_var : str | None
        Name of the selected top-level variable (if you picked one).
    """

    data: dict[str, Any]
    source_path: str | None = None
    selected_var: str | None = None

    @property
    def fs(self) -> float:
        # Sampling rate (Hz) from MATLAB params
        return float(self.data["params"]["analyzeHz"])

    # --- Primary loader (alternative constructor) ---
    @classmethod
    def from_mat73(cls, path: str, var: str | None = None) -> ExSum:
        """
        Load a v7.3 MAT-file via mat73 and return a ExSum instance.

        Parameters
        ----------
        path : str
            Path to the .mat file.
        var : str | None
            If provided, extract this top-level variable from the file.
            If omitted and there is exactly one top-level variable, that is used.
            Otherwise, the entire mapping is stored in .data.

        Returns
        -------
        ExSum
        """
        raw = mat73.loadmat(path)  # -> dict[str, Any]

        if var is not None:
            if var not in raw:
                raise KeyError(
                    f"Variable '{var}' not found. Top-level keys: {list(raw.keys())}"
                )
            return cls(data=raw[var], source_path=path, selected_var=var)

        # If there is exactly one top-level variable, unwrap it for convenience
        if isinstance(raw, Mapping) and len(raw) == 1:
            ((only_key, only_val),) = raw.items()
            return cls(data=only_val, source_path=path, selected_var=only_key)

        # Otherwise keep the whole mapping
        return cls(data=raw, source_path=path, selected_var=None)

    # --- Tiny toy example method ---
    def mean_im(self, DMD: int, channel: int) -> np.ndarray:
        """
        Return the number of top-level entries inside .data.
        For a 'picked' variable that is itself a dict, this counts its keys.
        For arrays/scalars, this returns 1.
        """
        return self.data["meanIM"][DMD - 1][:, :, channel - 1]

    def maxfp(self, DMD: int, chunk: int = 0) -> np.ndarray:
        """
        Return the maximum footprints for a given DMD and chunk.

        """
        return np.max(self.data["E"][chunk][DMD - 1]["footprints"], axis=2)

    def extract_soma_activity(self, dmd: int, version: str = "Fsvd") -> pl.DataFrame:
        DMD = dmd - 1
        edat = self.data["E"]
        roi_info = self.data["userROIs"]
        num_rois = edat[0][DMD]["ROIs"][version].shape[0]
        roi_names = [roi_info[DMD][i]["Label"] for i in range(num_rois)]
        fs = self.data["params"]["analyzeHz"]
        roi_dfs = []
        for i, name in enumerate(roi_names):
            dat_ch1 = []
            dat_ch2 = []
            for trl in np.arange(len(edat)):
                roi_dat1 = edat[trl][DMD]["ROIs"][version][i, :, 0]
                roi_dat2 = edat[trl][DMD]["ROIs"][version][i, :, 1]
                dat_ch1.append(roi_dat1)
                dat_ch2.append(roi_dat2)
            rd1 = np.concatenate(dat_ch1)
            rd2 = np.concatenate(dat_ch2)
            time = np.arange(len(rd1)) / fs
            # note that channel 1 and channel 2 are flipped here,
            # so we need to swap them (I think if
            # activityChannel is set to 2, then it gets assigned as 1 here)
            roi_df = pl.DataFrame(
                {
                    "time": time,
                    "roi_ch1": rd2,
                    "roi_ch2": rd1,
                    "roi_name": name,
                    "DMD": DMD,
                }
            )
            roi_dfs.append(roi_df)
        return pl.concat(roi_dfs)

    def gen_roidf(self, version: str = "Fsvd") -> pl.DataFrame:
        """
        Generate a dataframe of user-defined ROI across both DMDs.

        Parameters
        ----------
        version : str, optional
            F or Fsvd, by default "Fsvd"

        Returns
        -------
        pl.DataFrame
            A dataframe of user-defined ROI across both DMDs.
        """
        # for each dmd, first check if there is userROI info
        dmd_dfs = []
        for dmd in [1, 2]:
            if len(self.data["userROIs"][dmd - 1]) == 0:
                continue
            else:
                dmd_df = self.extract_soma_activity(dmd, version)
                dmd_dfs.append(dmd_df)
        return pl.concat(dmd_dfs)

    def gen_ls_df(self, trace_group: str = "dF"):
        ls_dfs = []
        n_trials = len(self.data["E"])
        for dmd in [1, 2]:
            n_sources = self.data["E"][0][dmd - 1]["F0"].shape[0]
            for src in range(n_sources):
                traces_ch1 = []
                traces_ch2 = []
                for trl in range(n_trials):
                    traces_ch1.append(
                        self.data["E"][trl][dmd - 1][trace_group]["ls"][src, :, 0]
                    )
                    traces_ch2.append(
                        self.data["E"][trl][dmd - 1][trace_group]["ls"][src, :, 1]
                    )
                traces_ch1 = np.concatenate(traces_ch1)
                traces_ch2 = np.concatenate(traces_ch2)
                traces_t = np.arange(len(traces_ch1)) / self.fs
                ls_df1 = pl.DataFrame(
                    {
                        "time": traces_t,
                        "data": traces_ch1,
                        "dmd": dmd,
                        "source": src,
                        "trace_group": trace_group,
                        "trace_type": "ls",
                        "channel": 1,
                    }
                )

                ls_df2 = pl.DataFrame(
                    {
                        "time": traces_t,
                        "data": traces_ch2,
                        "dmd": dmd,
                        "source": src,
                        "trace_group": trace_group,
                        "trace_type": "ls",
                        "channel": 2,
                    }
                )
                ls_dfs.append(ls_df1)
                ls_dfs.append(ls_df2)
        return pl.concat(ls_dfs)

    def gen_syndf(
        self, trace_group: str = "dF", to_pull: list[str] | None = None
    ) -> pl.DataFrame:
        """Generate a dataframe containing the traces across all trials of all sources
        on both DMDs, for any specified group of trace types.

        Parameters
        ----------
        trace_group : str, optional
            should be one of "dF" or "dFF", by default "dF"
        to_pull : list[str] | None, optional
            list of trace types to pull, should any combination of
            "ls", "matchFilt", "denoised", "events", "nonneg", by default
            None which pulls: ["matchFilt", "denoised", "events"]

        Returns
        -------
        pl.DataFrame
            A dataframe containing the traces across all trials of all sources
            on both DMDs, for any specified group of trace types.
        """
        if to_pull is None:
            to_pull = ["matchFilt", "denoised", "events"]
        fs = self.data["params"]["analyzeHz"]
        syndfs = []
        n_trials = len(self.data["E"])
        for dmd in [1, 2]:
            n_sources = self.data["E"][0][dmd - 1]["F0"].shape[0]
            for trace_type in to_pull:
                if trace_type == "ls":
                    lsdf = self.gen_ls_df(trace_group)
                    syndfs.append(lsdf)
                    continue
                for src in range(n_sources):
                    traces_list = []
                    for trl in range(n_trials):
                        traces = self.data["E"][trl][dmd - 1][trace_group][trace_type][
                            src, :
                        ]
                        traces_list.append(traces)
                    traces_full = np.concatenate(traces_list)
                    traces_full = traces_full.flatten()
                    traces_t = np.arange(len(traces_full)) / fs
                    syndf = pl.DataFrame(
                        {
                            "time": traces_t,
                            "data": traces_full,
                            "dmd": dmd,
                            "source": src,
                            "trace_group": trace_group,
                            "trace_type": trace_type,
                            "channel": 2,
                        }
                    )
                    syndfs.append(syndf)
        return pl.concat(syndfs)


# ---------------------- Utility Functions ----------------------
