from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional
import mat73
import numpy as np
import polars as pl
import xarray as xr

def extract_trace_group(dat: SlapData, group: str = 'dF', trace: str = 'matchFilt', extract_chan: int = 2):
    n_chunks = len(dat.data['E'])
    full_data = {}
    full_data['DMD1'] = []
    full_data['DMD2'] = []
    for dmd in [1, 2]:
        for chunk in range(n_chunks):
            if dat.data['E'][chunk][dmd-1][group][trace].ndim == 3:
                full_data[f'DMD{dmd}'].append(dat.data['E'][chunk][dmd-1][group][trace][:, :, extract_chan-1])
            elif dat.data['E'][chunk][dmd-1][group][trace].ndim == 2:
                full_data[f'DMD{dmd}'].append(dat.data['E'][chunk][dmd-1][group][trace])
            else:
                raise ValueError('Unexpected number of dimensions')
        full_data[f'DMD{dmd}'] = np.concatenate(full_data[f'DMD{dmd}'], axis=1)
    return full_data

@dataclass(slots=True)
class SlapData:
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
    source_path: Optional[str] = None
    selected_var: Optional[str] = None

    @property
    def fs(self) -> float:
        # Sampling rate (Hz) from MATLAB params
        return float(self.data['params']['analyzeHz'])

    # --- Primary loader (alternative constructor) ---
    @classmethod
    def from_mat73(cls, path: str, var: Optional[str] = None) -> "SlapData":
        """
        Load a v7.3 MAT-file via mat73 and return a SlapData instance.

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
        SlapData
        """
        raw = mat73.loadmat(path)  # -> dict[str, Any]

        if var is not None:
            if var not in raw:
                raise KeyError(f"Variable '{var}' not found. Top-level keys: {list(raw.keys())}")
            return cls(data=raw[var], source_path=path, selected_var=var)

        # If there is exactly one top-level variable, unwrap it for convenience
        if isinstance(raw, Mapping) and len(raw) == 1:
            (only_key, only_val), = raw.items()
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
        return self.data['meanIM'][DMD-1][:, :, channel-1]
    
    
    def maxfp(self, DMD: int, chunk: int = 0) -> np.ndarray:
        """
        Return the maximum footprints for a given DMD and chunk.

        """
        return np.max(self.data['E'][chunk][DMD-1]['footprints'], axis=2)
    
    def to_syndf(self) -> pl.DataFrame:
        ""
        return None