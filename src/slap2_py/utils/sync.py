from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import h5py
import pandas as pd

def load_sync_file(path: str):
    full_sync = h5py.File(path, 'r')
    return full_sync['slap2_acquiring_trigger'][:], full_sync['electrophysiology'][:]


@dataclass
class UpPeriods:
    indices: np.ndarray  # shape (K, 2): [start_idx, end_idx) in samples
    times_s: np.ndarray  # shape (K, 2): [start_time_s, end_time_s)
    durations_s: np.ndarray  # shape (K,)
    thresholds: tuple[float, float, float, float]  # (low_est, high_est, T_down, T_up)


def detect_up_periods(
    x: np.ndarray,
    fs: float = 5000.0,
    smooth_ms: float = 1.0,  # 0 to disable
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    up_frac: float = 0.6,  # position of T_up between low/high (0..1)
    down_frac: float = 0.4,  # position of T_down (must be < up_frac)
    min_high_ms: float = 10.0,  # discard UP bouts shorter than this
    max_gap_ms: float = 2.0,  # merge UP bouts separated by <= this gap
) -> UpPeriods:
    """
    Detect UP bouts in a (mostly) bimodal HIGH/LOW sync signal using hysteresis.

    Parameters
    ----------
    x : array_like
        1-D signal.
    fs : float
        Sampling rate in Hz.
    smooth_ms : float
        Moving-average smoothing window in ms (0 disables).
    low_pct, high_pct : float
        Percentiles used to estimate LOW and HIGH levels robustly.
    up_frac, down_frac : float
        Fractions in [0,1] to place the hysteresis thresholds between LOW and HIGH.
        Require down_frac < up_frac.
    min_high_ms : float
        Minimum allowed UP-bout duration (bouts shorter than this are dropped).
    max_gap_ms : float
        If two UP bouts are separated by a DOWN gap shorter than this,
        merge them (debounce).

    Returns
    -------
    UpPeriods
        Detected UP intervals and metadata.
    """
    x = np.asarray(x).astype(float)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    if not (0.0 <= down_frac < up_frac <= 1.0):
        raise ValueError("Require 0 <= down_frac < up_frac <= 1")

    # 1) Optional light smoothing (moving average)
    if smooth_ms and smooth_ms > 0:
        w = int(round(smooth_ms * 1e-3 * fs))
        w = max(1, w)
        # odd length is not required for MA; use simple convolution
        kernel = np.ones(w) / w
        x_f = np.convolve(x, kernel, mode="same")
    else:
        x_f = x

    # 2) Robust level estimates
    low_est, high_est = np.percentile(x_f, [low_pct, high_pct])
    # Handle pathological case where estimates collapse
    if high_est <= low_est:
        # fall back to min/max
        low_est, high_est = float(np.min(x_f)), float(np.max(x_f))
        if high_est == low_est:
            # completely flat signal
            empty = np.empty((0, 2), dtype=int)
            return UpPeriods(
                empty,
                empty.astype(float),
                np.array([]),
                (low_est, high_est, high_est, high_est),
            )

    # 3) Hysteresis thresholds
    T_up = low_est + up_frac * (high_est - low_est)
    T_down = low_est + down_frac * (high_est - low_est)

    # 4) Find crossings
    x0, x1 = x_f[:-1], x_f[1:]
    rises = np.where((x0 < T_up) & (x1 >= T_up))[0] + 1
    falls = np.where((x0 > T_down) & (x1 <= T_down))[0] + 1

    # Pair rises with the first subsequent fall
    bouts = []
    j = 0
    for i in range(len(rises)):
        r = rises[i]
        # advance falls pointer until fall > rise
        while j < len(falls) and falls[j] <= r:
            j += 1
        if j < len(falls):
            f = falls[j]
            bouts.append((r, f))
            j += 1
        else:
            # no fall after the last rise -> treat as open until end
            bouts.append((r, len(x_f)))
            break

    # If the trace starts HIGH, prepend a start at sample 0
    if x_f[0] >= T_up and (len(bouts) == 0 or bouts[0][0] != 0):
        # first fall after 0
        first_fall_idx = falls[falls > 0][0] if np.any(falls > 0) else len(x_f)
        bouts.insert(0, (0, int(first_fall_idx)))

    # 5) Debounce: merge short gaps between bouts
    max_gap = int(round(max_gap_ms * 1e-3 * fs))
    merged = []
    for s, e in bouts:
        if not merged:
            merged.append([s, e])
            continue
        prev_s, prev_e = merged[-1]
        if s - prev_e <= max_gap:
            merged[-1][1] = max(prev_e, e)  # extend
        else:
            merged.append([s, e])

    bouts = np.array(merged, dtype=int)
    if bouts.size == 0:
        empty = np.empty((0, 2), dtype=int)
        return UpPeriods(
            empty, empty.astype(float), np.array([]), (low_est, high_est, T_down, T_up)
        )

    # 6) Enforce minimum duration
    min_len = int(round(min_high_ms * 1e-3 * fs))
    keep = (bouts[:, 1] - bouts[:, 0]) >= min_len
    bouts = bouts[keep]

    # Final packaging
    times = bouts / fs
    durs = (bouts[:, 1] - bouts[:, 0]) / fs
    return UpPeriods(bouts, times, durs, (low_est, high_est, T_down, T_up))

def generate_scope_index_df(
    scope,
    fs=5000,
    smooth_ms=0,  # tiny smoothing
    up_frac=0.1,  # enter UP at 60% between low/high
    down_frac=0.001,  # leave UP at 40%
    min_high_ms=30000,  # ignore micro-bursts
    max_gap_ms=2.0,  # merge brief drops
) -> pd.DataFrame:
    result = detect_up_periods(
        scope,
        fs=fs,
        smooth_ms=smooth_ms,
        up_frac=up_frac,
        down_frac=down_frac,
        min_high_ms=min_high_ms,
        max_gap_ms=max_gap_ms,
    )
    df = pd.DataFrame(
        {
            "start_idx": result.indices[:, 0],
            "end_idx": result.indices[:, 1],
            "start_time_s": result.times_s[:, 0],
            "end_time_s": result.times_s[:, 1],
            "duration_s": result.durations_s,
        }
    )
    return df