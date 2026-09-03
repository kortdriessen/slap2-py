from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd


def load_datarec_file(path: str):
    datarec = h5py.File(path, "r")
    return datarec


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
    df = pd.DataFrame({
        "start_idx": result.indices[:, 0],
        "end_idx": result.indices[:, 1],
        "start_time_s": result.times_s[:, 0],
        "end_time_s": result.times_s[:, 1],
        "duration_s": result.durations_s,
    })
    return df


# ---------------------------------------------------------------------------
# Acquisition-signature detection on the SLAP2 acquiring-trigger line
# ---------------------------------------------------------------------------
def trigger_duty(
    x: np.ndarray, fs: float = 5000.0, bin_s: float = 1.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-bin fraction of samples above half-scale for a TTL-like trigger line.

    Returns ``(bin_start_idx, duty, half_scale)``. ``half_scale`` is the
    midpoint of robust LOW/HIGH level estimates (0.5th / 99.5th percentiles,
    falling back to min/max when those collapse). A flat signal yields empty
    arrays.
    """
    x = np.asarray(x)
    n_bin = int(round(bin_s * fs))
    n = x.size // n_bin
    lo, hi = np.percentile(x, [0.5, 99.5]) if x.size else (0.0, 0.0)
    if hi <= lo and x.size:
        lo, hi = float(np.min(x)), float(np.max(x))
    half = 0.5 * (lo + hi)
    if n == 0 or hi <= lo:
        return np.array([], dtype=int), np.array([], dtype=float), float(half)
    is_high = (x[: n * n_bin] > half).reshape(n, n_bin)
    return np.arange(n) * n_bin, is_high.mean(axis=1), float(half)


def _refine_step(is_high: np.ndarray, k0: int, w: int, rising: bool) -> int:
    """Sample in ``[k0 - w, k0 + w)`` where a step in ``is_high`` is sharpest.

    Maximises ``mean(is_high[k:k+w]) - mean(is_high[k-w:k])`` (negated for a
    falling step): the matched-filter response of a step of width ``w``, which
    peaks exactly at the step for a clean transition.
    """
    a = max(w, k0 - w)
    b = min(is_high.size - w, k0 + w)
    if b <= a:
        return int(min(max(k0, 0), is_high.size))
    seg = is_high[a - w : b + w].astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(seg)])
    j = np.arange(a, b) - (a - w)
    after = cs[j + w] - cs[j]
    before = cs[j] - cs[j - w]
    resp = (after - before) if rising else (before - after)
    return int(a + np.argmax(resp))


def generate_acq_segment_df(
    x: np.ndarray,
    fs: float = 5000.0,
    bin_s: float = 1.0,
    duty_min: float = 0.9,
    min_duration_s: float = 10.0,
) -> pd.DataFrame:
    """Find *acquisition* segments of the SLAP2 acquiring-trigger line.

    While the microscope is actually acquiring, the trigger line is high for
    ~98-99.5 % of every second (brief dips at the trial structure). During
    preview / live scanning it toggles slowly (duty ~0.2-0.8).
    :func:`generate_scope_index_df`'s hysteresis treats both alike, so a
    preview episode that runs straight into an acquisition is merged into one
    scope-UP window whose start is NOT the acquisition start. This detector
    instead returns runs of ``bin_s`` bins whose duty is >= ``duty_min`` and
    that last >= ``min_duration_s``; each run's start and end are refined to
    the sample with a step matched filter of width ``bin_s``.

    Returns a DataFrame with the same columns as
    :func:`generate_scope_index_df` (``start_idx, end_idx, start_time_s,
    end_time_s, duration_s``); empty if no segment qualifies.
    """
    x = np.asarray(x)
    bin_start, duty, half = trigger_duty(x, fs=fs, bin_s=bin_s)
    n_bin = int(round(bin_s * fs))
    if duty.size == 0:
        return _empty_segment_df()
    high = duty >= duty_min
    edges = np.diff(np.concatenate([[0], high.astype(np.int8), [0]]))
    run_starts = np.flatnonzero(edges == 1)
    run_ends = np.flatnonzero(edges == -1)  # exclusive, in bins
    min_bins = int(np.ceil(min_duration_s / bin_s))
    is_high = x > half
    bouts = []
    for rs, re in zip(run_starts, run_ends, strict=True):
        if re - rs < min_bins:
            continue
        start = _refine_step(is_high, int(rs * n_bin), n_bin, rising=True)
        end = _refine_step(is_high, int(min(re * n_bin, x.size)), n_bin, rising=False)
        end = max(end, start + 1)
        bouts.append((start, end))
    if not bouts:
        return _empty_segment_df()
    b = np.array(bouts, dtype=int)
    return pd.DataFrame({
        "start_idx": b[:, 0],
        "end_idx": b[:, 1],
        "start_time_s": b[:, 0] / fs,
        "end_time_s": b[:, 1] / fs,
        "duration_s": (b[:, 1] - b[:, 0]) / fs,
    })


def _empty_segment_df() -> pd.DataFrame:
    return pd.DataFrame({
        "start_idx": np.array([], dtype=int),
        "end_idx": np.array([], dtype=int),
        "start_time_s": np.array([], dtype=float),
        "end_time_s": np.array([], dtype=float),
        "duration_s": np.array([], dtype=float),
    })
