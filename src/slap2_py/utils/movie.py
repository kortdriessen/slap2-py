# ===============================================
# This module contains functions for movie generation
# using outputs from the ophys-slap2-analysis
# preprocessing and localization pipelines.
# ===============================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from typing import Sequence, Optional, Tuple, Dict, List


def make_synapse_activity_video(
    series_list: Sequence[np.ndarray],
    timestamps: np.ndarray,
    synapse_ids: Sequence[int],
    synapse_map: np.ndarray,  # HxW int map; pixels==ID belong to that synapse; -1 otherwise
    base_image: np.ndarray,  # HxW mean image (float or uint); NaNs allowed
    out_path: str = "syn_activity.mp4",
    *,
    # Optional soma layer
    soma_timeseries: Optional[Sequence[np.ndarray]] = None,
    soma_map: Optional[np.ndarray] = None,
    soma_ids: Optional[
        Sequence[int]
    ] = None,  # if None, we infer from soma_map (sorted unique)
    # Optional line panel
    line_series: Optional[np.ndarray] = None,
    line_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    line_height_ratio: float = 0.28,
    # Playback / encoding
    playback_speed: float = 1.0,  # 1.0 = real time (median dt)
    fps: Optional[int] = None,  # if None, computed from timestamps & playback_speed
    max_fps: int = 60,
    dpi: int = 150,
    codec: str = "libx264",
    crf: int = 18,  # lower = higher quality
    bitrate: Optional[str] = None,  # e.g. "6M" to override CRF
    # Appearance
    syn_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),  # bright green
    soma_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),  # bright red
    alpha_max_syn: float = 1,
    alpha_max_soma: float = 0.9,
    center_value: float = 1.0,  # white/noise level of traces
    syn_q_hi: float = 0.98,  # robust high quantile for syn scaling
    soma_q_hi: float = 0.95,  # robust high quantile for soma scaling
    syn_gamma: float = 1.0,  # gamma on normalized activity
    soma_gamma: float = 1.0,
    base_q: Tuple[float, float] = (
        0.01,
        0.99,
    ),  # grayscale robust window for base image
    base_alpha: float = 0.9,
    figsize: Optional[Tuple[float, float]] = None,  # if None, chosen from image size
    progress: bool = True,
) -> Dict[str, object]:
    """
    Create a video that animates synapse (and optional soma) activity directly on top of a
    grayscale mean image, with an optional synchronized line plot above.

    One frame per sample by default (≈ real-time from timestamps). Activities > center_value
    (default 1.0) are mapped to overlay opacity; per-ID shapes come from the label maps.

    Returns a dict with metadata: {'fps', 'frames', 'out_path', 'vmax_syn', 'vmax_soma', ...}
    """
    # --------- Validate inputs ---------
    if len(series_list) == 0:
        raise ValueError("series_list is empty.")
    T = len(timestamps)
    if any(arr.shape != (T,) for arr in series_list):
        bad = [i for i, a in enumerate(series_list) if a.shape != (T,)]
        raise ValueError(f"All series must have shape ({T},). Offenders: {bad}")
    if line_series is not None and line_series.shape != (T,):
        raise ValueError(f"line_series must have shape ({T},).")
    if synapse_map.shape != base_image.shape:
        raise ValueError("synapse_map and base_image must have identical HxW.")
    H, W = base_image.shape
    if len(synapse_ids) != len(series_list):
        raise ValueError("synapse_ids length must equal len(series_list).")

    have_soma = soma_timeseries is not None and soma_map is not None
    if have_soma:
        if any(s.shape != (T,) for s in soma_timeseries):
            bad = [i for i, s in enumerate(soma_timeseries) if s.shape != (T,)]
            raise ValueError(
                f"All soma_timeseries must have shape ({T},). Offenders: {bad}"
            )
        if soma_map.shape != (H, W):
            raise ValueError("soma_map must match base_image shape.")
        if soma_ids is None:
            # Infer IDs from map
            uniq = np.unique(soma_map)
            uniq = uniq[uniq >= 0]
            uniq.sort()
            if len(uniq) != len(soma_timeseries):
                raise ValueError(
                    "soma_ids not provided and inferred unique IDs in soma_map "
                    "do not match len(soma_timeseries). Pass soma_ids explicitly."
                )
            soma_ids = list(map(int, uniq))
        elif len(soma_ids) != len(soma_timeseries):
            raise ValueError("soma_ids length must equal len(soma_timeseries).")

    # --------- Timebase & FPS ---------
    t = np.asarray(timestamps, dtype=float)
    if np.any(~np.isfinite(t)):
        raise ValueError("timestamps contain non-finite values.")
    if np.any(np.diff(t) <= 0):
        # Allow equal spacing jitter-fix, forbid decreasing
        diffs = np.diff(t)
        if np.any(diffs < 0):
            raise ValueError("timestamps must be non-decreasing.")
        # jitter exact duplicates
        eps = np.finfo(float).eps
        for k in np.where(diffs == 0)[0]:
            t[k + 1] = t[k + 1] + eps

    med_dt = float(np.median(np.diff(t)))
    if med_dt <= 0:
        med_dt = 1.0  # fallback
    if fps is None:
        fps = int(np.clip(np.round((1.0 / med_dt) * playback_speed), 1, max_fps))
    frames = T

    # --------- Base image prep (grayscale on black) ---------
    base = np.asarray(base_image, dtype=float)
    base[np.isnan(base)] = np.nan  # keep NaNs for scaling then set to black
    # Robust scaling
    finite_vals = base[np.isfinite(base)]
    if finite_vals.size == 0:
        raise ValueError("base_image has no finite pixels.")
    vmin = np.quantile(finite_vals, base_q[0])
    vmax = np.quantile(finite_vals, base_q[1])
    if vmax <= vmin:
        vmax = vmin + 1.0
    base_norm = np.clip((base - vmin) / (vmax - vmin), 0.0, 1.0)
    base_norm[~np.isfinite(base_norm)] = 0.0  # NaNs/inf to black
    base_rgb = np.stack([base_norm] * 3, axis=-1)  # grayscale 3-ch

    # --------- Build ID -> index maps for synapses (and somas) ---------
    syn_id_to_idx = {int(i): j for j, i in enumerate(synapse_ids)}
    syn_idx_map = np.full((H, W), -1, dtype=np.int32)
    # Fill in a vectorized pass by IDs (one-time)
    for sid, j in syn_id_to_idx.items():
        syn_idx_map[synapse_map == sid] = j

    have_soma_idx = False
    if have_soma:
        soma_id_to_idx = {int(i): j for j, i in enumerate(soma_ids)}
        soma_idx_map = np.full((H, W), -1, dtype=np.int32)
        for sid, j in soma_id_to_idx.items():
            soma_idx_map[soma_map == sid] = j
        have_soma_idx = True

    # --------- Precompute activity -> opacity mapping ---------
    def series_to_alpha(
        stack: np.ndarray, q_hi: float, alpha_max: float, gamma: float
    ) -> Tuple[np.ndarray, float]:
        """
        Map (N,T) stack of series to normalized alphas in [0, alpha_max], zeroed for <= center_value.
        Robust upper radius from quantile of (values > center_value).
        """
        above = stack[stack > center_value]
        if above.size == 0:
            radius = 1.0
        else:
            radius = float(np.quantile(above, q_hi) - center_value)
            if radius <= 1e-9:
                radius = 1.0
        # Normalize, floor at 0
        norm = np.clip((stack - center_value) / radius, 0.0, 1.0)
        if gamma != 1.0:
            norm = norm**gamma
        return (norm * alpha_max), (center_value + radius)

    syn_stack = np.vstack(
        [np.asarray(a, dtype=float) for a in series_list]
    )  # (Nsyn, T)
    syn_alpha, syn_vmax = series_to_alpha(syn_stack, syn_q_hi, alpha_max_syn, syn_gamma)

    if have_soma:
        soma_stack = np.vstack(
            [np.asarray(a, dtype=float) for a in soma_timeseries]
        )  # (Nsoma, T)
        soma_alpha, soma_vmax = series_to_alpha(
            soma_stack, soma_q_hi, alpha_max_soma, soma_gamma
        )
    else:
        soma_alpha, soma_vmax = None, None

    # --------- Figure / axes ---------
    if figsize is None:
        # Heuristic: ~ 6 inches width, height from aspect + line panel
        base_height = max(3.5, 6.0 * (H / max(W, 1)))
        figsize = (
            6.0,
            base_height * (1.0 + line_height_ratio if line_series is not None else 1.0),
        )

    fig = plt.figure(figsize=figsize, facecolor="black")
    import matplotlib.gridspec as gridspec

    if line_series is not None:
        gs = gridspec.GridSpec(
            nrows=2, ncols=1, height_ratios=[line_height_ratio, 1.0], hspace=0.02
        )
        ax_line = fig.add_subplot(gs[0])
        ax_img = fig.add_subplot(gs[1])
        # Style the line axis (black bg; minimal ink)
        ax_line.set_facecolor("black")
        ax_line.spines["top"].set_visible(False)
        ax_line.spines["right"].set_visible(False)
        ax_line.spines["bottom"].set_color("white")
        ax_line.spines["left"].set_color("white")
        ax_line.tick_params(colors="white", labelsize=8)
        # Draw empty line; we will update data each frame
        (ln,) = ax_line.plot([], [], color=line_color, lw=1.25)
        ax_line.set_xlim(t[0], t[-1])
        # Y limits from robust range on line_series
        finite_ls = line_series[np.isfinite(line_series)]
        if finite_ls.size == 0:
            ymin, ymax = 0.0, 1.0
        else:
            ymin = np.quantile(finite_ls, 0.01)
            ymax = np.quantile(finite_ls, 0.99)
            if ymax <= ymin:
                ymax = ymin + 1.0
        pad = 0.05 * (ymax - ymin)
        ax_line.set_ylim(ymin - pad, ymax + pad)
        # Hide x tick labels to avoid duplication; the image axis carries time labels
        plt.setp(ax_line.get_xticklabels(), visible=False)
    else:
        ax_line = None
        ax_img = fig.add_subplot(111)

    # Image axis styling
    ax_img.set_facecolor("black")
    ax_img.axis("off")

    # Initial composite (frame 0)
    disp = ax_img.imshow(
        base_rgb, interpolation="nearest", origin="upper", alpha=base_alpha
    )
    ax_img.set_xlim(-0.5, W - 0.5)
    ax_img.set_ylim(H - 0.5, -0.5)  # keep array orientation

    # --------- Writer setup ---------
    metadata = dict(title="Synapse Activity", artist="make_synapse_activity_video")
    writer = FFMpegWriter(fps=fps, metadata=metadata, codec=codec)
    extra_args = []
    if bitrate is not None:
        extra_args += ["-b:v", str(bitrate)]
    else:
        extra_args += ["-crf", str(crf)]
    # Ensure black background in the video canvas
    fig.patch.set_facecolor("black")

    # --------- Frame loop ---------
    with writer.saving(fig, out_path, dpi):
        for k in range(frames):
            # Build alpha images by indexing with ID->index maps (vectorized)
            # Synapses
            syn_alpha_k = np.zeros((H, W), dtype=float)
            mask = syn_idx_map >= 0
            if mask.any():
                syn_alpha_k[mask] = syn_alpha[syn_idx_map[mask], k]

            # Somas
            if have_soma_idx:
                soma_alpha_k = np.zeros((H, W), dtype=float)
                mask_s = soma_idx_map >= 0
                if mask_s.any():
                    soma_alpha_k[mask_s] = soma_alpha[soma_idx_map[mask_s], k]
            else:
                soma_alpha_k = None

            # Composite overlays on base (alpha blend to pure color)
            frame_rgb = base_rgb.copy()
            if np.any(syn_alpha_k > 0):
                a = syn_alpha_k[..., None]
                frame_rgb = (
                    frame_rgb * (1.0 - a) + np.array(syn_color)[None, None, :] * a
                )
            if soma_alpha_k is not None and np.any(soma_alpha_k > 0):
                a = soma_alpha_k[..., None]
                frame_rgb = (
                    frame_rgb * (1.0 - a) + np.array(soma_color)[None, None, :] * a
                )

            # Update image
            disp.set_data(frame_rgb)

            # Update line panel, if present
            if ax_line is not None:
                ln.set_data(t[: k + 1], line_series[: k + 1])

            writer.grab_frame()
            if progress and (k % max(1, frames // 100) == 0):
                # lightweight textual progress
                print(f"\rFrame {k+1}/{frames}", end="")

    if progress:
        print("\nDone:", out_path)

    return {
        "out_path": out_path,
        "fps": fps,
        "frames": frames,
        "syn_vmax_used": syn_vmax,
        "soma_vmax_used": soma_vmax,
        "base_vmin": vmin,
        "base_vmax": vmax,
    }
