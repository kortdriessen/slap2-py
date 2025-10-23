import matplotlib.pyplot as plt
import numpy as np

import slap2_py as spy


def single_synapse_id_plot(d, dmd, channel, source_number, chunk=0, buffer=25):
    """
    Plots a single synapse ID plot.

    Parameters
    ----------
    d : slap2_py.ExSum
        ExSum object
    dmd : int
        DMD number
    channel : int
        Channel number
    source_number : int
        Source number
    chunk : int, optional
        Chunk number, by default 0
    buffer : int, optional
        Buffer around the source, by default 25

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    """
    # get the mean image
    mean_im = d.mean_im(dmd, channel)
    # find a box for inset plot
    r0, c0, h, w = spy.img.utils.find_empty_rectangle(mean_im)

    # get the source's footprint overlay
    fp = d.data["E"][chunk][dmd - 1]["footprints"][:, :, source_number]
    non_nan_indices = np.where(fp > 0)
    ymin, ymax = np.min(non_nan_indices[0]), np.max(non_nan_indices[0])
    xmin, xmax = np.min(non_nan_indices[1]), np.max(non_nan_indices[1])
    xmin = xmin - buffer
    xmax = xmax + buffer
    ymin = ymin - buffer
    ymax = ymax + buffer
    mag = mean_im[ymin:ymax, xmin:xmax]
    fp_only = fp[ymin:ymax, xmin:xmax]
    fp_only[fp_only == 0] = np.nan

    # plot the mean image
    fig_height = mean_im.shape[0] / 40
    fig_width = mean_im.shape[1] / 40
    p5, p98 = np.nanpercentile(mag, 5), np.nanpercentile(mag, 98)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.imshow(mean_im, origin="lower", cmap="viridis", vmin=p5, vmax=p98)
    ax.add_patch(
        plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )

    # plot the source's ID plot in the inset

    r0, c0, h, w = spy.img.utils.find_empty_rectangle(mean_im)
    axins = ax.inset_axes([c0, r0, w, h], transform=ax.transData, zorder=10)

    axins.imshow(mag, origin="lower", cmap="viridis", aspect="equal", vmin=p5, vmax=p98)
    axins.imshow(fp_only, origin="lower", cmap="Reds_r", alpha=0.7, aspect="equal")

    # tidy up the inset appearance
    axins.set_xticks([])
    axins.set_yticks([])
    for s in axins.spines.values():
        s.set_edgecolor("red")
        s.set_linewidth(2)

    ax.set_title(f"DMD-{dmd} | Channel-{channel} | Source-{source_number}")
    return fig, ax


def synapse_id_plot(
    mean_im,
    fps,
    dmd,
    source_number,
    upper_vmax_pct=98,
    buffer=25,
    channel=2,
    subject=None,
    exp=None,
    loc=None,
    acq=None,
):
    """
    Plots a single synapse ID plot.

    Parameters
    ----------
    d : slap2_py.ExSum
        ExSum object
    dmd : int
        DMD number
    channel : int
        Channel number
    source_number : int
        Source number
    chunk : int, optional
        Chunk number, by default 0
    buffer : int, optional
        Buffer around the source, by default 25

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    """

    # get the mean image
    mean_im = mean_im[dmd][channel - 1, :, :]
    r0, c0, h, w = spy.img.utils.find_empty_rectangle(mean_im)

    # get the source's footprint overlay
    fp = fps[dmd][source_number, :, :]
    non_nan_indices = np.where(fp > 0)
    ymin, ymax = np.min(non_nan_indices[0]), np.max(non_nan_indices[0])
    xmin, xmax = np.min(non_nan_indices[1]), np.max(non_nan_indices[1])
    xmin = xmin - buffer
    xmax = xmax + buffer
    ymin = ymin - buffer
    ymax = ymax + buffer
    fp_display = fp.copy()
    fp_mask = fp > 0
    fp_display[~fp_mask] = np.nan
    mag = mean_im[ymin:ymax, xmin:xmax]
    fp_only = fp[ymin:ymax, xmin:xmax]
    fp_only[fp_only == 0] = np.nan

    # plot the mean image
    fig_height = mean_im.shape[0] / 60
    fig_width = mean_im.shape[1] / 60
    print(fig_height, fig_width)
    p5, p_upper = np.nanpercentile(mag, 5), np.nanpercentile(mag, upper_vmax_pct)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.imshow(mean_im, origin="upper", cmap="viridis", vmin=p5, vmax=p_upper)
    ax.imshow(fp_display, origin="upper", cmap="Reds_r", alpha=0.7)
    ax.add_patch(
        plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )

    # plot the source's ID plot in the inset
    r0, c0, h, w = spy.img.utils.find_empty_rectangle(mean_im)
    axins = ax.inset_axes([c0, r0, w, h], transform=ax.transData, zorder=10)

    axins.imshow(
        mag, origin="upper", cmap="viridis", aspect="equal", vmin=p5, vmax=p_upper
    )
    axins.imshow(fp_only, origin="upper", cmap="Reds_r", alpha=0.7, aspect="equal")

    # tidy up the inset appearance
    axins.set_xticks([])
    axins.set_yticks([])
    for s in axins.spines.values():
        s.set_edgecolor("red")
        s.set_linewidth(2)

    if subject is not None and exp is not None and loc is not None and acq is not None:
        ax.set_title(
            f"{subject} | {exp} | {loc}--{acq} | DMD-{dmd} | Source-{source_number}"
        )
    else:
        ax.set_title(f"DMD-{dmd} | Channel-{channel} | Source-{source_number}")

    return fig, ax
