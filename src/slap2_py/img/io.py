import tifffile
def load_tiff(path):
    """Loads tiff file using tifffile

    Parameters
    ----------
    path : str
        Path to the TIFF file.

    Returns
    -------
    np.ndarray
        The loaded image data.
    """
    return tifffile.imread(path)