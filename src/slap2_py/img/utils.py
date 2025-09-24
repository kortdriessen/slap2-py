import numpy as np


def find_empty_rectangle(img, *, aspect=None, aspect_tol=0.35, min_h=1, min_w=1):
    """
    Find the largest all-NaN axis-aligned rectangle in an image. Useful for finding a
    large space for inset plots of mean_im's.

    Parameters
    ----------
    img : 2D np.ndarray
        Your image array; NaNs are considered empty/available.
    aspect : float or None
        Desired width/height ratio to bias toward (None = any).
    aspect_tol : float
        Allowed relative deviation from `aspect` (e.g., 0.35 = ±35%).
    min_h, min_w : int
        Minimum rectangle height/width (pixels).

    Returns
    -------
    r0, c0, h, w : ints
        Top-left row/col and height/width in pixel coordinates.
        Returns (None, None, 0, 0) if nothing found.
    """
    H, W = img.shape
    empty = np.isnan(img)

    heights = np.zeros(W, dtype=int)
    best = (None, None, 0, 0, 0.0)  # r0, c0, h, w, area

    def ok_ratio(w, h):
        if aspect is None:
            return True
        if h == 0:
            return False
        r = w / h
        return abs(r - aspect) <= aspect_tol * aspect

    for r in range(H):
        # update histogram of consecutive-empty heights
        row = empty[r]
        heights = np.where(row, heights + 1, 0)

        # largest-rectangle-in-histogram with a monotonic stack
        stack = []  # tuples: (start_col, height)
        for c in range(W + 1):  # sentinel at end
            curr_h = heights[c] if c < W else 0
            start = c
            # pop while the current bar is lower than the stack top
            while stack and stack[-1][1] > curr_h:
                sc, sh = stack.pop()
                w = c - sc
                h = sh
                r0 = r - h + 1
                c0 = sc
                if h >= min_h and w >= min_w and ok_ratio(w, h):
                    area = w * h
                    if area > best[4]:
                        best = (r0, c0, h, w, area)
                start = sc
            stack.append((start, curr_h))

    r0, c0, h, w, _ = best
    return r0, c0, h, w
