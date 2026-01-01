# ===============================================
# This module contains utility functions for working with
# synaptic data and localizations performed in ophys-slap2-analysis
# ===============================================

import numpy as np
from typing import Sequence, Optional, Dict, Tuple, List, Literal
from scipy.ndimage import center_of_mass
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from skimage.morphology import skeletonize

Method = Literal["auto", "skeleton", "mst", "pca"]


def create_synmap(fp, dmds=[1, 2]):
    """Create a synapse map from the full set of footprints.
    The synapse map is a 2D array where each pixel is the ID of the synapse that occupies that pixel.
    The synapse IDs are 0-indexed, and -1 is used for pixels that do not belong to any synapse.

    Parameters
    ----------
    fp : dict
        A dictionary of footprints, where each key is a DMD index and each value is a 3D array of shape (N, H, W).
    dmds : list, optional
        The DMD indices to include in the synapse map.

    Returns
    -------
    synmaps : dict
        A dictionary of synapse maps, where each key is a DMD index and each value is a 2D array of shape (H, W).
    """
    synmaps = {}
    for dmd in dmds:
        syns = fp[dmd]
        maps = []
        for i in range(syns.shape[0]):
            syn = syns[i]
            syn[syn > 0] = int(i + 1)
            syn[syn <= 0] = int(-1)
            syn[np.isnan(syn)] = int(-1)
            maps.append(syn)
        maps = np.array(maps)
        maps.shape
        synmap = np.max(maps, axis=0)
        synmap[synmap == 0] = int(-1)
        synmap = synmap.astype(int)
        synmap[synmap >= 1] -= 1
        synmaps[dmd] = synmap
    return synmaps


def sort_synapses_topologically(
    syn_map: np.ndarray,
    syn_ids: Sequence[int],
    dendrite_mask: Optional[np.ndarray] = None,
    method: Method = "auto",
    connectivity: int = 8,
) -> Dict:
    """
    Topologically sort synapses by arclength along a dendrite.
    Ideally, these should be synapses already localized to a single dendritic brach or domain.

    If `dendrite_mask` is provided and `method in {"auto","skeleton"}`, the function:
      1) skeletonizes the dendrite mask
      2) builds an 8-connected skeleton graph
      3) chooses a start endpoint from the graph diameter (orientation set by PC1)
      4) computes geodesic distances from start via Dijkstra
      5) snaps each synapse centroid to nearest skeleton pixel and sorts by distance

    Otherwise, if `method in {"auto","mst"}`, it:
      1) computes synapse centroids
      2) builds a k-NN graph on centroids, extracts MST
      3) chooses a start endpoint from the tree diameter (orientation set by PC1)
      4) computes tree-path distances from start and sorts

    As a last fallback, `method=="pca"` orders by projection onto the first principal component.

    Returns
    -------
    dict with keys:
      - sorted_ids: List[int]
      - arclengths: Dict[int, float]
      - centroids_rc: Dict[int, Tuple[float,float]]
      - method_used: str
      - start_endpoint_rc: Optional[Tuple[float,float]]
      - debug: Dict with extra internals
    """
    syn_ids = [int(i) for i in syn_ids if i >= 0]
    if len(syn_ids) == 0:
        return dict(
            sorted_ids=[],
            arclengths={},
            centroids_rc={},
            method_used=method,
            start_endpoint_rc=None,
            debug={},
        )

    # --- Centroids (row, col) for each synapse ID ---
    centroids_rc = _centroids_from_labels(syn_map, syn_ids)  # {id: (r,c)}
    # Filter out any IDs that had no pixels (NaNs)
    syn_ids = [i for i in syn_ids if i in centroids_rc]
    if len(syn_ids) <= 1:
        arcl = {i: 0.0 for i in syn_ids}
        return dict(
            sorted_ids=syn_ids,
            arclengths=arcl,
            centroids_rc=centroids_rc,
            method_used="trivial",
            start_endpoint_rc=None,
            debug={},
        )

    # Choose method
    use_skeleton = (
        (method in ("auto", "skeleton"))
        and dendrite_mask is not None
        and np.any(dendrite_mask)
    )
    if use_skeleton:
        method_used = "skeleton"
        result = _order_by_skeleton(
            dendrite_mask.astype(bool), centroids_rc, syn_ids, connectivity=connectivity
        )
        return result

    if method in ("auto", "mst"):
        method_used = "mst"
        result = _order_by_mst(centroids_rc, syn_ids)
        return result

    # PCA fallback
    method_used = "pca"
    ids_sorted, _, proj1 = _order_by_pca(centroids_rc, syn_ids)
    arclengths = {i: float(p) for i, p in zip(ids_sorted, proj1)}
    return dict(
        sorted_ids=ids_sorted,
        arclengths=arclengths,
        centroids_rc=centroids_rc,
        method_used=method_used,
        start_endpoint_rc=None,
        debug={"proj1": proj1},
    )


# ----------------- Helpers -----------------


def _centroids_from_labels(
    label_im: np.ndarray, ids: Sequence[int]
) -> Dict[int, Tuple[float, float]]:
    ids = list(dict.fromkeys([int(i) for i in ids]))  # unique, keep order
    coms = center_of_mass(
        np.ones_like(label_im, dtype=np.uint8), labels=label_im, index=ids
    )
    centroids = {}
    for lab, com in zip(ids, coms):
        if com is None:
            continue
        r, c = com
        if not (np.isnan(r) or np.isnan(c)):
            centroids[lab] = (float(r), float(c))
    return centroids


def _principal_axis(points_rc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, first principal direction) for (N,2) points in (row,col)."""
    X = np.asarray(points_rc, dtype=float)
    mu = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    pc1 = Vt[0]  # unit vector in (row,col) space
    return mu, pc1


def _build_skeleton_graph(
    skel: np.ndarray, connectivity: int = 8
) -> Tuple[np.ndarray, csr_matrix]:
    coords = np.column_stack(np.nonzero(skel))  # (N,2) -> rows, cols
    n = coords.shape[0]
    if n == 0:
        return coords, csr_matrix((0, 0))
    H, W = skel.shape
    index_map = -np.ones(skel.shape, dtype=np.int32)
    index_map[coords[:, 0], coords[:, 1]] = np.arange(n, dtype=np.int32)

    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    rows, cols, data = [], [], []
    for i, (r, c) in enumerate(coords):
        for dr, dc in offsets:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                j = index_map[rr, cc]
                if j >= 0 and j > i:  # add undirected edge once
                    w = 1.4142135623730951 if (dr != 0 and dc != 0) else 1.0
                    rows.append(i)
                    cols.append(j)
                    data.append(w)
    adj_upper = csr_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj_upper + adj_upper.T  # undirected
    return coords, adj


def _graph_diameter_endpoints(adj: csr_matrix) -> Tuple[int, int]:
    if adj.shape[0] == 0:
        return -1, -1
    # 2-sweep (approx diameter): 0 -> u (farthest), u -> v (farthest)
    d0 = dijkstra(adj, directed=False, indices=0)
    u = int(np.nanargmax(d0))
    du = dijkstra(adj, directed=False, indices=u)
    v = int(np.nanargmax(du))
    return u, v


def _order_by_skeleton(
    dendrite_mask: np.ndarray,
    centroids_rc: Dict[int, Tuple[float, float]],
    syn_ids: List[int],
    connectivity: int = 8,
) -> Dict:
    # 1) Skeletonize
    skel = skeletonize(dendrite_mask.astype(bool))
    coords, adj = _build_skeleton_graph(skel, connectivity=connectivity)
    if adj.shape[0] == 0:
        # Fallback to MST over centroids if skeletonization failed
        return _order_by_mst(centroids_rc, syn_ids, note="skeleton_empty")

    # 2) Choose start endpoint from graph diameter, orient by PC1
    u, v = _graph_diameter_endpoints(adj)
    if u < 0 or v < 0:
        return _order_by_mst(centroids_rc, syn_ids, note="skeleton_no_endpoints")

    # Orientation using PC1 of skeleton coordinates (row, col)
    mu, pc1 = _principal_axis(coords)
    proj_u = float((coords[u] - mu) @ pc1)
    proj_v = float((coords[v] - mu) @ pc1)
    start = u if proj_u < proj_v else v

    # 3) Geodesic distance from start to all skeleton nodes
    dist_from_start = dijkstra(adj, directed=False, indices=start)

    # 4) Snap each synapse centroid to nearest skeleton pixel
    skel_tree = cKDTree(coords)
    id_list = list(syn_ids)
    cent_mat = np.array([centroids_rc[i] for i in id_list], dtype=float)
    _, nn_idx = skel_tree.query(cent_mat, k=1)
    syn_dists = dist_from_start[nn_idx]  # geodesic arclength in pixels

    # tie-breaker: projection of centroids on PC1 of centroids
    _, pc1_cent = _principal_axis(cent_mat)
    proj_cent = (cent_mat - cent_mat.mean(axis=0)) @ pc1_cent

    # 5) Sort by (arclength, proj_cent) to break occasional ties
    order = np.lexsort((proj_cent, syn_dists))
    sorted_ids = [id_list[i] for i in order]
    arclengths = {id_list[i]: float(syn_dists[i]) for i in range(len(id_list))}

    return dict(
        sorted_ids=sorted_ids,
        arclengths=arclengths,
        centroids_rc=centroids_rc,
        method_used="skeleton",
        start_endpoint_rc=(float(coords[start, 0]), float(coords[start, 1])),
        debug={
            "skeleton_nodes": coords,
            "start_index": int(start),
            "diameter_pair": (int(u), int(v)),
            "dist_from_start": dist_from_start,
            "nearest_skel_index_per_synapse": nn_idx,
        },
    )


def _order_by_mst(
    centroids_rc: Dict[int, Tuple[float, float]],
    syn_ids: List[int],
    note: Optional[str] = None,
) -> Dict:
    # Points matrix
    ids = list(syn_ids)
    P = np.array([centroids_rc[i] for i in ids], dtype=float)  # (row, col)

    # Build sparse k-NN graph (k=3) to avoid O(N^2) full matrix for large N
    k = min(3, len(ids) - 1)
    tree = cKDTree(P)
    dists, nbrs = tree.query(P, k=k + 1)  # self included at index 0
    # Assemble symmetric adjacency
    rows, cols, data = [], [], []
    for i in range(len(ids)):
        for j_idx in range(1, k + 1):
            j = int(nbrs[i, j_idx])
            if j == i:
                continue
            w = float(np.linalg.norm(P[i] - P[j]))
            # keep only one direction for MST input; we will symmetrize later
            if j > i:
                rows.append(i)
                cols.append(j)
                data.append(w)
    n = len(ids)
    adj_upper = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Minimum spanning tree and symmetrize to undirected tree
    mst = minimum_spanning_tree(adj_upper)
    tree_adj = mst + mst.T

    # Graph diameter endpoints on the MST
    u, v = _graph_diameter_endpoints(tree_adj)

    # Orientation by PC1 of centroids
    mu, pc1 = _principal_axis(P)
    proj_u = float((P[u] - mu) @ pc1)
    proj_v = float((P[v] - mu) @ pc1)
    start = u if proj_u < proj_v else v

    # Distances along the tree
    dist_from_start = dijkstra(tree_adj, directed=False, indices=start)

    # Tie-breaker: PC2 (or projection itself). We'll reuse PC1 projection
    proj_cent = (P - mu) @ pc1

    order = np.lexsort((proj_cent, dist_from_start))
    sorted_ids = [ids[i] for i in order]
    arclengths = {ids[i]: float(dist_from_start[i]) for i in range(n)}

    return dict(
        sorted_ids=sorted_ids,
        arclengths=arclengths,
        centroids_rc=centroids_rc,
        method_used="mst" if note is None else f"mst[{note}]",
        start_endpoint_rc=(float(P[start, 0]), float(P[start, 1])),
        debug={
            "mst_adj": tree_adj,
            "start_index": int(start),
            "diameter_pair": (int(u), int(v)),
            "dist_from_start": dist_from_start,
        },
    )


def _order_by_pca(
    centroids_rc: Dict[int, Tuple[float, float]],
    syn_ids: List[int],
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    ids = list(syn_ids)
    P = np.array([centroids_rc[i] for i in ids], dtype=float)
    mu, pc1 = _principal_axis(P)
    proj = (P - mu) @ pc1  # 1D coordinate
    order = np.argsort(proj)
    ids_sorted = [ids[i] for i in order]
    return ids_sorted, order, proj


from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt


def gen_syn_sorting_video(synmap, synapse_ids_sorted, background_im, outpath):

    # MP4 export of a faster version (~3x speed)
    ids = list(synapse_ids_sorted)
    mask_list = [(synmap == src) for src in ids]

    cmap_bg = plt.cm.gray.copy()
    cmap_bg.set_bad((0.2, 0.2, 0.2, 1.0))

    f2, ax2 = plt.subplots(1, 1, figsize=(15, 5))
    ax2.imshow(np.asarray(background_im, dtype=float), cmap=cmap_bg)
    ax2.set_axis_off()

    accum_rgba = np.zeros((*synmap.shape, 4), dtype=float)
    im_accum2 = ax2.imshow(accum_rgba)
    im_current2 = ax2.imshow(np.zeros_like(accum_rgba))
    title2 = ax2.set_title("")

    def update(i):
        if i == 0:
            accum_mask = mask_list[0].copy()
        else:
            accum_mask = np.logical_or.reduce(mask_list[: i + 1])
        current = mask_list[i]

        accum_rgba[...] = 0.0
        m = accum_mask.astype(float)
        accum_rgba[..., 1] = m
        accum_rgba[..., 3] = m * 0.35
        im_accum2.set_data(accum_rgba)

        current_rgba = np.zeros_like(accum_rgba)
        c = current.astype(float)
        current_rgba[..., 1] = c
        current_rgba[..., 3] = c * 0.9
        im_current2.set_data(current_rgba)

        title2.set_text(f"Adding source {ids[i]} ({i+1}/{len(ids)})")
        return im_accum2, im_current2, title2

    fps_fast = 15  # ~3x faster than 5 fps
    ani = animation.FuncAnimation(
        f2, update, frames=len(ids), interval=1000 / fps_fast, blit=False, repeat=False
    )

    try:
        writer = animation.FFMpegWriter(fps=fps_fast, codec="libx264")
        ani.save(outpath, writer=writer, dpi=150)
        print(f"Saved fast MP4 to {outpath}")
    except Exception as e:
        print("MP4 save failed. Ensure ffmpeg is installed.", e)

    plt.close(f2)
