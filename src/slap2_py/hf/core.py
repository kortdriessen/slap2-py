from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any, Union

import h5py
import numpy as np

try:
    import scipy.sparse as sp  # optional

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

IndexLike = Union[int, slice, tuple[int | slice | None, ...]]


def _is_ref_dtype(dt) -> bool:
    return h5py.check_dtype(ref=dt) is not None


def _is_mat_char(dset: h5py.Dataset) -> bool:
    cls = dset.attrs.get("MATLAB_class", None)
    if isinstance(cls, bytes):
        cls = cls.decode("ascii", errors="ignore")
    return cls == "char" or (dset.dtype.kind in {"u"} and dset.dtype.itemsize == 2)


def _decode_char_array(u16: np.ndarray) -> str:
    arr = np.ascontiguousarray(u16.astype(np.uint16))
    s = arr.flatten(order="F").tobytes().decode("utf-16le", errors="ignore")
    return s.rstrip()


def _maybe_sparse_from_group(grp: h5py.Group) -> Any | None:
    keys = set(grp.keys())
    if {"ir", "jc"}.issubset(keys) and any(
        k in keys for k in ("data", "values", "entries")
    ):
        data_name = (
            "data" if "data" in keys else ("values" if "values" in keys else "entries")
        )
        ir = np.array(grp["ir"])
        jc = np.array(grp["jc"])
        data = np.array(grp[data_name])
        if "dims" in keys:
            m, n = np.array(grp["dims"]).astype(int).tolist()
        else:
            if "m" in grp and "n" in grp:
                m = int(np.array(grp["m"]))
                n = int(np.array(grp["n"]))
            else:
                m = int(grp.attrs.get("m", 0))
                n = int(grp.attrs.get("n", 0))
        if m and n:
            if _HAVE_SCIPY:
                return sp.csc_matrix((data, ir, jc), shape=(m, n))
            else:
                return {
                    "format": "CSC",
                    "data": data,
                    "ir": ir,
                    "jc": jc,
                    "shape": (m, n),
                }
    return None


class MatV73Reader:
    def __init__(
        self,
        path: str,
        *,
        rdcc_nbytes: int = 256 * 1024 * 1024,
        rdcc_nslots: int = 1_000_003,
        rdcc_w0: float = 0.75,
    ):
        self._f = h5py.File(
            path,
            "r",
            rdcc_nbytes=rdcc_nbytes,
            rdcc_nslots=rdcc_nslots,
            rdcc_w0=rdcc_w0,
        )

    def __enter__(self) -> MatV73Reader:
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

    def exists(self, path: str) -> bool:
        return path in self._f

    def ls(self, path: str = "/") -> Sequence[str]:
        obj = self._f[path]
        if isinstance(obj, h5py.Group):
            return sorted(list(obj.keys()))
        return []

    def info(self, path: str) -> dict:
        obj = self._f[path]
        info = {"path": path, "type": type(obj).__name__, "attrs": {}}
        if isinstance(obj, h5py.Dataset):
            info.update(
                {
                    "shape": tuple(obj.shape),
                    "dtype": str(obj.dtype),
                    "chunks": obj.chunks,
                    "compression": obj.compression,
                }
            )
        for k, v in obj.attrs.items():
            try:
                if isinstance(v, bytes):
                    v = v.decode("utf-8", "ignore")
                info["attrs"][k] = v
            except Exception:
                info["attrs"][k] = "<unrepr>"
        return info

    def array(self, path: str, sel: IndexLike | None = None) -> np.ndarray:
        dset = self._require_dataset(path)
        if _is_ref_dtype(dset.dtype):
            raise TypeError(
                f"{path} is a dataset of references; use cell_any/struct_at/etc."
            )
        if _is_mat_char(dset):
            data = dset[sel] if sel is not None else dset[...]
            return np.array(_decode_char_array(np.array(data)))
        return dset[sel] if sel is not None else dset[...]

    def cell_any(self, path: str, index: IndexLike) -> Any:
        dset = self._require_dataset(path)
        if not _is_ref_dtype(dset.dtype):
            raise TypeError(
                f"{path} is not a cell (dataset of references). dtype={dset.dtype}"
            )
        ref = dset[index]
        return self._load_ref(ref)

    def cell_slice(self, path: str, sel: IndexLike) -> list:
        dset = self._require_dataset(path)
        if not _is_ref_dtype(dset.dtype):
            raise TypeError(
                f"{path} is not a cell (dataset of references). dtype={dset.dtype}"
            )
        refs = dset[sel]
        refs = np.atleast_1d(refs)
        return [self._load_ref(r) for r in refs.flat]

    def cell_str(self, path: str, index: IndexLike) -> str:
        val = self.cell_any(path, index)
        if isinstance(val, str):
            return val
        if isinstance(val, np.ndarray) and val.dtype.kind in {"U", "S"}:
            return str(val)
        raise TypeError(f"Cell element at {path}{index} is not a string-like object.")

    def struct_at(
        self,
        path: str,
        index: int | tuple[int, ...],
        fields: Iterable[str] | None = None,
    ) -> dict:
        grp_or_dset = self._f[path]
        if isinstance(grp_or_dset, h5py.Group):
            field_names = list(grp_or_dset.keys()) if fields is None else list(fields)
            out = {}
            for fld in field_names:
                obj = grp_or_dset[fld]
                if isinstance(obj, h5py.Dataset) and _is_ref_dtype(obj.dtype):
                    ref = obj[index]
                    out[fld] = self._load_ref(ref)
                else:
                    out[fld] = self._load_obj(obj)
            return out
        if isinstance(grp_or_dset, h5py.Dataset) and _is_ref_dtype(grp_or_dset.dtype):
            ref = grp_or_dset[index]
            elem = self._load_ref(ref)
            if isinstance(elem, dict) and fields is not None:
                return {k: elem[k] for k in fields if k in elem}
            return elem
        raise TypeError(f"{path} does not look like a MATLAB struct array in v7.3.")

    def _require_dataset(self, path: str) -> h5py.Dataset:
        obj = self._f[path]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"{path} is not a Dataset (found {type(obj).__name__}).")
        return obj

    def _load_ref(self, ref: h5py.Reference) -> Any:
        if ref is None or (isinstance(ref, (bytes, np.void)) and len(ref) == 0):
            return None
        obj = self._f[ref]
        return self._load_obj(obj)

    def _load_obj(self, obj: h5py.Dataset | h5py.Group) -> Any:
        if isinstance(obj, h5py.Dataset):
            if _is_ref_dtype(obj.dtype):
                refs = obj[()]
                if np.ndim(refs) == 0:
                    return self._load_ref(refs)
                return [self._load_ref(r) for r in np.atleast_1d(refs).flat]
            if _is_mat_char(obj):
                return _decode_char_array(np.array(obj[...]))
            if obj.dtype.names and set(obj.dtype.names) >= {"real", "imag"}:
                real = obj.fields("real")[...]
                imag = obj.fields("imag")[...]
                return real + 1j * imag
            return obj[...]
        sparse = _maybe_sparse_from_group(obj)
        if sparse is not None:
            return sparse
        out = {}
        for k in obj.keys():
            out[k] = self._load_obj(obj[k])
        return out


# ---- Parser and meta-loader ----


def _parse_slice_token(tok: str):
    tok = tok.strip()
    if tok == ":" or tok == "":
        return slice(None)
    if ":" in tok:
        parts = tok.split(":")
        if len(parts) == 2:
            start = int(parts[0]) if parts[0] else None
            stop = int(parts[1]) if parts[1] else None
            return slice(start, stop)
        elif len(parts) == 3:
            start = int(parts[0]) if parts[0] else None
            stop = int(parts[1]) if parts[1] else None
            step = int(parts[2]) if parts[2] else None
            return slice(start, stop, step)
        else:
            raise ValueError(f"Invalid slice token: {tok}")
    return int(tok)


def _parse_index_expr(expr: str):
    expr = expr.strip()
    if not expr:
        return slice(None)
    parts = [p.strip() for p in expr.split(",")]
    items = [_parse_slice_token(p) for p in parts]
    if len(items) == 1:
        return items[0]
    return tuple(items)


def _tokenize_expr(expr: str):
    brack_pat = re.compile(r"\[[^\]]*\]")
    tokens = brack_pat.findall(expr)
    first = expr.find("[")
    base = expr if first == -1 else expr[:first]
    base = base.strip()
    steps: list[tuple[str, Any]] = []
    for tok in tokens:
        inner = tok[1:-1].strip()
        if (len(inner) >= 2) and ((inner[0] == inner[-1]) and inner[0] in ("'", '"')):
            field = inner[1:-1]
            steps.append(("field", field))
        else:
            idx = _parse_index_expr(inner)
            steps.append(("index", idx))
    return base, steps


def _unwrap_singleton(x: Any) -> Any:
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x


def _materialize_cell(reader, cell_path: str, unwrap_singletons: bool) -> list:
    info = reader.info(cell_path)
    if info["type"] != "Dataset" or "object" not in str(info.get("dtype", "")):
        raise TypeError(f"{cell_path} is not a dataset of references (cell array).")
    vals = reader.cell_slice(cell_path, np.s_[:, :])  # flat list
    shape = info["shape"]
    arr = np.array(vals, dtype=object).reshape(shape)
    out = []
    for i in range(shape[0]):
        row = []
        for j in range(shape[1]):
            elem = arr[i, j]
            if unwrap_singletons:
                elem = _unwrap_singleton(elem)
            row.append(elem)
        out.append(row)
    return out


def load_any(mat_path: str, expr: str, *, unwrap_singletons: bool = True) -> Any:
    base, steps = _tokenize_expr(expr)
    with MatV73Reader(mat_path) as r:
        if base not in r._f:
            raise KeyError(f"Base path not found: {base}")
        obj = r._f[base]

        if not steps:
            if isinstance(obj, h5py.Dataset):
                if _is_ref_dtype(obj.dtype):
                    return _materialize_cell(r, base, unwrap_singletons)
                else:
                    return r.array(base)
            elif isinstance(obj, h5py.Group):
                return r._load_obj(obj)
            else:
                return r._load_obj(obj)

        i = 0
        current: Any = None

        if isinstance(obj, h5py.Dataset) and _is_ref_dtype(obj.dtype):
            idx_parts = []
            while i < len(steps) and steps[i][0] == "index":
                idx_parts.append(steps[i][1])
                i += 1
            combined: list[Any] = []
            for part in idx_parts:
                if isinstance(part, tuple):
                    combined.extend(list(part))
                else:
                    combined.append(part)
            index = (
                tuple(combined)
                if len(combined) > 1
                else (combined[0] if combined else slice(None))
            )
            if any(
                isinstance(p, slice) for p in (combined if combined else [slice(None)])
            ):
                vals = r.cell_slice(
                    base, index if isinstance(index, tuple) else (index,)
                )
                current = vals
            else:
                current = r.cell_any(base, index)
                if unwrap_singletons:
                    current = _unwrap_singleton(current)

        elif isinstance(obj, h5py.Group):
            if steps[0][0] == "index":
                index = steps[0][1]
                if not isinstance(index, tuple):
                    index = (index,)
                current = r.struct_at(base, index)
                i += 1
            else:
                current = r._load_obj(obj)

        else:
            if steps[0][0] == "index":
                sel = steps[0][1]
                current = r.array(base, sel)
                i += 1
            else:
                current = r.array(base)

        while i < len(steps):
            kind, payload = steps[i]
            if kind == "field":
                if isinstance(current, dict):
                    current = current[payload]
                else:
                    raise TypeError(
                        f"Cannot apply field access ['{payload}'] to object of type {type(current)}"
                    )
            elif kind == "index":
                idx = payload
                if isinstance(current, np.ndarray):
                    current = current[idx]
                elif isinstance(current, list):
                    if isinstance(idx, tuple):
                        x = current[idx[0]]
                        for jdx in idx[1:]:
                            x = x[jdx]
                        current = x
                    else:
                        current = current[idx]
                else:
                    raise TypeError(f"Cannot index object of type {type(current)}")
            else:
                raise RuntimeError("Unknown step type")
            i += 1

        return current
