"""Recover the two channels of a corrupt DataRec SYNC HDF5 by *layout*.

The SLAP2 DataRecorder writes ``slap2_acquiring_trigger`` and
``electrophysiology`` as uncompressed float32 datasets with 50 000-sample
chunks, alternating S, E, S, E, ... in the file (the strict alternation can
break later in a recording), with B-tree metadata blocks of 2096 / 4192 bytes
inserted at deterministic chunk positions. A file that was never closed has a
broken root object header (``bad object header version number``) but all its
chunk payloads are on disk, back-to-back, in that layout. :func:`recover`

1. reads the chunk layout (first offset, order, metadata gap sizes) from a
   HEALTHY template file written by the same recorder (h5py chunk info),
2. walks the BAD file chunk by chunk: every chunk starts where the previous
   one ended plus the smallest known metadata gap at which a plausible,
   non-flat chunk is found; each chunk is assigned to a dataset by level
   continuity with that dataset's previous chunk (with a level prior and a
   chunk-count tie-break),
3. stops at the unflushed tail (chunk cache lost at the crash), and
4. writes a clean HDF5 with the template's structure plus a text report.

Validate a recovery with the byte audit in the report (header + chunks +
expected metadata blocks must equal the file size) and by checking ephys
continuity across chunk boundaries.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import h5py
import numpy as np

CHUNK = 50_000
CHUNK_BYTES = 4 * CHUNK
SEARCH_BYTES = 16_384  # local search radius when a predicted offset fails
SEARCH_STEP = 8  # HDF5 allocations are 8-byte aligned
NAMES = {"S": "slap2_acquiring_trigger", "E": "electrophysiology"}


@dataclass
class Template:
    first_offset: int
    order: list[str]  # dataset tag per chunk position, e.g. S,E,S,E,...
    extra_after: dict[int, int]  # chunk position -> extra bytes before the next chunk
    fs: float
    attrs: dict[str, dict]


@dataclass
class Recovered:
    chunks: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {"S": [], "E": []}
    )
    offsets: dict[str, list[int]] = field(default_factory=lambda: {"S": [], "E": []})
    corrections: list[tuple[int, int, int]] = field(
        default_factory=list
    )  # (pos, predicted, found)
    continuity_breaks: list[tuple[str, int, float]] = field(default_factory=list)
    stop_reason: str = ""


def read_template(path: str) -> Template:
    with h5py.File(path, "r") as h:
        rows = []
        for name in h.keys():
            tag = "S" if "trigger" in name.lower() else "E"
            dsid = h[name].id
            for i in range(dsid.get_num_chunks()):
                ci = dsid.get_chunk_info(i)
                if ci.size != CHUNK_BYTES:
                    raise RuntimeError(f"unexpected chunk size {ci.size} in {path}")
                rows.append((ci.byte_offset, tag))
        rows.sort()
        extra = {}
        for k in range(len(rows) - 1):
            gap = rows[k + 1][0] - rows[k][0] - CHUNK_BYTES
            if gap:
                extra[k] = gap
        attrs = {name: dict(h[name].attrs) for name in h.keys()}
        return Template(
            rows[0][0],
            [t for _, t in rows],
            extra,
            float(h.attrs.get("samplerate", 5000.0)),
            attrs,
        )


def plausible(x: np.ndarray, tag: str) -> bool:
    if x.size != CHUNK or not np.all(np.isfinite(x)):
        return False
    lo, hi = float(x.min()), float(x.max())
    if lo < -1.0 or hi > 6.0:
        return False
    if tag == "S":  # TTL 0 / ~5 V with brief dips; the low state sits at ~0.001 V
        near = np.mean((np.abs(x) < 0.05) | (np.abs(x - 5.0) < 0.4))
        return near >= 0.9
    # ephys copy of the #Enable line: ~0-0.5 V physiological-ish, or ~3.3 V TTL
    return hi < 4.5


def _flat(x: np.ndarray) -> bool:
    """True when the first 200 samples carry no ADC noise (σ < 1e-6): zero-filled
    or denormal-garbage metadata, never real data (its noise is ≥ ~1e-4 V)."""
    return float(np.std(x[:200].astype(np.float64))) < 1e-6


def _tag_for(
    x: np.ndarray,
    last_mean: dict[str, float | None],
    last_val: dict[str, float | None],
    counts: dict[str, int],
) -> tuple[str, float] | None:
    """Assign a plausible chunk to a dataset by level continuity.

    Returns ``(tag, distance)``: distance = |mean(first 100 samples) − mean(last
    100 samples of that dataset's previous chunk)| + |x[0] − last sample|, plus
    a penalty of 1.0 when the chunk's median contradicts the dataset's level
    prior (trigger median ≈ 0.001 V or ≈ 5 V; the ephys copy never sits at 5 V
    and its quiet baseline is ≥ ~0.003 V). Large distances are allowed (a real
    TTL edge can fall on a chunk boundary); the caller logs them. None if the
    chunk is not plausible for either dataset. Ties go to the dataset with
    fewer chunks so far.
    """
    if x.size != CHUNK or not np.all(np.isfinite(x)) or _flat(x):
        return None
    head = float(x[:100].mean())
    med = float(np.median(x[::25]))
    best = None
    for tag in ("S", "E"):
        if not plausible(x, tag):
            continue
        d = 0.0
        if last_mean[tag] is not None:
            d = abs(head - last_mean[tag]) + abs(float(x[0]) - last_val[tag])
        if tag == "S" and not (med < 0.0025 or med > 4.5):
            d += 1.0
        if tag == "E" and med > 4.5:
            d += 1.0
        if (
            best is None
            or d < best[1] - 1e-4
            or (abs(d - best[1]) <= 1e-4 and counts[tag] < counts[best[0]])
        ):
            best = (tag, d)
    return best


def carve(bad_path: str, tpl: Template) -> Recovered:
    rec = Recovered()
    size = os.path.getsize(bad_path)
    last_mean: dict[str, float | None] = {"S": None, "E": None}
    last_val: dict[str, float | None] = {"S": None, "E": None}
    counts = {"S": 0, "E": 0}
    known_gaps = sorted({
        0,
        2096,
        4192,
        2096 * 2,
        4192 * 2,
        2096 + 4192,
        *tpl.extra_after.values(),
    })
    forward_search = 2_000_000
    pos = 0
    offset = (
        tpl.first_offset
    )  # every later chunk starts where the previous one ended (+ metadata)
    with open(bad_path, "rb") as f:

        def read_at(off: int) -> np.ndarray | None:
            if off < 0 or off + CHUNK_BYTES > size:
                return None
            f.seek(off)
            return np.frombuffer(f.read(CHUNK_BYTES), dtype="<f4")

        def find_next(base: int):
            # Chunks are back-to-back except for metadata blocks of known sizes:
            # take the SMALLEST gap whose chunk is plausible. (Ranking gaps by
            # continuity is unreliable in flat regions, where every offset inside
            # a chunk is "continuous"; zero-filled / denormal metadata is
            # rejected by the flat-run test inside _tag_for.)
            for gap in known_gaps:
                x = read_at(base + gap)
                if x is None:
                    continue
                t = _tag_for(x, last_mean, last_val, counts)
                if t is not None:
                    return base + gap, x, t[0], t[1], gap
            # Rare: metadata larger than any known gap. Forward search with a
            # cheap prefilter (first sample continuous with either dataset,
            # first 16 samples finite/in range, not a flat run).
            f.seek(base)
            buf = f.read(min(forward_search + CHUNK_BYTES, size - base))
            arr = np.frombuffer(buf, dtype="<f4")
            for off8 in range(0, len(buf) - CHUNK_BYTES, SEARCH_STEP):
                i = off8 // 4
                v0 = float(arr[i])
                if not np.isfinite(v0) or not any(
                    lv is not None and abs(v0 - lv) < 0.5 for lv in last_val.values()
                ):
                    continue
                head = arr[i : i + 16]
                if (
                    not np.all(np.isfinite(head))
                    or head.min() < -1.0
                    or head.max() > 6.0
                ):
                    continue
                x = arr[i : i + CHUNK]
                if _flat(x):
                    continue
                t = _tag_for(x, last_mean, last_val, counts)
                if t is not None:
                    return base + off8, x.copy(), t[0], t[1], off8
            return None

        import time as _time

        t0 = _time.time()
        while True:
            if pos and pos % 200 == 0:
                print(
                    f"  ... chunk {pos} (S {counts['S']}, E {counts['E']}) "
                    f"at {_time.time() - t0:.1f} s",
                    flush=True,
                )
            hit = find_next(offset)
            if hit is None:
                rec.stop_reason = (
                    f"no plausible chunk within {forward_search} B after offset "
                    f"{offset} (chunk position {pos})"
                )
                break
            off, x, tag, dist, gap = hit
            if gap not in (0, 2096, 4192) and gap not in tpl.extra_after.values():
                rec.corrections.append((pos, offset, off))
            if last_val[tag] is not None and abs(float(x[0]) - last_val[tag]) > 0.5:
                rec.continuity_breaks.append((
                    tag,
                    pos,
                    round(float(abs(x[0] - last_val[tag])), 3),
                ))
            rec.chunks[tag].append(x.copy())
            rec.offsets[tag].append(off)
            counts[tag] += 1
            last_mean[tag] = float(x[-100:].mean())
            last_val[tag] = float(x[-1])
            offset = off + CHUNK_BYTES
            pos += 1
    return rec


def write_out(out_path: str, tpl: Template, rec: Recovered, report: list[str]) -> None:
    with h5py.File(out_path, "w") as h:
        h.attrs["samplerate"] = tpl.fs
        h.attrs["recovered_by"] = "recover_sync_layout.py"
        for tag, name in NAMES.items():
            data = (
                np.concatenate(rec.chunks[tag])
                if rec.chunks[tag]
                else np.zeros(0, "<f4")
            )
            d = h.create_dataset(name, data=data, chunks=(CHUNK,), dtype="<f4")
            for k, v in tpl.attrs.get(name, {}).items():
                d.attrs[k] = v
    with open(out_path + "_report.txt", "w") as fh:
        fh.write("\n".join(report) + "\n")


def recover(good: str, bad: str, out: str) -> list[str]:
    """Carve ``bad`` using ``good`` as the layout template; write ``out`` and
    ``out + '_report.txt'``. Returns the report lines."""
    tpl = read_template(good)
    rec = carve(bad, tpl)
    report = build_report(good, bad, tpl, rec)
    write_out(out, tpl, rec, report)
    return report


def _metadata_decomposition(remaining: int) -> str:
    """Express leftover bytes as a×4192 + b×2096 (+ tail) — the only block sizes
    the recorder inserts between chunks — so a slip shows up as a non-zero tail."""
    for a in range(0, 8):
        rest = remaining - a * 4192
        if rest >= 0 and rest % 2096 == 0:
            return f"{a} x 4192 + {rest // 2096} x 2096 (exact)"
    return f"NOT a×4192 + b×2096 (tail {remaining % 2096} B) — inspect"


def integrity_checks(rec: Recovered, tpl: Template, bad_size: int) -> list[str]:
    """Checks a mis-assembled recovery cannot pass; printed in the report."""
    n_s, n_e = len(rec.chunks["S"]), len(rec.chunks["E"])
    used = (n_s + n_e) * CHUNK_BYTES
    remaining = bad_size - tpl.first_offset - used
    lines = [
        f"byte audit: header {tpl.first_offset} + chunks {used} + remaining "
        f"{remaining} B = file size {bad_size}; remaining = "
        f"{_metadata_decomposition(remaining)}",
        f"dataset balance: |trigger − ephys| = {abs(n_s - n_e)} chunks "
        "(expect ≤ ~5: chunk-cache tail only)",
    ]
    if n_e > 1:
        e = np.concatenate(rec.chunks["E"]).astype(np.float64)
        b = np.arange(CHUNK, e.size, CHUNK)
        jumps = np.abs(e[b] - e[b - 1])
        inner = np.abs(np.diff(e))
        lines.append(
            f"ephys continuity at {b.size} chunk boundaries: "
            f"max |Δ| {jumps.max():.4f} V "
            f"vs within-chunk 99.9 % {np.percentile(inner, 99.9):.4f} V "
            f"(boundaries > 0.05 V: {int((jumps > 0.05).sum())}; "
            "expect 0 apart from the #Enable rise)"
        )
    return lines


def build_report(good: str, bad: str, tpl: Template, rec: Recovered) -> list[str]:
    n_s, n_e = len(rec.chunks["S"]), len(rec.chunks["E"])
    size = os.path.getsize(bad)
    return integrity_checks(rec, tpl, size) + [
        f"bad file: {bad} ({size} B)",
        f"template: {good} (first chunk @ {tpl.first_offset}, {len(tpl.order)} chunks, "
        f"{len(tpl.extra_after)} metadata gaps)",
        f"recovered chunks: trigger {n_s}, ephys {n_e}  -> "
        f"{n_s * CHUNK / tpl.fs:.1f} s / {n_e * CHUNK / tpl.fs:.1f} s",
        f"stop reason: {rec.stop_reason}",
        f"offset corrections (pos, predicted, found): {rec.corrections[:20]}"
        f"{' ...' if len(rec.corrections) > 20 else ''}",
        "continuity breaks > 0.5 V at chunk boundaries (tag, pos, jump): "
        f"{rec.continuity_breaks[:20]}"
        f"{' ...' if len(rec.continuity_breaks) > 20 else ''}",
    ]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--good",
        required=True,
        help="healthy SYNC file from the same recorder (layout template)",
    )
    ap.add_argument("--bad", required=True, help="corrupt SYNC file")
    ap.add_argument("--out", required=True, help="output HDF5 path")
    args = ap.parse_args(argv)
    report = recover(args.good, args.bad, args.out)
    print("\n".join(report))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
