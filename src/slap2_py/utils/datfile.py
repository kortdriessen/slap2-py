"""Pure-Python reader for SLAP2 .dat file headers.

The SLAP2 raw .dat file format stores a compact binary header describing
recording parameters, followed by a stream of fixed-size "line headers"
interleaved with photon-count line data. This module provides a minimal
reader that extracts the file header fields and the first line header's
uint64 timestamp — everything you need to reconstruct the per-epoch
start time (on the microscope's FPGA clock) without reading any line
data.

The on-disk layout is documented in MATLAB at:

- File header (V2): ``slap2/+slap2/+util/@DataFile/private/loadFileHeaderV2.m``
- Line header:      ``slap2/+slap2/+util/@DataFile/parseLineHeader.m``
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# File-header field IDs (V2) — 0-indexed enum per loadFileHeaderV2.m
_V2_FIELD_NAMES: tuple[str, ...] = (
    "firstCycleOffsetBytes",
    "lineHeaderSizeBytes",
    "laserPathIdx",
    "bytesPerCycle",
    "linesPerCycle",
    "superPixelsPerCycle",
    "dmdPixelsPerRow",
    "dmdPixelsPerColumn",
    "numChannels",
    "channelMask",
    "numSlices",
    "channelsInterleave",
    "fpgaSystemClock_Hz",
    "referenceTimestamp_lower",
    "referenceTimestamp_upper",
)


@dataclass(frozen=True)
class DatHeaderInfo:
    """Parsed SLAP2 .dat file header + first-line timestamp."""

    file_version: int
    fpga_system_clock_hz: int
    first_cycle_offset_bytes: int
    line_header_size_bytes: int
    reference_timestamp: int  # uint64, (upper << 32) | lower
    first_line_timestamp: int  # uint64, raw FPGA tick count

    @property
    def first_line_timestamp_s(self) -> float:
        """First line's FPGA timestamp in seconds (tick / clock_rate)."""
        return self.first_line_timestamp / self.fpga_system_clock_hz


def read_dat_header(path: str | Path) -> DatHeaderInfo:
    """Read a SLAP2 .dat file's header and first line-header timestamp.

    Only reads enough bytes to parse the two headers (typically well under
    1 KB), without loading any line data. V2 files are the only supported
    format; V1 files will raise ``ValueError``.

    Parameters
    ----------
    path : str | Path
        Path to a SLAP2 .dat file.

    Returns
    -------
    DatHeaderInfo
        Dataclass with ``fpga_system_clock_hz``, ``first_line_timestamp``
        (raw FPGA ticks), and the convenience property
        ``first_line_timestamp_s`` giving seconds.

    Raises
    ------
    ValueError
        If the file's version is not 2 or its header magic numbers don't
        match.
    """
    with open(path, "rb") as f:
        magic_start = struct.unpack("<I", f.read(4))[0]
        version = struct.unpack("<I", f.read(4))[0]
        if version != 2:
            raise ValueError(
                f"Unsupported SLAP2 .dat version {version}; V2 expected "
                f"({path})"
            )
        header_size_bytes = struct.unpack("<I", f.read(4))[0]

        # The header layout is: [magic_start u32][version u32][size u32]
        # then (fieldId u32, fieldVal u32) pairs, then [magic_end u32].
        # Pairs count = ((size / 4) - 3 header u32s - 1 magic_end u32) / 2.
        header_entries = header_size_bytes // 4
        n_pairs = (header_entries - 4) // 2

        hdr: dict[str, int] = {}
        for _ in range(n_pairs):
            fid, fval = struct.unpack("<II", f.read(8))
            if 0 <= fid < len(_V2_FIELD_NAMES):
                hdr[_V2_FIELD_NAMES[fid]] = fval
            # Unknown field IDs are silently skipped — matches the MATLAB
            # reader's behavior (it emits a warning; we don't, to keep the
            # reader quiet for bulk use).

        magic_end = struct.unpack("<I", f.read(4))[0]
        if magic_start != magic_end:
            raise ValueError(
                f"SLAP2 .dat header magic mismatch in {path}: "
                f"start=0x{magic_start:08x} end=0x{magic_end:08x}"
            )

        # referenceTimestamp is stored as two u32 halves
        ref_lo = hdr.get("referenceTimestamp_lower", 0)
        ref_hi = hdr.get("referenceTimestamp_upper", 0)
        reference_ts = (ref_hi << 32) | ref_lo

        # Jump to the first line header; only its uint64 timestamp (at
        # byte offset 24 of the 32-byte base layout) is needed here.
        f.seek(hdr["firstCycleOffsetBytes"])
        line_hdr_bytes = f.read(32)
        # u32 lineSize, u32 magic, u32 lineNumber, u32 acqNumber,
        # u32 flags, i16 xOff, i16 yOff, u64 timestamp
        _, _, _, _, _, _, _, line_ts = struct.unpack(
            "<IIIIIhhQ", line_hdr_bytes
        )

    return DatHeaderInfo(
        file_version=version,
        fpga_system_clock_hz=hdr["fpgaSystemClock_Hz"],
        first_cycle_offset_bytes=hdr["firstCycleOffsetBytes"],
        line_header_size_bytes=hdr["lineHeaderSizeBytes"],
        reference_timestamp=reference_ts,
        first_line_timestamp=line_ts,
    )


# Matches ``..._YYYYMMDD_HHMMSS_DMD{n}-CYCLE-000000.dat`` filenames. The
# YYYYMMDD_HHMMSS substring is the microscope's wall-clock start for the
# epoch; we sort on this (not on any ``acq-N`` prefix, which is not
# guaranteed to be chronologically monotonic).
_EPOCH_FILE_RE = re.compile(r"(\d{8}_\d{6})_DMD(\d+)-CYCLE-000000\.dat$")


def compute_epoch_offsets_from_dats(
    acq_dir: str | Path, dmd: int = 1
) -> dict[int, float]:
    """Compute per-epoch offsets (in seconds) relative to epoch 1 using
    the first line-header FPGA timestamp of each epoch's ``.dat`` file.

    The microscope's FPGA clock is continuous across stop/restarts, so
    these offsets are exact to sub-μs regardless of how long the user
    paused between epochs.

    Parameters
    ----------
    acq_dir : str | Path
        Path to the acquisition directory containing the raw ``.dat``
        files (e.g. ``<data_root>/<subject>/<exp>/<loc>/<acq>/``).
    dmd : int
        Which DMD's first-cycle files to read. Default 1. Either DMD
        produces the same offsets since both share the FPGA clock.

    Returns
    -------
    dict[int, float]
        ``{1: 0.0, 2: t2_s - t1_s, 3: t3_s - t1_s, ...}`` — epoch index
        (1-based) mapped to seconds since epoch 1.
    """
    acq_dir = Path(acq_dir)
    if not acq_dir.is_dir():
        raise FileNotFoundError(f"Acq dir not found: {acq_dir}")

    found: list[tuple[datetime, Path]] = []
    for fname in os.listdir(acq_dir):
        m = _EPOCH_FILE_RE.search(fname)
        if m is None:
            continue
        if int(m.group(2)) != dmd:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        found.append((ts, acq_dir / fname))
    if not found:
        raise FileNotFoundError(
            f"No ``*_DMDx-CYCLE-000000.dat`` files found in {acq_dir} for DMD {dmd}"
        )
    found.sort(key=lambda x: x[0])

    first_ts_s: float | None = None
    offsets: dict[int, float] = {}
    for i, (_wall_ts, path) in enumerate(found, start=1):
        hdr = read_dat_header(path)
        fpga_s = hdr.first_line_timestamp_s
        if i == 1:
            first_ts_s = fpga_s
        offsets[i] = float(fpga_s - first_ts_s)
    return offsets
