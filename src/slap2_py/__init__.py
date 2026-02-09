# Blessed top-level API
# Make subpackages appear under `slap2_py.`
# Optional imports are wrapped so lightweight modules (like event detection)
# can be used even if heavier, optional dependencies are missing.
try:
    from . import img as img
except Exception:
    img = None  # type: ignore[assignment]

try:
    from . import plot as plot
except Exception:
    plot = None  # type: ignore[assignment]

try:
    from . import utils as utils
except Exception:
    utils = None  # type: ignore[assignment]

try:
    from . import hf as hf
except Exception:
    hf = None  # type: ignore[assignment]

from .core import (
    xsum as xsum,
)
from .core import (
    xsum_df as xsum_df,
)

__all__ = [
    "xsum",
    "xsum_df",
    "utils",
    "plot",
    "hf",
    "img",
]
