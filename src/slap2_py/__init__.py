# Blessed top-level API
from .core.SlapData import SlapData

# Make subpackages appear under `wisco_slap.`
from . import img as img
from . import utils as utils

__all__ = ['SlapData', 'img', 'utils']