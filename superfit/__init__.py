"""
ADOBE

Copyright 2026 Adobe

All Rights Reserved.

NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe
and its suppliers and are protected by all applicable intellectual
property laws, including trade secret and copyright laws.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Adobe.
"""
import sys

from .utils import config as _config  # noqa: F401 — bind utils package before star imports

from .mat_opt import *
from .shader import *
from .symbolic import *
from .torch_compute import *

if "superfit.utils" in sys.modules:
    sys.modules[__name__].utils = sys.modules["superfit.utils"]

__version__ = "0.1.0"
