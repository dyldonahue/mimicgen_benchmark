"""MimicGen Benchmark - Custom environments and objects."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent
ASSETS_ROOT = PACKAGE_ROOT / "assets"

from . import custom_env
from . import custom_objects
from . import custom_tasks
from . import custom_subtasks

__version__ = "0.1.0"