"""VoxeLibre zero-shot evaluation support.

The Craftium dependency is intentionally imported lazily by :mod:`.env` so
that installing or importing GameWAM does not change Minecraft training or
evaluation environments.
"""

from .tasks import TASK_NAMES, TASK_SPECS, VoxeLibreTaskSpec

__all__ = ["TASK_NAMES", "TASK_SPECS", "VoxeLibreTaskSpec"]
