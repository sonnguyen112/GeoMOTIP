"""Optional visual trail context for MOTIP. No independent identity dictionary."""

from .visual_trail import VisualTrail, TrailState
from .trail_encoder import TrailEncoder
from .trail_gate import TrailGate
from .trail_memory_manager import TrailMemoryManager, TrailMemoryState, PendingObservation

__all__ = ["VisualTrail", "TrailState", "TrailMemoryState", "TrailMemoryManager",
           "PendingObservation", "TrailEncoder", "TrailGate", "trail_mode"]


def trail_mode(config):
    # USE_TRAIL is authoritative, including over an inherited V0 flag.
    enabled = config.get("USE_TRAIL", config.get("TRAIL_ENABLED", False))
    mode = config.get("TRAIL_TYPE", "v0" if config.get("TRAIL_ENABLED", False) else "visual")
    if enabled and mode not in ("v0", "visual"):
        raise ValueError(f"Unknown TRAIL_TYPE: {mode}")
    return mode if enabled and not config.get("ONLY_DETR", False) else None
