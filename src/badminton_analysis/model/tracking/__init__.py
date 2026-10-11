"""Object tracking contracts."""

from badminton_analysis.model.tracking.interface import (
    TrackedDetection,
    TrackedSegmentation,
    TrackingModel,
)

__all__ = ["TrackedDetection", "TrackedSegmentation", "TrackingModel"]
