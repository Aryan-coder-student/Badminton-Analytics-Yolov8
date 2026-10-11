from abc import ABC, abstractmethod
from dataclasses import dataclass

from badminton_analysis.model.detection import Detection
from badminton_analysis.model.segmentation import Segmentation


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """A detection assigned a persistent identity by a tracker."""

    detection: Detection
    track_id: int


@dataclass(frozen=True, slots=True, eq=False)
class TrackedSegmentation:
    """A local track ID paired with its exact detection and mask."""

    track_id: int
    segmentation: Segmentation

    @property
    def detection(self) -> Detection:
        return self.segmentation.detection


class TrackingModel(ABC):
    """Contract for stateful, frame-by-frame object trackers."""

    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> list[TrackedDetection]:
        """Update tracker state with one frame of ordered detections."""
