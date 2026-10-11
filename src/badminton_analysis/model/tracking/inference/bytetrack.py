from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from badminton_analysis.model.detection import Detection
from badminton_analysis.model.settings import get_settings
from badminton_analysis.model.tracking.interface import (
    TrackedDetection,
    TrackingModel,
)


class ByteTrackBackend(Protocol):
    """Ultralytics ByteTrack operations used by the adapter."""

    def update(self, results: Boxes) -> np.ndarray:
        """Advance the tracker by one frame."""


def create_bytetrack_backend() -> BYTETracker:
    """Create ByteTrack from Ultralytics' bundled configuration."""
    config_path = check_yaml(get_settings().tracking_config_name)
    arguments = IterableSimpleNamespace(**YAML.load(config_path))
    return BYTETracker(arguments)


@dataclass
class ByteTrack(TrackingModel):
    """Adapt domain detections to Ultralytics' stateful ByteTrack tracker."""

    backend: ByteTrackBackend = field(
        default_factory=create_bytetrack_backend,
        repr=False,
    )

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> list[TrackedDetection]:
        tracker_input = np.asarray(
            [
                [*detection.bbox, detection.confidence, detection.class_id]
                for detection in detections
            ],
            dtype=np.float32,
        ).reshape(-1, 6)
        tracked_rows = self.backend.update(Boxes(tracker_input, frame_shape))

        return [
            TrackedDetection(
                detection=detections[int(row[-1])],
                track_id=int(row[4]),
            )
            for row in tracked_rows
        ]
