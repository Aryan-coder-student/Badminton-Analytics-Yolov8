from dataclasses import dataclass, field

from badminton_analysis.model.detection import Detection
from badminton_analysis.model.tracking import TrackedDetection, TrackingModel
from badminton_analysis.model.tracking.inference import ByteTrack


TRACKING_STRATEGIES: dict[str, type[TrackingModel]] = {
    "bytetrack": ByteTrack,
}


def create_tracking_strategy(
    tag: str,
    **options: object,
) -> TrackingModel:
    """Create a tracker from a client-facing strategy tag."""
    return TRACKING_STRATEGIES[tag.casefold()](**options)


@dataclass
class Tracker:
    """Client-facing interface for stateful tracking strategies."""

    strategy: str
    options: dict[str, object] = field(default_factory=dict)
    model: TrackingModel = field(init=False)

    def __post_init__(self) -> None:
        self.model = create_tracking_strategy(self.strategy, **self.options)

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> list[TrackedDetection]:
        return self.model.update(detections, frame_shape)

    def set_strategy(
        self,
        strategy: str,
        **options: object,
    ) -> None:
        self.strategy = strategy
        self.options = options
        self.model = create_tracking_strategy(strategy, **options)
