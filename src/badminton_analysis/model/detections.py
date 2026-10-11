from dataclasses import dataclass, field
from typing import Any

import numpy as np

from badminton_analysis.model.detection.interface import Detection, DetectionModel
from badminton_analysis.model.detection.inference import YOLODetector


DETECTION_STRATEGIES: dict[str, type[DetectionModel]] = {
    "yolo": YOLODetector,
}


def create_detection_strategy(
    tag: str,
    **options: Any,
) -> DetectionModel:
    """Create a detection model from a client-facing strategy tag."""
    normalized_tag = tag.casefold()
    strategy_class = DETECTION_STRATEGIES.get(normalized_tag)
    return strategy_class(**options)


@dataclass
class Detector:
    """Client-facing interface for detection strategies."""

    strategy: str
    options: dict[str, Any] = field(default_factory=dict)
    model: DetectionModel = field(init=False)

    def __post_init__(self) -> None:
        self.model = create_detection_strategy(self.strategy, **self.options)

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.model.detect(image)

    def detect_batch(
        self,
        images: list[np.ndarray],
    ) -> list[list[Detection]]:
        return self.model.detect_batch(images)

    def set_strategy(
        self,
        strategy: str,
        **options: Any,
    ) -> None:
        self.strategy = strategy
        self.options = options
        self.model = create_detection_strategy(strategy, **options)
