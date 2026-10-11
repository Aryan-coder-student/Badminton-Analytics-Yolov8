from dataclasses import dataclass, field

import numpy as np

from badminton_analysis.model.segmentation import Segmentation, SegmentationModel
from badminton_analysis.model.segmentation.inference import YOLOSegmenter


SEGMENTATION_STRATEGIES: dict[str, type[SegmentationModel]] = {
    "yolo": YOLOSegmenter,
}


def create_segmentation_strategy(
    tag: str,
    **options: object,
) -> SegmentationModel:
    """Create a segmentation model from a client-facing strategy tag."""
    return SEGMENTATION_STRATEGIES[tag.casefold()](**options)


@dataclass
class Segmenter:
    """Client-facing interface for segmentation strategies."""

    strategy: str
    options: dict[str, object] = field(default_factory=dict)
    model: SegmentationModel = field(init=False)

    def __post_init__(self) -> None:
        self.model = create_segmentation_strategy(self.strategy, **self.options)

    def segment(self, image: np.ndarray) -> list[Segmentation]:
        return self.model.segment(image)

    def segment_batch(
        self,
        images: list[np.ndarray],
    ) -> list[list[Segmentation]]:
        return self.model.segment_batch(images)

    def set_strategy(
        self,
        strategy: str,
        **options: object,
    ) -> None:
        self.strategy = strategy
        self.options = options
        self.model = create_segmentation_strategy(strategy, **options)
