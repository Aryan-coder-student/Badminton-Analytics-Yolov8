from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from badminton_analysis.model.detection import Detection


@dataclass(frozen=True, slots=True, eq=False)
class Segmentation:
    """Object detection paired with its full-resolution boolean mask."""

    detection: Detection
    mask: np.ndarray


class SegmentationModel(ABC):
    """Contract implemented by image segmentation strategies."""

    def segment(self, image: np.ndarray) -> list[Segmentation]:
        return self.segment_batch([image])[0]

    @abstractmethod
    def segment_batch(
        self,
        images: list[np.ndarray],
    ) -> list[list[Segmentation]]:
        """Segment objects in a batch of BGR images."""
