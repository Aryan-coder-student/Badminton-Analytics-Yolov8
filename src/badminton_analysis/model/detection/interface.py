from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """Framework-independent object detection result in XYXY coordinates."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class DetectionModel(ABC):
    """Contract implemented by object detection strategies."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect objects in one BGR image."""

    @abstractmethod
    def detect_batch(
        self,
        images: list[np.ndarray],
    ) -> list[list[Detection]]:
        """Detect objects in a batch of BGR images."""
