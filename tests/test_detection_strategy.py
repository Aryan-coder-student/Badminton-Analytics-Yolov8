import unittest
from unittest.mock import patch

import numpy as np

from badminton_analysis.model.detection import Detection, DetectionModel
from badminton_analysis.model.detections import DETECTION_STRATEGIES, Detector


class FakeDetectionModel(DetectionModel):
    def __init__(self, class_name: str = "person") -> None:
        self.class_name = class_name

    def detect(self, image: np.ndarray) -> list[Detection]:
        del image
        return [Detection((1.0, 2.0, 3.0, 4.0), 0.9, 0, self.class_name)]

    def detect_batch(
        self,
        images: list[np.ndarray],
    ) -> list[list[Detection]]:
        return [self.detect(image) for image in images]


class AlternateDetectionModel(FakeDetectionModel):
    def __init__(self) -> None:
        super().__init__(class_name="sports ball")


class DetectionStrategyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((16, 16, 3), dtype=np.uint8)

    def test_detector_configures_strategy_from_tag(self) -> None:
        with patch.dict(DETECTION_STRATEGIES, {"fake": FakeDetectionModel}):
            detector = Detector("FAKE")

        self.assertEqual(detector.detect(self.image)[0].class_name, "person")

    def test_detector_forwards_options(self) -> None:
        with patch.dict(DETECTION_STRATEGIES, {"fake": FakeDetectionModel}):
            detector = Detector("fake", options={"class_name": "racket"})

        self.assertEqual(detector.detect(self.image)[0].class_name, "racket")

    def test_detector_delegates_batch_and_replaces_strategy(self) -> None:
        strategies = {
            "fake": FakeDetectionModel,
            "alternate": AlternateDetectionModel,
        }
        with patch.dict(DETECTION_STRATEGIES, strategies):
            detector = Detector("fake")
            self.assertEqual(len(detector.detect_batch([self.image, self.image])), 2)

            detector.set_strategy("alternate")

        self.assertEqual(detector.detect(self.image)[0].class_name, "sports ball")


if __name__ == "__main__":
    unittest.main()
