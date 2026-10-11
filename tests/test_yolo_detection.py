import unittest
from types import SimpleNamespace

import numpy as np

from badminton_analysis.model.detection import Detection, DetectionModel
from badminton_analysis.model.detection.inference import YOLODetector


class ArrayValue:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)

    def cpu(self) -> "ArrayValue":
        return self

    def tolist(self) -> list[object]:
        return self.values.tolist()


def create_result(
    boxes: list[list[float]],
    confidences: list[float],
    class_ids: list[int],
) -> SimpleNamespace:
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=ArrayValue(boxes),
            conf=ArrayValue(confidences),
            cls=ArrayValue(class_ids),
        )
    )


class FakeYOLO:
    names = {0: "person", 1: "sports ball"}

    def __init__(self, results: list[SimpleNamespace]) -> None:
        self.results = results
        self.predict_arguments: dict[str, object] = {}

    def predict(self, **arguments: object) -> list[SimpleNamespace]:
        self.predict_arguments = arguments
        return self.results


class YOLODetectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((16, 16, 3), dtype=np.uint8)

    def test_detect_converts_model_boxes_to_domain_detections(self) -> None:
        result = create_result(
            boxes=[[1, 2, 10, 15], [3, 4, 7, 8]],
            confidences=[0.91, 0.80],
            class_ids=[0, 1],
        )
        model = FakeYOLO([result])
        detector = YOLODetector(model=model, confidence_threshold=0.5)

        detections = detector.detect(self.image)

        self.assertIsInstance(detector, DetectionModel)
        self.assertEqual(
            detections,
            [
                Detection((1.0, 2.0, 10.0, 15.0), 0.91, 0, "person"),
                Detection((3.0, 4.0, 7.0, 8.0), 0.80, 1, "sports ball"),
            ],
        )
        self.assertEqual(model.predict_arguments["conf"], 0.5)
        self.assertNotIn("classes", model.predict_arguments)

    def test_detect_batch_returns_detections_per_image(self) -> None:
        first_result = create_result([[1, 1, 5, 5]], [0.8], [0])
        second_result = SimpleNamespace(boxes=None)
        model = FakeYOLO([first_result, second_result])
        detector = YOLODetector(model=model)

        detections = detector.detect_batch([self.image, self.image])

        self.assertEqual(len(detections), 2)
        self.assertEqual(len(detections[0]), 1)
        self.assertEqual(detections[1], [])


if __name__ == "__main__":
    unittest.main()
