from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from badminton_analysis.model.detection.interface import Detection, DetectionModel
from badminton_analysis.model.settings import get_settings





@dataclass
class YOLODetector(DetectionModel):
    """Detect objects using an Ultralytics YOLO model."""

    model: YOLO = field(
        default_factory=lambda: YOLO(get_settings().detection_model_name),
        repr=False,
    )
    confidence_threshold: float = get_settings().detection_confidence

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.detect_batch([image])[0]

    def detect_batch(
        self,
        images: list[np.ndarray],
    ) -> list[list[Detection]]:
        results = self.model.predict(
            source=images,
            conf=self.confidence_threshold,
            verbose=False,
        )
        return [self._convert_the_detection(result) for result in results]


    def _convert_the_detection(self, result: Results) -> list[Detection]:
        if result.boxes is None:
            return []

        detections: list[Detection] = []
        box_coordinates = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()

        for bbox, confidence, class_id_value in zip(
            box_coordinates,
            confidences,
            class_ids,
            strict=True,
        ):
            class_id = int(class_id_value)
            x_min, y_min, x_max, y_max = (float(value) for value in bbox)
            detections.append(
                Detection(
                    bbox=(x_min, y_min, x_max, y_max),
                    confidence=float(confidence),
                    class_id=class_id,
                    class_name=self.model.names[class_id],
                )
            )
        return detections

